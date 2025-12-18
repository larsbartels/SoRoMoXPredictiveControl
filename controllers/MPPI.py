import jax
from jax import numpy as jnp
import equinox
from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    Tsit5,
    Euler,
    PIDController,
    ImplicitEuler,
)
from time import perf_counter


class ModelPredictivePathIntegralControl:
    def __init__(
        self,
        dynamics,
        N,
        dt,
        nu,
        Q,
        R,
        x_ref,
        u_min=None,
        u_max=None,
        lam=1,
        sigma=100,
        correlation=0.0,
        covariance_type="Gaussian",
    ):
        self.dynamics = lambda t, x, control_input: dynamics(t, x, control_input(t))
        self.N = N
        self.dt = dt
        self.T = N * dt
        self.nu = nu

        self.Q = Q
        self.R = R
        self.u_min = u_min
        self.u_max = u_max
        self.x_ref = x_ref.reshape(-1, 1)

        self.rng_key = jax.random.PRNGKey(0)
        self.K = 32  # number of samples
        self.lam_inv = 1 / lam  # inverse temperature

        self.mu = jnp.zeros((self.nu, self.N))
        if covariance_type == "independent" or correlation == 0.0:
            self.Sigma = jnp.eye(self.N)
        elif covariance_type == "Toeplitz" or correlation == 1.0:
            self.Sigma = jnp.fromfunction(
                lambda j, i: correlation ** abs(i - j), (self.N, self.N)
            )
        elif covariance_type == "Gaussian":
            self.Sigma = jnp.exp(
                -0.5
                * (jnp.subtract.outer(jnp.arange(self.N), jnp.arange(self.N))) ** 2
                / (-0.5 / jnp.log(correlation))
            )
        else:
            raise ValueError(
                f"Unknown covariance_type '{covariance_type}'. Options are 'independent', 'Toeplitz', and 'Gaussian'."
            )
        self.Sigma = jnp.stack([sigma * self.Sigma + 1e-6 * jnp.eye(self.N)] * self.nu)

        self.batch_compute_trajectory_cost = jax.jit(
            lambda u_batch, x0: jax.vmap(lambda u: self.compute_trajectory_cost(u, x0))(
                u_batch
            )
        )

    @equinox.filter_jit
    def compute_trajectory_cost(self, u, x0, t0=0.0, solver_dt=1e-4):
        u = u.T
        control_input = lambda t: jnp.clip(
            u[jnp.floor((t - t0) / self.dt).astype(int)], min=self.u_min, max=self.u_max
        )

        term = ODETerm(self.dynamics)
        saveat = SaveAt(ts=(t0 + self.dt * jnp.arange(self.N + 1)))
        sol = diffeqsolve(
            terms=term,
            solver=Euler(),
            t0=t0,
            t1=t0 + self.T + solver_dt,
            dt0=solver_dt,
            y0=x0,
            args=control_input,
            saveat=saveat,
        )

        x = sol.ys.reshape(self.N + 1, -1, 1)
        u = u.reshape(self.N, -1, 1)
        J = jnp.sum((x - self.x_ref).mT @ self.Q @ (x - self.x_ref), axis=0) + jnp.sum(
            u.mT @ self.R @ u, axis=0
        )

        return J

    @equinox.filter_jit
    def sample_u(self, mu, Sigma, add_zero_sample=True, rng_key=jax.random.PRNGKey(0)):
        u = jax.random.multivariate_normal(
            key=rng_key, mean=mu, cov=Sigma, shape=(self.K, self.nu)
        )
        if add_zero_sample:
            return jnp.concatenate([mu[None, :], u], axis=0)
        return u

    @equinox.filter_jit
    def update_distribution(self, u, J, update_variance=False):
        w = jnp.exp(-self.lam_inv * (J - jnp.min(J)))

        m = jnp.sum(w * u, axis=0) / jnp.sum(w, axis=0)

        if update_variance:
            du = (u - m)[:, :, :, None]
            S = jnp.sum(w[:, :, :, None] * du @ du.mT, axis=0) / jnp.sum(w)
            # S = 0.8 * self.Sigma + 0.2 * S
        else:
            S = self.Sigma

        return m, S, w

    def solve(self, x0, num_iter=1):
        compute_times = {"sampling": 0.0, "trajectory cost": 0.0, "update": 0.0}
        for iter in range(num_iter):
            self.rng_key, rng_key = jax.random.split(self.rng_key)

            tic = perf_counter()
            u = self.sample_u(self.mu, self.Sigma, rng_key=rng_key)
            compute_times["sampling"] += perf_counter() - tic
            if not jnp.isfinite(u).all():
                print("u: ", u.squeeze())
                print(jnp.isfinite(self.mu).all(), jnp.isfinite(self.Sigma).all())
                print(jnp.linalg.eigvals(self.Sigma))
                raise ValueError()

            tic = perf_counter()
            J = self.batch_compute_trajectory_cost(u, x0)
            compute_times["trajectory cost"] += perf_counter() - tic

            tic = perf_counter()
            valid_samples_idx = jnp.isfinite(J.squeeze())
            if not jnp.all(valid_samples_idx):
                print("WARNING: non-finite cost detected")
            self.mu, self.Sigma, w = self.update_distribution(
                u[valid_samples_idx, :, :], J[valid_samples_idx, :, :]
            )
            compute_times["update"] += perf_counter() - tic
            if not jnp.isfinite(w).all():
                print(jnp.isfinite(w).all(), jnp.isfinite(self.mu).all())
                print(w.squeeze(), J.squeeze() - jnp.min(J))
                raise ValueError()

        return compute_times

    @property
    def control_input(self):
        return jnp.clip(self.mu, self.u_min, self.u_max)

    @equinox.filter_jit
    def predict_trajectory(self, u, x0, t0=0.0, solver_dt=2e-4):
        u = u.T

        term = ODETerm(
            lambda t, x, _: self.dynamics(
                t, x, u[jnp.floor((t - t0) / self.dt).astype(int)]
            )
        )
        saveat = SaveAt(ts=(t0 + self.dt * jnp.arange(self.N + 1)))
        sol = diffeqsolve(
            terms=term,
            solver=Euler(),
            t0=t0,
            t1=t0 + self.T + solver_dt,
            dt0=solver_dt,
            y0=x0,
            saveat=saveat,
            max_steps=None,
        )

        return sol.ys

    def get_control_input_and_step(self):
        u = self.control_input

        self.mu = jnp.concatenate([u[:, 1:], jnp.zeros_like(u[:, -1:])], axis=1)
        # self.Sigma = jax.scipy.linalg.block_diag(self.Sigma[nu:, nu:], 100 * jnp.eye(nu))

        return u[:, 0]

    def auto_choose_temperature(self, x=None):
        if x is None:
            x = self.x_ref.flatten()
        u = self.sample_u(self.mu, self.Sigma)
        J = jax.vmap(lambda u: self.compute_trajectory_cost(u, x))(u)

        delta_J = jnp.max(J) - jnp.min(J)
        self.lam_inv = (
            100.0 / delta_J
        )  # 100 is a somewhat arbitrary choice to give a reasonable spread of cost
        print(delta_J, self.lam_inv)
