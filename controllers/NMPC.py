import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cvx
import diffrax


class NMPC:
    """
    Nonlinear Model Predictive Controller (NMPC) class. Using SQP method to solve the OCP at each time step.
    """

    def __init__(
        self,
        horizon: int,
        time_step: float,
        dynamics: callable,
        output: callable,
        nx: int,
        nu: int,
        ny: int,
    ):
        self.N = horizon
        self.dt = time_step
        self.nx = nx
        self.nu = nu
        self.ny = ny

        self.f = jax.jit(lambda x, u: x + self.dt * dynamics(x, u))
        # self.f = jax.jit(lambda x, u: diffrax.diffeqsolve(diffrax.ODETerm(system.forward_dynamics), diffrax.Tsit5()))
        self.Jf_x = jax.jit(jax.jacfwd(self.f, argnums=0))
        self.Jf_u = jax.jit(jax.jacfwd(self.f, argnums=1))
        self.h = jax.jit(lambda x: output(x))
        self.Jh_x = jax.jit(jax.jacfwd(self.h))

        self.setup_ocp()

    def setup_ocp(self):
        self.dx = cvx.Variable((self.N + 1, self.nx))
        self.du = cvx.Variable((self.N, self.nu))
        self.y_bar = cvx.Variable((self.N + 1, self.ny))
        self.x_hat = cvx.Parameter(
            (self.N + 1, self.nx), value=np.zeros((self.N + 1, self.nx))
        )
        self.u_hat = cvx.Parameter((self.N, self.nu), value=np.zeros((self.N, self.nu)))
        self.f_hat = cvx.Parameter((self.N, self.nx), value=np.zeros((self.N, self.nx)))
        self.df_x = cvx.Parameter(
            (self.N, self.nx, self.nx), value=np.zeros((self.N, self.nx, self.nx))
        )
        self.df_u = cvx.Parameter(
            (self.N, self.nx, self.nu), value=np.zeros((self.N, self.nx, self.nu))
        )
        self.h_hat = cvx.Parameter(
            (self.N + 1, self.ny), value=np.zeros((self.N + 1, self.ny))
        )
        self.dh_x = cvx.Parameter(
            (self.N + 1, self.ny, self.nx),
            value=np.zeros((self.N + 1, self.ny, self.nx)),
        )
        self.u_min = cvx.Parameter((self.nu,), value=-10 * np.ones((self.nu,)))
        self.u_max = cvx.Parameter((self.nu,), value=10 * np.ones((self.nu,)))
        self.y_ref = cvx.Parameter(
            (self.N + 1, self.ny), value=np.zeros((self.N + 1, self.ny))
        )

        self.Q = np.eye(self.nx)
        self.R = 0.001 * np.eye(self.nu)

        cost = 0.0
        for i in range(self.N):
            cost += cvx.quad_form(self.x_hat[i, :] + self.dx[i, :], self.Q)
            cost += cvx.quad_form(self.u_hat[i, :] + self.du[i, :], self.R)
        cost += cvx.quad_form(self.y_bar[self.N, :], self.Q)

        constraints = [self.dx[0, :] == 0.0]
        for i in range(self.N):
            constraints += [
                self.x_hat[i + 1, :] + self.dx[i + 1, :]
                == self.x_hat[i, :]
                + self.dt
                * (
                    self.f_hat[i, :]
                    + self.df_x[i, :, :] @ self.dx[i, :]
                    + self.df_u[i, :, :] @ self.du[i, :]
                )
            ]
            constraints += [
                self.y_bar[i, :]
                == self.h_hat[i, :]
                + self.dh_x[i, :, :] @ self.dx[i, :]
                - self.y_ref[i, :]
            ]
            # constraints += [self.u_min <= self.u_hat[i, :] + self.du[i, :], self.u_hat[i, :] + self.du[i, :] <= self.u_max]

        self.ocp = cvx.Problem(cvx.Minimize(cost), constraints)

    def solve_ocp(
        self,
        x: jnp.ndarray,
        y_ref: jnp.ndarray,
        sqp_iterations: int = 5,
        verbose=False,
        u_cand=None,
    ):
        if u_cand is not None:
            self.u_hat.value = u_cand
        x_init = [x]
        for i in range(self.N):
            x_init.append(self.f(x_init[-1], self.u_hat.value[i, :]))
        self.x_hat.value = np.array(x_init)
        if y_ref.shape == (self.ny,):
            y_ref = np.repeat(y_ref[None, :], self.N + 1, axis=0)
        self.y_ref.value = y_ref

        for iter in range(sqp_iterations):
            self.dx.value = np.zeros((self.N + 1, self.nx))
            self.du.value = np.zeros((self.N, self.nu))
            for i in range(self.N):
                self.f_hat.value[i, :] = np.array(
                    self.f(self.x_hat.value[i, :], self.u_hat.value[i, :])
                )
                self.df_x.value[i, :, :] = np.array(
                    self.Jf_x(self.x_hat.value[i, :], self.u_hat.value[i, :])
                )
                self.df_u.value[i, :, :] = np.array(
                    self.Jf_u(self.x_hat.value[i, :], self.u_hat.value[i, :])
                )
                self.h_hat.value[i, :] = np.array(self.h(self.x_hat.value[i, :]))
                self.dh_x.value[i, :, :] = np.array(self.Jh_x(self.x_hat.value[i, :]))
            self.h_hat.value[self.N, :] = np.array(self.h(self.x_hat.value[self.N, :]))
            self.dh_x.value[self.N, :, :] = np.array(
                self.Jh_x(self.x_hat.value[self.N, :])
            )

            self.ocp.solve(solver=cvx.OSQP)

            self.x_hat.value += self.dx.value
            self.u_hat.value += self.du.value

            if verbose:
                print(
                    f"SQP iteration {iter}, cost: {self.ocp.value}, ||du||: {np.linalg.norm(self.du.value)}"
                )
            if np.linalg.norm(self.du.value) < 1e-6:
                break

        x_sol = self.x_hat.value
        u_sol = self.u_hat.value

        self.x_hat.value = np.vstack(
            [
                self.x_hat.value[1:, :],
                self.f(self.x_hat.value[-1, :], self.u_hat.value[-1, :]),
            ]
        )
        self.u_hat.value = np.vstack([self.u_hat.value[1:, :], self.u_hat.value[-1, :]])

        return x_sol, u_sol

    def compute_trajectory_cost(self, x, u, x0, y_ref):
        if x is None:
            x = [x0]
            for i in range(self.N):
                x.append(self.f(x[-1], u[i]))
        y = [None] * (self.N + 1)
        for i in range(self.N + 1):
            y[i] = self.h(x[i])
        y_hat = np.array(y - y_ref).reshape(self.N + 1, self.ny, 1)
        u = np.array(u).reshape(self.N, self.nu, 1)
        cost = np.sum(y_hat.mT @ self.Q @ y_hat, axis=0) + np.sum(
            u.mT @ self.R @ u, axis=0
        )
        return cost
