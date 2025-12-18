import numpy as np
import cvxpy as cvx
import jax


class LinearizedMPC:
    def __init__(self, dynamics, N, dt, nx, nu, Q, R):
        self.N = N
        self.dt = dt
        self.nx = nx
        self.nu = nu

        self.Jac_x = jax.jit(jax.jacrev(lambda x, u: dynamics(x, u), argnums=0))
        self.Jac_u = jax.jit(jax.jacrev(lambda x, u: dynamics(x, u), argnums=1))

        self.setup_ocp(Q, R)

        self.use_warm_start = False

    def setup_ocp(self, Q, R):
        self.Q = Q
        self.R = R

        self.x0 = cvx.Parameter(self.nx, "x0")
        self.x_ref = cvx.Parameter(self.nx, "x_ref")
        self.u_max = cvx.Parameter(self.nu, "u_max", value=200.0 * np.ones(self.nu))
        self.x_hat = cvx.Parameter(
            (self.N + 1, self.nx), "x_hat", value=np.zeros((self.N + 1, self.nx))
        )
        self.u_hat = cvx.Parameter(
            (self.N, self.nu), "u_hat", value=np.zeros((self.N, self.nu))
        )
        self.A = cvx.Parameter(
            (self.N, self.nx, self.nx), "A", value=np.zeros((self.N, self.nx, self.nx))
        )
        self.B = cvx.Parameter(
            (self.N, self.nx, self.nu), "B", value=np.zeros((self.N, self.nx, self.nu))
        )
        self.dx = cvx.Variable((self.N + 1, self.nx), "dx")
        self.du = cvx.Variable((self.N, self.nu), "du")

        cost = 0.0
        constraints = [self.x_hat[0, :] + self.dx[0, :] == self.x0]
        for i in range(self.N):
            cost += cvx.quad_form(self.x_hat[i, :] + self.dx[i, :] - self.x_ref, self.Q)
            cost += cvx.quad_form(self.u_hat[i, :] + self.du[i, :], self.R)
            constraints += [
                self.dx[i + 1, :]
                == self.A[i, :, :] @ self.dx[i, :] + self.B[i, :, :] @ self.du[i, :]
            ]
            constraints += [cvx.abs(self.u_hat[i, :] + self.du[i, :]) <= self.u_max]
            constraints += [cvx.abs(self.dx[i, :]) <= 5.0]
        cost += cvx.quad_form(
            self.x_hat[self.N, :] + self.dx[self.N, :] - self.x_ref, self.Q
        )

        self.ocp = cvx.Problem(cvx.Minimize(cost), constraints)

    def solve_ocp(self, x, x_ref, num_iterations=None, verbose=True, tolerance=1e-3):
        self.x0.value = x
        self.x_ref.value = x_ref

        if self.use_warm_start:
            self.x_hat.value = np.vstack(
                [self.x_hat.value[1:, :], self.x_hat.value[-1:, :]]
            )
            self.u_hat.value = np.vstack(
                [self.u_hat.value[1:, :], np.zeros((1, self.nu))]
            )
            if num_iterations is None:
                num_iterations = 1
        else:
            self.x_hat.value = np.tile(x, (self.N + 1, 1))
            self.u_hat.value = np.zeros(self.u_hat.shape)
            self.use_warm_start = True
            if num_iterations is None:
                num_iterations = 50

        for iteration in range(num_iterations):
            for i in range(self.N):
                self.A.value[i, :, :] = np.asarray(
                    self.Jac_x(self.x_hat.value[i, :], self.u_hat.value[i, :])
                )
                self.B.value[i, :, :] = np.asarray(
                    self.Jac_u(self.x_hat.value[i, :], self.u_hat.value[i, :])
                )

            self.dx.value = np.zeros(self.dx.shape)
            self.du.value = np.zeros(self.du.shape)

            solve_status = self.ocp.solve(solver=cvx.OSQP)
            print(np.linalg.norm(self.x_hat.value), np.linalg.norm(self.u_hat.value))
            print(np.linalg.norm(self.A.value), np.linalg.norm(self.B.value))
            print(np.linalg.norm(self.dx.value), np.linalg.norm(self.du.value))

            du_norm = np.linalg.norm(self.du.value)
            if verbose:
                print(
                    f"Iteration {iteration+1}: ||du|| = {du_norm}, status: {solve_status}"
                )

            cost_mpc_sol = self.ocp.value
            x_mpc_sol = self.x_hat.value + self.dx.value
            u_mpc_sol = self.u_hat.value + self.du.value

            if du_norm < tolerance:
                break

            self.x_hat.value = x_mpc_sol
            self.u_hat.value = u_mpc_sol
            self.dx.value = np.zeros(self.dx.shape)
            self.du.value = np.zeros(self.du.shape)

        return x_mpc_sol, u_mpc_sol, cost_mpc_sol
