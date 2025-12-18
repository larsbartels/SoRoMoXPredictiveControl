__all__ = ["SimplifiedTendonActuatedPCS"]
from jax import Array, lax, vmap
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple

from soromox.utils.integration import scale_gaussian_quadrature
import soromox.utils.lie_algebra as lie
from soromox.systems.tendon_actuated_pcs import TendonActuatedPCS


class SimplifiedTendonActuatedPCS(TendonActuatedPCS):
    """
    Tendon-driven Piecewise Constant Strain (PCS) model for 3D soft continuum robots.
    Neglecting Coriolis force in the dynamics for simplified computations.

    Refer to TendonActuatedPCS from SoRoMoX for detailed documentation.

    """

    @eqx.filter_jit
    def _J_local_tips(self, q: Array) -> Array:
        """
        Compute the Jacobian for the forward kinematics at all segment tips.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            J_local_tips (Array): Jacobian of the forward kinematics at point s, shape (num_segments, 6, num_strains)
        """
        # compute the strain and strain rate for each segment
        xi = self.strain(q).reshape(self.num_segments, 6)

        # precompute the Lie algebra expressions for all segment tips
        Ad_inv_tips = vmap(
            lambda xi_i, L_i: lie.Adjoint_gi_se3_inv(xi_i, L_i, eps=self.global_eps)
        )(xi, self.L)
        T_tips = vmap(
            lambda xi_i, L_i: lie.Tangent_gi_se3(xi_i, L_i, eps=self.global_eps)
        )(xi, self.L)

        # initialize zeros
        zeros = jnp.zeros((self.num_segments, 6, 6), dtype=xi.dtype)

        def scan_body(carry: Array, i: Array) -> Tuple[Array, Array]:
            J_prev = carry

            # extract the current segment variables
            Ad_inv_i = lax.dynamic_index_in_dim(Ad_inv_tips, i, axis=0, keepdims=False)
            T_i = lax.dynamic_index_in_dim(T_tips, i, axis=0, keepdims=False)

            J_rot = jnp.einsum("ij, njk->nik", Ad_inv_i, J_prev)
            J_next = J_rot.at[i].set(Ad_inv_i @ T_i)

            return J_next, J_next

        indices = jnp.arange(self.num_segments, dtype=jnp.int32)
        _, J_local_tips = lax.scan(scan_body, zeros, indices)

        J_local_tips = vmap(self._final_size_jacobian)(J_local_tips)

        return J_local_tips

    @eqx.filter_jit
    def _J_local_batched(self, q: Array, s_ps: Array) -> Array:
        """
        Compute the Jacobian for the forward kinematics at a batch of points s_ps along the robot.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): point coordinates along the robot in the interval [0, L] of shape (N,).

        Returns:
            J_local_ps (Array): Jacobians evaluated at all points, shape (N, 6, num_strains)
        """
        # num points
        N = s_ps.shape[0]

        # compute the strain and strain rate for each segment
        xi = self.strain(q).reshape(self.num_segments, 6)

        # classify all points along the robot to the corresponding segment and compute local coordinates
        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)

        # compute the Jacobian at the tips
        J_tips = self._J_local_tips(q)  # shape (num_segments, 6, num_strains)

        # select the base Jacobian for each point (g0 for the first, previous tip otherwise)
        J_bases = jnp.concatenate([jnp.zeros_like(J_tips[:1]), J_tips[:-1]], axis=0)

        J_base_ps = J_bases[segment_indices]
        # select the other variables for each point
        xi_ps = xi[segment_indices]
        idx_ps = segment_indices

        # reshape for easier indexing
        J_base_ps = J_base_ps.reshape(N, 6, self.num_segments, 6).transpose(
            0, 2, 1, 3
        )  # shape (N, 6, num_segments, 6)

        def integrate_segment(
            i: Array,
            xi_i: Array,
            s_local: Array,
            J_base: Array,
        ) -> Tuple[Array, Array]:
            Ad_inv = lie.Adjoint_gi_se3_inv(xi_i, s_local, eps=self.global_eps)
            T = lie.Tangent_gi_se3(xi_i, s_local, eps=self.global_eps)

            J_rot = jnp.einsum("ij, njk->nik", Ad_inv, J_base)
            J_next = J_rot.at[i].set(Ad_inv @ T)

            return J_next

        # vmap the segment integration over all points
        J_local_ps = vmap(integrate_segment)(idx_ps, xi_ps, s_local_ps, J_base_ps)

        # reshape back to (N, 6, num_strains)
        J_local_ps = vmap(self._final_size_jacobian)(J_local_ps)

        return J_local_ps

    @eqx.filter_jit
    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: Optional[Tuple] = None
    ) -> Array:
        """
        Simplified forward dynamics function. Neglects Coriolis forces.

        Args:
            t (Array): Current time.
            y (Array): State vector containing configuration and velocity.
                Shape is (2 * num_strains,).
            actuation_args (Tuple, optional): Additional arguments for the actuation mapping function.
                Default is None.
        Returns:
            yd: Time derivative of the state vector.
        """
        # Split the state vector into configuration and velocity
        q, qd = jnp.split(y, 2)

        # split the actuation arguments if provided
        if actuation_args is None:
            u, tau_ext = None, None
        elif len(actuation_args) == 1:
            u = actuation_args[0]
            tau_ext = None
        elif len(actuation_args) == 2:
            u, tau_ext = actuation_args
        else:
            raise ValueError("actuation_args must be a tuple of length 1 or 2.")

        if u is None:
            u = jnp.zeros((self.num_actuators,))
        if tau_ext is None:
            tau_ext = jnp.zeros((q.shape[-1],))

        # compute the gauss quadrature points and weights for each segment
        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(
            self.Xs, self.Ws, self.L_cum[:-1], self.L_cum[1:]
        )  # shape (num_segments, num_gauss_points) for both Xs_scaled and Ws_scaled

        # compute the forward kinematics for each quadrature point
        g_ps = self.forward_kinematics_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 4, 4)
        g_ps = g_ps.reshape(
            self.num_segments, self.num_gauss_points, 4, 4
        )  # shape (num_segments, num_gauss_points, 4, 4)

        # compute the jacobian for each quadrature point
        # jax.jit is smart enough to not bother computing Jd
        # J_ps, _ = self._J_Jd_local_batched(q, qd, Xs_scaled.flatten())  # shape (num_segments * num_gauss_points, 6, num_active_strains)
        J_ps = self._J_local_batched(
            q, Xs_scaled.flatten()
        )  # shape (num_segments * num_gauss_points, 6, num_active_strains)
        J_ps = J_ps.reshape(
            self.num_segments, self.num_gauss_points, *J_ps.shape[1:]
        )  # shape (num_segments, num_gauss_points, 6, num_active_strains)

        def dynamical_matrices_i(i: Array) -> Tuple[Array, Array, Array]:
            """
            Compute the integrand for the dynamical matrices at the i-th segment.
            Args:
                i (Array): index of the segment
            Returns:
                Tuple[Array, Array, Array]: The inertia matrix, Coriolis matrix, and gravitational force integrands.
            """
            M_i = self._local_mass_matrix(i)

            def dynamical_matrices_ij(j: Array) -> Tuple[Array, Array, Array]:
                """
                Compute the integrand for the dynamical matrices at the j-th quadrature point of the i-th segment.
                Args:
                    j (Array): index of the quadrature point
                Returns:
                    Tuple[Array, Array, Array]: The inertia matrix, Coriolis matrix, and gravitational force integrands.
                """
                # select the j-th quadrature weight
                Ws_ij = Ws_scaled[i][j]
                # select the j-th Cartesian pose
                g_ij = g_ps[i, j]
                # select the j-th jacobian and its time-derivative
                J_ij = J_ps[i, j]

                # compute the lie algebra expressions.
                Ad_g_inv_ij = lie.Adjoint_g_inv_SE3(g_ij)

                # compute the inertia matrix integrand
                B_ij = Ws_ij * J_ij.T @ M_i @ J_ij

                # compute the gravitational force integrand
                G_ij = -Ws_ij * J_ij.T @ M_i @ Ad_g_inv_ij @ self.g

                return B_ij, G_ij

            # we can skip the first and last quadrature points since their weight is zero
            B_blocks_i, G_blocks_i = vmap(dynamical_matrices_ij)(
                jnp.arange(1, self.num_gauss_points - 1)
            )

            return B_blocks_i, G_blocks_i

        # compute the dynamical matrices for each segment
        B_blocks_tot, G_blocks_tot = vmap(dynamical_matrices_i)(
            jnp.arange(self.num_segments)
        )

        # sum over segments and Gauss points
        B_full = jnp.sum(B_blocks_tot, axis=(0, 1))
        G_full = jnp.sum(G_blocks_tot, axis=(0, 1))

        # construct the dynamical matrices
        B = self.B_xi.T @ B_full @ self.B_xi
        G = self.B_xi.T @ G_full
        D = self.damping_matrix(q)
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)

        B_inv = jnp.linalg.inv(B)  # Inverse of the inertia matrix
        qdd = B_inv @ (
            tau_u + tau_ext - G - tau_el - D @ qd
        )  # Compute the acceleration

        yd = jnp.concatenate([qd, qdd])

        return yd
