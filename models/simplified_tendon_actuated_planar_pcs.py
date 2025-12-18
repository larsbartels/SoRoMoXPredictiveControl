__all__ = ["SimplifiedTendonActuatedPlanarPCS"]
from jax import Array, lax, vmap
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple

from soromox.utils.integration import scale_gaussian_quadrature
import soromox.utils.lie_algebra as lie
from soromox.systems.tendon_actuated_planar_pcs import TendonActuatedPlanarPCS


class SimplifiedTendonActuatedPlanarPCS(TendonActuatedPlanarPCS):
    """
    Tendon-driven Planar Piecewise Constant Strain (PCS) model for 2D soft continuum robots.
    Neglecting Coriolis force in the dynamics for simplified computations.

    Refer to TendonActuatedPlanarPCS from SoRoMoX for detailed documentation.

    """

    @eqx.filter_jit
    def _J_local_tips(self, q: Array) -> Array:
        """
        Compute the body-frame Jacobian and its time derivative at the tips of all segments.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).

        Returns:
            J_local_tips (Array): Jacobian and at each segment tip of shape (num_segments, 3, num_strains).
        """
        xi = self.strain(q).reshape(self.num_segments, 3)

        zeros = jnp.zeros((self.num_segments, 3, 3), dtype=xi.dtype)

        Ad_inv_tips = vmap(
            lambda xi_i, L_i: lie.Adjoint_gi_se2_inv(xi_i, L_i, eps=self.global_eps)
        )(xi, self.L)
        T_tips = vmap(
            lambda xi_i, L_i: lie.Tangent_gi_se2(xi_i, L_i, eps=self.global_eps)
        )(xi, self.L)

        def scan_body(
            carry: Array,
            i: Array,
        ) -> Tuple[Array, Array]:
            J_prev = carry

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
        Compute the body-frame Jacobian at a batch of arc-length positions.

        Args:
            q (Array): generalized coordinates of shape (num_active_strains,).
            s_ps (Array): arc-length positions in [0, L] of shape (N,).

        Returns:
            J_local_ps (Array): Jacobians evaluated at all points of shape (N, 3, num_strains).
        """
        xi = self.strain(q).reshape(self.num_segments, 3)

        segment_indices, s_local_ps = vmap(self.classify_segment)(s_ps)

        # J_tips, _ = self._J_Jd_local_tips(q, qd)
        J_tips = self._J_local_tips(q)
        zeros_like_tip = jnp.zeros_like(J_tips[:1])

        J_bases = jnp.concatenate([zeros_like_tip, J_tips[:-1]], axis=0)

        J_base_ps = J_bases[segment_indices]

        xi_ps = xi[segment_indices]

        N = s_ps.shape[0]
        J_base_ps = J_base_ps.reshape(N, 3, self.num_segments, 3).transpose(0, 2, 1, 3)

        def integrate_segment(
            i: Array, xi_i: Array, arc_len: Array, J_base: Array
        ) -> Array:
            Ad_inv = lie.Adjoint_gi_se2_inv(xi_i, arc_len, eps=self.global_eps)
            T = lie.Tangent_gi_se2(xi_i, arc_len, eps=self.global_eps)

            J_rot = jnp.einsum("ij, njk->nik", Ad_inv, J_base)
            J_next = J_rot.at[i].set(Ad_inv @ T)

            return J_next

        J_local_ps = vmap(integrate_segment)(
            segment_indices, xi_ps, s_local_ps, J_base_ps
        )

        J_local_ps = vmap(self._final_size_jacobian)(J_local_ps)

        return J_local_ps

    @eqx.filter_jit
    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: Optional[Tuple] = None
    ) -> Array:
        """
        Simplified forward dynamics function. Neglect Coriolis forces.

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

        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(self.Xs, self.Ws, self.L_cum[:-1], self.L_cum[1:])

        chi_ps = self.forward_kinematics_batched(q, Xs_scaled.flatten())
        g_ps = vmap(lie.exp_SE2)(chi_ps.reshape(-1, 3))
        g_ps = g_ps.reshape(self.num_segments, self.num_gauss_points, 3, 3)

        # J_ps, _ = self._J_Jd_local_batched(q, qd, Xs_scaled.flatten())  # jax.jit is smart enough to not bother computing Jd
        J_ps = self._J_local_batched(q, Xs_scaled.flatten())
        J_ps = J_ps.reshape(self.num_segments, self.num_gauss_points, *J_ps.shape[1:])

        def dynamical_matrices_i(i: Array) -> Tuple[Array, Array, Array]:
            M_i = self._local_mass_matrix(i)

            def dynamical_matrices_ij(j: Array) -> Tuple[Array, Array, Array]:
                Ws_ij = Ws_scaled[i][j]
                g_ij = g_ps[i, j]
                J_ij = J_ps[i, j]

                Ad_g_inv_ij = lie.Adjoint_g_inv_SE2(g_ij)

                B_ij = Ws_ij * J_ij.T @ M_i @ J_ij
                G_ij = -Ws_ij * J_ij.T @ M_i @ Ad_g_inv_ij @ self.g

                return B_ij, G_ij

            return vmap(dynamical_matrices_ij)(jnp.arange(1, self.num_gauss_points - 1))

        B_blocks_tot, G_blocks_tot = vmap(dynamical_matrices_i)(
            jnp.arange(self.num_segments)
        )

        B_full = jnp.sum(B_blocks_tot, axis=(0, 1))
        G_full = jnp.sum(G_blocks_tot, axis=(0, 1))

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
