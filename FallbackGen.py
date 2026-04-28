import pygyre as pg
import numpy as np
import scipy as cp
import matplotlib.pyplot as plt
from scipy.special import ellipk
from scipy.integrate import fixed_quad

class FallbackGen:
    "Does dT calculation offline"

    def __init__(self, mass_ratio, a, rtde, rp, E, Lz, Q, dE, dLz, dQ, N):
        self.dE = dE
        self.dLz = dLz
        self.dQ = dQ
        self.a = a
        self.R_TDE = rtde
        self.Rp = rp
        self.OrbitEnergy = E
        self.mom = Lz
        self.Carter = Q
        self.mass_ratio = mass_ratio
        self.N = N

        self.total_E = self.dE + np.ones(self.dE.shape) * self.OrbitEnergy
        self.total_Q = self.dQ + np.ones(self.dLz.shape) * self.Carter
        self.total_Lz = self.dLz + np.ones(self.dLz.shape) * self.mom

        Omegap = 1.5 * np.sqrt((1 + self.mass_ratio) / (2 * self.Rp**3))
        t_ini  = self.N * (0.012 / Omegap)
        self.t = np.linspace(-t_ini, t_ini, self.N)
        self.obs_t = np.zeros(len(self.t))

        self.N_QUAD = 96  # Gauss-Legendre quadrature points
        self.chunk_size = 500_000
        # Pre-compute Gauss-Legendre nodes/weights on [0, pi] once
        x_gl, w_gl        = np.polynomial.legendre.leggauss(self.N_QUAD)
        self._chi_nodes   = 0.5 * np.pi * (1.0 + x_gl)   # (N_QUAD,)
        self._chi_weights = 0.5 * np.pi * w_gl            # (N_QUAD,)

        self._compute_dT()

    def geodesic_kerr_s2(self, τ,  y):
        """
        Testing new geodesic function based off of Kesden 2012.
        """
        t, r = y

        E = self.bound_E
        a = self.a
        Lz = self.bound_Lz
        theta = np.pi / 2

        sigma = r**2 + a**2 * np.cos(theta)**2
        delta = r**2 + a**2 - 2 * r
        alpha = (r**2 + a**2)**2 - delta * a**2 * np.sin(theta)**2

        rad = self.R_of_r(r)
        
        if rad <= 0:
            self.radial_sign *= -1

        dt_dτ = ((alpha * E - 2 * a * r * Lz) / delta) / sigma
        dr_dτ = self.radial_sign * np.sqrt(rad) / sigma
        # dφ_dτ = (Lz * np.csc(theta)**2 + (2 * a * r * E - a**2 * Lz) / delta) / sigma
        # dθ_dτ = np.sqrt(q - Lz**2 * np.cot(theta)**2 - a**2 * (1 - E**2) * np.cos(theta)**2) / sigma
        # dpsi_dτ = np.abs(a - Lz) * (((r**2 + a**2) - a * Lz) / ((a - Lz)**2 + r**2) + a * (Lz - a) / (a - Lz)**2) / r**2

        return [dt_dτ, dr_dτ]
    
    def _R_coeffs(self, E, Lz, Q):
        a   = self.a
        aE  = a * E
        aLz = a * Lz
        c4  =  E**2 - 1.0
        c3  =  2.0 * np.ones_like(E)
        c2  =  2.0 * E * (E * a**2 - aLz) - (Lz - aE)**2 - Q - a**2
        c1  =  2.0 * ((Lz - aE)**2 + Q)
        c0  =  (E * a**2 - aLz)**2 - a**2 * ((Lz - aE)**2 + Q)
        return c4, c3, c2, c1, c0

    # ------------------------------------------------------------------
    # Root-finding via batched companion matrix
    #
    # The companion matrix of a monic degree-4 poly p(r)/c4 is:
    #
    #   C = [[ 0,  0,  0, -c0/c4 ],
    #        [ 1,  0,  0, -c1/c4 ],
    #        [ 0,  1,  0, -c2/c4 ],
    #        [ 0,  0,  1, -c3/c4 ]]
    #
    # np.linalg.eigvals accepts a stack of matrices (..., 4, 4) and
    # returns all eigenvalues in one LAPACK call — much faster than
    # looping np.roots() over each particle.
    # ------------------------------------------------------------------

    def _four_roots_batched(self, E_flat, Lz_flat, Q_flat):
        """
        Returns shape (N, 4) sorted descending (r1 > r2 > r3 > r4).
        Unbound (E >= 1) or invalid rows are NaN.
        """
        N   = E_flat.size
        c4, c3, c2, c1, c0 = self._R_coeffs(E_flat, Lz_flat, Q_flat)

        # Only process bound particles
        bound = E_flat < 1.0
        n_b   = bound.sum()
        roots_out = np.full((N, 4), np.nan)

        if n_b == 0:
            return roots_out

        # Monic coefficients for bound particles
        c4b = c4[bound];  c3b = c3[bound]
        c2b = c2[bound];  c1b = c1[bound];  c0b = c0[bound]

        # Build stacked companion matrices: shape (n_b, 4, 4)
        C = np.zeros((n_b, 4, 4))
        C[:, 1, 0] = 1.0
        C[:, 2, 1] = 1.0
        C[:, 3, 2] = 1.0
        C[:, 0, 3] = -c0b / c4b
        C[:, 1, 3] = -c1b / c4b
        C[:, 2, 3] = -c2b / c4b
        C[:, 3, 3] = -c3b / c4b

        # Batch eigenvalue solve — returns (n_b, 4) complex eigenvalues
        eigs = np.linalg.eigvals(C)   # single LAPACK call

        # Keep real roots
        real_mask = np.abs(eigs.imag) < 1e-6 * (np.abs(eigs.real) + 1.0)
        real_roots = np.where(real_mask, eigs.real, np.nan)

        # Sort descending (NaN sorts to end in np.sort by default for float)
        real_roots = -np.sort(-real_roots, axis=1)   # descending

        # Validate: need 4 real roots and r2 (pericenter) > 0
        n_real = real_mask.sum(axis=1)               # (n_b,)
        r2_col = real_roots[:, 1]
        valid  = (n_real == 4) & (r2_col > 0.0)

        roots_out[bound] = np.where(valid[:, None], real_roots, np.nan)
        return roots_out

    # ------------------------------------------------------------------
    # Lambda_r: Mino-time radial period  (Fujita Eq. 15)
    # k_r^2 = (r1-r2)(r3-r4) / [(r1-r3)(r2-r4)]
    # ------------------------------------------------------------------

    def _Lambda_r(self, E, r1, r2, r3, r4):
        d13 = r1 - r3;  d24 = r2 - r4
        with np.errstate(divide='ignore', invalid='ignore'):
            k_sq = np.where((d13 > 0) & (d24 > 0),
                            (r1 - r2) * (r3 - r4) / (d13 * d24), np.nan)
        ok  = np.isfinite(k_sq) & (k_sq > 0.0) & (k_sq < 1.0)
        K_r = np.where(ok, ellipk(np.clip(k_sq, 0.0, 1.0 - 1e-12)), np.nan)
        with np.errstate(invalid='ignore'):
            Ups = np.pi * np.sqrt(np.abs((1.0 - E**2) * d13 * d24)) / (2.0 * K_r)
            Lr  = 2.0 * np.pi / Ups
        return np.where(np.isfinite(Lr) & (Lr > 0), Lr, np.nan)

    # ------------------------------------------------------------------
    # <T_r>_lambda  (Fujita Eq. 7, radial term)
    # Cosine substitution: r = mid + half*cos(chi), chi in [0, pi]
    # ------------------------------------------------------------------

    def _avg_Tr(self, E, Lz, Q, r1, r2, Lambda_r):
        a   = self.a
        chi = self._chi_nodes;  w = self._chi_weights

        mid  = 0.5 * (r1 + r2)
        half = 0.5 * (r1 - r2)
        r_q  = mid[:, None] + half[:, None] * np.cos(chi)[None, :]

        E_  = E[:, None];  Lz_ = Lz[:, None];  Q_ = Q[:, None]

        P_r     = E_ * (r_q**2 + a**2) - a * Lz_
        Delta_r = r_q**2 - 2.0 * r_q + a**2
        T_r     = (r_q**2 + a**2) * P_r / Delta_r

        c4, c3, c2, c1, c0 = self._R_coeffs(E_, Lz_, Q_)
        R_r = ((((c4 * r_q + c3) * r_q + c2) * r_q + c1) * r_q + c0)
        R_r = np.maximum(R_r, 0.0)

        integ = half[:, None] * np.sin(chi)[None, :] * T_r / np.sqrt(R_r + 1e-300)
        return 2.0 * np.einsum('iq,q->i', integ, w) / Lambda_r

    # ------------------------------------------------------------------
    # <T_theta>_lambda  (Fujita Eq. 7, polar term)
    # For Q ~ 0: returns -a^2 * E exactly
    # For Q > 0: integrates over polar turning points
    # T_theta(u) = -a^2 E (1 - u^2),  Theta(u) = Q - B u^2 + A u^4
    # A = a^2(1-E^2),  B = Q + A + Lz^2
    # ------------------------------------------------------------------

    def _avg_Ttheta(self, E, Lz, Q):
        a   = self.a
        chi = self._chi_nodes;  w = self._chi_weights

        result = np.full(E.shape, np.nan)

        eq  = np.abs(Q) < 1e-12 * (1.0 + np.abs(E))
        result[eq] = -a**2 * E[eq]

        inc = ~eq & (Q > 0.0)
        if not np.any(inc):
            result[~eq] = -a**2 * E[~eq]
            return result

        Ei  = E[inc];  Lzi = Lz[inc];  Qi = Q[inc]
        A_c = a**2 * (1.0 - Ei**2)
        B_c = Qi + A_c + Lzi**2
        disc = B_c**2 - 4.0 * A_c * Qi
        pv  = np.isfinite(disc) & (disc >= 0.0) & (A_c > 1e-20)

        z_m = np.where(pv,
            (B_c - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * np.maximum(A_c, 1e-30)), 0.0)
        z_m = np.maximum(z_m, 0.0)
        z_p = np.where(pv,
            (B_c + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * np.maximum(A_c, 1e-30)), 1.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            k_sq = np.where(pv & (z_p > 1e-20), z_m / z_p, 0.0)
        k_sq   = np.clip(k_sq, 0.0, 1.0 - 1e-12)
        K_th   = ellipk(k_sq)
        eps0   = A_c / np.maximum(Lzi**2, 1e-30)
        Ups_th = np.pi * np.abs(Lzi) * np.sqrt(eps0 * z_p) / (2.0 * K_th)
        Lam_th = 2.0 * np.pi / np.maximum(Ups_th, 1e-30)

        sz  = np.sqrt(z_m)[:, None]
        u_q = 0.5 * sz * (1.0 + np.cos(chi)[None, :])

        T_th = -a**2 * Ei[:, None] * (1.0 - u_q**2)
        Th_q = Qi[:, None] - B_c[:, None] * u_q**2 + A_c[:, None] * u_q**4
        Th_q = np.maximum(Th_q, 0.0)

        integ  = 0.5 * sz * np.sin(chi)[None, :] * T_th / np.sqrt(Th_q + 1e-300)
        I_th   = np.einsum('iq,q->i', integ, w)
        result[inc] = np.where(pv, 4.0 * I_th / Lam_th, -a**2 * Ei)

        bad = ~np.isfinite(result)
        result[bad] = -a**2 * E[bad]
        return result

    # ------------------------------------------------------------------
    # Master computation
    # ------------------------------------------------------------------

    def _compute_dT(self):
        orig_shape = self.dE.shape
        N_total    = self.dE.size

        E_flat  = self.total_E.ravel()
        Lz_flat = self.total_Lz.ravel()
        Q_flat  = self.total_Q.ravel()

        print(f"Computing radial periods for {N_total:,} particles ...")
        print(f"  E  range: [{E_flat.min():.6f}, {E_flat.max():.6f}]")
        print(f"  Q  range: [{Q_flat.min():.3e}, {Q_flat.max():.3e}]")
        print(f"  chunk_size = {self.chunk_size:,}")

        # ---- 1: roots via batched companion matrix eigensolver ----
        print("  Finding roots (batched eigensolver) ...")
        roots = self._four_roots_batched(E_flat, Lz_flat, Q_flat)
        r1, r2, r3, r4 = roots[:, 0], roots[:, 1], roots[:, 2], roots[:, 3]
        valid = np.isfinite(r1) & np.isfinite(r2) & (r2 > 0.0)
        print(f"  Bound: {(E_flat < 1.0).sum():,} / {N_total:,}")
        print(f"  Valid roots: {valid.sum():,}")

        # ---- 2: Lambda_r ----
        Lambda_r = np.full(N_total, np.nan)
        if valid.sum() > 0:
            Lambda_r[valid] = self._Lambda_r(
                E_flat[valid], r1[valid], r2[valid], r3[valid], r4[valid])
        lam_valid = valid & np.isfinite(Lambda_r)
        print(f"  Valid Lambda_r: {lam_valid.sum():,}")

        # ---- 3 & 4: <T_r> and <T_theta> in memory-safe chunks ----
        idx   = np.where(lam_valid)[0]
        n_val = len(idx)
        avg_Tr     = np.full(N_total, np.nan)
        avg_Ttheta = np.full(N_total, np.nan)

        n_chunks = (n_val + self.chunk_size - 1) // self.chunk_size
        print(f"  Quadrature: {n_chunks} chunks ...")

        for k, start in enumerate(range(0, n_val, self.chunk_size)):
            sl  = idx[start : start + self.chunk_size]
            Ec  = E_flat[sl];  Lzc = Lz_flat[sl]
            Qc  = Q_flat[sl];  Lrc = Lambda_r[sl]
            r1c = r1[sl];      r2c = r2[sl]

            avg_Tr[sl]     = self._avg_Tr(Ec, Lzc, Qc, r1c, r2c, Lrc)
            avg_Ttheta[sl] = self._avg_Ttheta(Ec, Lzc, Qc)

            pct = min(100, int((k + 1) / n_chunks * 100))
            print(f"    chunk {k+1}/{n_chunks}  ({pct}%)", end='\r')

        print()   # newline after \r progress

        # ---- 5 & 6: T_r = Gamma * Lambda_r ----
        Gamma = avg_Tr + avg_Ttheta + self.a * Lz_flat
        T_r   = Gamma * Lambda_r
        T_r   = np.where((T_r > 0) & np.isfinite(T_r), T_r, np.nan)
        print(f"  Successful T_r: {np.isfinite(T_r).sum():,} / {N_total:,}")

        dTs = T_r.reshape(orig_shape)

        self.dTs = dTs

        # store intermediates
        self.Lambda_r    = Lambda_r.reshape(orig_shape)
        self.avg_Tr_arr  = avg_Tr.reshape(orig_shape)
        self.avg_Tth_arr = avg_Ttheta.reshape(orig_shape)