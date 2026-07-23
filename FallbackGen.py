import numpy as np
from scipy.special import ellipk

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

        self.N_QUAD = 96
        self.chunk_size = 50_000

        # Gauss-Legendre nodes for both radial and polar integrals.
        # The cosine substitution (Eq. 2.26) makes the T_r integrand smooth
        # on [0, pi], so GL is appropriate for both.
        x_gl, w_gl = np.polynomial.legendre.leggauss(self.N_QUAD)
        self._chi_nodes   = 0.5 * np.pi * (1.0 + x_gl)   # (N_QUAD,) on [0, pi]
        self._chi_weights = 0.5 * np.pi * w_gl

        self._compute_dT()

    def geodesic_kerr_s2(self, τ, y):
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

    def _four_roots_batched(self, E_flat, Lz_flat, Q_flat):
        """
        Returns shape (N, 4) sorted descending (r1 > r2 > r3 > r4).
        Unbound (E >= 1) or invalid rows are NaN.
        Chunked to avoid peak memory from large batched eigensolver.
        """
        N   = E_flat.size
        c4, c3, c2, c1, c0 = self._R_coeffs(E_flat, Lz_flat, Q_flat)

        bound = (E_flat < 1.0) 
        roots_out = np.full((N, 4), np.nan)
        if not bound.sum():
            return roots_out

        c4b = c4[bound]; c3b = c3[bound]
        c2b = c2[bound]; c1b = c1[bound]; c0b = c0[bound]
        n_b = bound.sum()

        chunk = 5_000
        real_roots_all = np.full((n_b, 4), np.nan)

        for start in range(0, n_b, chunk):
            sl = slice(start, start + chunk)
            nc = c4b[sl].size

            C = np.zeros((nc, 4, 4))
            C[:, 1, 0] = 1.0
            C[:, 2, 1] = 1.0
            C[:, 3, 2] = 1.0
            C[:, 0, 3] = -c0b[sl] / c4b[sl]
            C[:, 1, 3] = -c1b[sl] / c4b[sl]
            C[:, 2, 3] = -c2b[sl] / c4b[sl]
            C[:, 3, 3] = -c3b[sl] / c4b[sl]

            eigs = np.linalg.eigvals(C)
            real_mask  = np.abs(eigs.imag) < 1e-6 * (np.abs(eigs.real) + 1.0)
            real_roots = np.where(real_mask, eigs.real, np.nan)
            real_roots = -np.sort(-real_roots, axis=1)

            n_real = real_mask.sum(axis=1)
            r2_col = real_roots[:, 1]

            # r_horizon for Kerr
            r_horizon = 1.0 + np.sqrt(1.0 - self.a**2)
            valid = (n_real == 4) & (r2_col > r_horizon)
            # valid  = (n_real == 4) & (r2_col > 0.0)
            real_roots_all[sl] = np.where(valid[:, None], real_roots, np.nan)

            pct = min(100, int((start + chunk) / n_b * 100))
            print(f"  roots chunk {start//chunk + 1}/{(n_b+chunk-1)//chunk} ({pct}%)", end='\r')

        print()
        roots_out[bound] = real_roots_all
        return roots_out

    def _Lambda_r(self, E, r1, r2, r3, r4):
        d13 = r1 - r3;  d24 = r2 - r4
        with np.errstate(divide='ignore', invalid='ignore'):
            k_sq = np.where((d13 > 0) & (d24 > 0),
                            (r1 - r2) * (r3 - r4) / (d13 * d24), np.nan)

        ok  = np.isfinite(k_sq) & (k_sq > 0.0) & (k_sq < 1.0 - 1e-14)
        K_r = np.where(ok, ellipk(np.clip(k_sq, 0.0, 1.0 - 1e-14)), np.nan)
        with np.errstate(invalid='ignore'):
            Ups = np.pi * np.sqrt(np.abs((1.0 - E**2) * d13 * d24)) / (2.0 * K_r)
            Lr  = 2.0 * np.pi / Ups
        return np.where(np.isfinite(Lr) & (Lr > 0), Lr, np.nan)

    def _avg_Tr(self, E, Lz, Q, r1, r2, r3, r4, Lambda_r):
        """
        <T_r> via cosine substitution (Eq. 2.26-2.27):
            r(chi) = 0.5*(r1+r2) + 0.5*(r1-r2)*cos(chi),  chi in [0, pi]

        dr/dchi = -half*sin(chi)
        sqrt(R) = |half|*sin(chi)*sqrt(-c4*(r-r3)*(r-r4))

        The sin(chi) and |half| cancel between dr/dchi and sqrt(R), giving
        a smooth integrand T_r / sqrt(-c4*(r-r3)*(r-r4)) on [0, pi].
        The factor of 2 accounts for the full radial period (rp->ra->rp),
        since the cosine substitution covers only one half (rp->ra).
        GL quadrature is appropriate for the smooth integrand.
        """
        a   = self.a
        chi = self._chi_nodes    # (N_QUAD,) GL nodes on [0, pi]
        w   = self._chi_weights  # (N_QUAD,) GL weights

        # r(chi): shape (N, N_QUAD)
        mid  = 0.5 * (r1 + r2)                               # (N,)
        half = 0.5 * (r1 - r2)                               # (N,)
        r_q  = mid[:, None] + half[:, None] * np.cos(chi)    # (N, N_QUAD)

        # T_r(r) = (r^2 + a^2) * P / Delta,  P = E*(r^2+a^2) - a*Lz
        E_  = E[:, None];  Lz_ = Lz[:, None]
        P       = E_ * (r_q**2 + a**2) - a * Lz_
        Delta   = np.maximum(r_q**2 - 2.0*r_q + a**2, 1e-300)
        T_r_val = (r_q**2 + a**2) * P / Delta                # (N, N_QUAD)

        # After cancellation of |half|*sin(chi), the integrand is:
        # T_r / sqrt(-c4*(r-r3)*(r-r4))
        c4   = E**2 - 1.0                                     # (N,) negative
        r_r3 = r_q - r3[:, None]
        r_r4 = r_q - r4[:, None]
        denom = np.sqrt(np.maximum(-c4[:, None] * r_r3 * r_r4, 0.0)) + 1e-300

        integrand = T_r_val / denom                           # (N, N_QUAD)

        # Factor of 2: cosine sub covers rp->ra (half period); full period = 2x
        integral = 2.0 * np.einsum('iq,q->i', integrand, w)  # (N,)
        return integral / Lambda_r

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

        # ---- 1: roots via chunked companion matrix eigensolver ----
        print("  Finding roots (chunked eigensolver) ...")
        roots = self._four_roots_batched(E_flat, Lz_flat, Q_flat)
        r1, r2, r3, r4 = roots[:, 0], roots[:, 1], roots[:, 2], roots[:, 3]
        valid = np.isfinite(r1) & np.isfinite(r2) & (r2 > 0.0)
        print(f"  Bound: {(E_flat < 1.0).sum():,} / {N_total:,}")
        print(f"  Valid roots: {valid.sum():,}")

        separatrix = valid & ((r2 - r3) < 1e-6 * r2)
        if separatrix.sum() > 0:
            print(f"  Warning: {separatrix.sum()} particles near separatrix (r2≈r3), excluding")
            valid &= ~separatrix

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

        n_chunks = max(1, (n_val + self.chunk_size - 1) // self.chunk_size)
        print(f"  Quadrature: {n_chunks} chunks ...")

        for k, start in enumerate(range(0, n_val, self.chunk_size)):
            sl  = idx[start : start + self.chunk_size]
            Ec  = E_flat[sl];  Lzc = Lz_flat[sl]
            Qc  = Q_flat[sl];  Lrc = Lambda_r[sl]
            r1c = r1[sl];  r2c = r2[sl]
            r3c = r3[sl];  r4c = r4[sl]

            avg_Tr[sl]     = self._avg_Tr(Ec, Lzc, Qc, r1c, r2c, r3c, r4c, Lrc)

            pct = min(100, int((k + 1) / n_chunks * 100))
            print(f"    chunk {k+1}/{n_chunks}  ({pct}%)", end='\r')

        print()

        # ---- 5 & 6: T_r = Gamma * Lambda_r  (Eq. 2.31) ----
        Gamma = avg_Tr + self.a * Lz_flat - self.a**2 * E_flat
        T_r   = Gamma * Lambda_r
        T_r   = np.where((T_r > 0) & np.isfinite(T_r), T_r, np.nan)
        print(f"  Successful T_r: {np.isfinite(T_r).sum():,} / {N_total:,}")

        self.dTs         = T_r.reshape(orig_shape)
        self.Lambda_r    = Lambda_r.reshape(orig_shape)
        self.avg_Tr_arr  = avg_Tr.reshape(orig_shape)