import pygyre as pg
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


class TDECalculator:
    """
    Calculates the time and radius of tidal disruption (TDE) for a star
    orbiting a supermassive black hole, plus random‐sampling diagnostics
    at the moment of disruption.
    """
    def __init__(self, star_name, MBH=1e6, Rp=17, N=1000):
        """
        Parameters:
        -----------
        star_name : str
            Directory with GYRE/MESA outputs, e.g. 'MAMS1Msun'.
        MBH : float
            Black hole mass in solar masses.
        Rp : float
            Pericenter distance in units of GM_BH.
        N : int
            Number of time steps for the orbit integration.
        """
        self.star_name = star_name
        self.MBH       = MBH
        self.Rp        = Rp
        self.N         = N

        # ——— Read stellar structure ———
        self.summary = pg.read_output(f"star-files/{star_name}/summary.h5")
        self.detail0 = pg.read_output(f"star-files/{star_name}/detail.l2.n+0.h5")
        self.r   = self.detail0['x']          # radial grid (in R_star units)
        self.rho = self.detail0['rho']        # density (g/cm^3)
        M_r       = self.detail0['M_r']       # enclosed mass profile

        # Convert quantities to dimensionless units
        self.Mstar      = self.summary['M_star'][0] / 1.9884e33  # solar masses
        self.mass_ratio = self.Mstar / self.MBH
        self.Rstar      = (self.summary['R_star'][0]/100) / (self.MBH*1476.55)
        self.Rtidal     = self.Rstar / self.mass_ratio**(1/3)

        # Interpolate gamma from external table
        Masses, Gammas = np.loadtxt('M-gamma.txt', unpack=True)
        self.gamma = np.interp(self.Mstar, Masses, Gammas)

        # Gravitational binding energy U (in G M_star^2 / R_star)
        self.U = np.trapz(
            4*np.pi*self.r * self.rho * M_r,
            self.r
        ) * self.summary['R_star'][0]**2 / (self.summary['M_star'][0]**2 / self.summary['R_star'][0])

        # ——— Time grid & orbit ———
        Omegap = 1.5 * np.sqrt((1 + self.mass_ratio) / (2 * self.Rp**3))
        t_ini  = self.N * (0.012 / Omegap)
        self.t = np.linspace(-t_ini, t_ini, self.N)
        self._compute_orbit(Omegap)

        # ——— Modes & overlaps ———
        self.omega  = self.summary['omega'].real * np.sqrt(self.mass_ratio / self.Rstar**3)
        self._load_mode_details()
        self._compute_tidal_field()
        self._compute_overlaps()
        self._compute_greens_functions()
        self._compute_quadrupoles()

        # ——— Mode amplitudes & displacements ———
        self._compute_mode_amplitudes()

        # ——— Tidal energy & TDE detection ———
        self._compute_tidal_energy()
        self._detect_tde_indices()

    def _compute_orbit(self, Omegap):
        """
        Compute the orbital phase (Phi), radius (R), and their time derivatives
        for a parabolic trajectory parameterized by the pericenter frequency Omegap.
        """
        t = self.t  # time array centered on pericenter in GM_BH units

        # Change‐of‐variables Xi(t) solves the Keplerian motion implicitly
        Xi = (Omegap * t + np.sqrt(1 + (Omegap * t)**2))**(1/3)

        # Orbital phase Φ(t) from the parametric variable Xi
        self.Phi = 2 * np.arctan(Xi - 1/Xi)

        # Instantaneous radius R(t) = Rp * (Xi^2 - 1 + Xi^-2)
        self.R = self.Rp * (Xi**2 - 1 + Xi**(-2))

        # Time derivative dXi/dt from chain rule
        Xidot = Omegap * Xi / (3 * np.sqrt(1 + (Omegap * t)**2))

        # Radial velocity Ṙ = dR/dt
        self.Rdot = 2 * self.Rp * (Xi**4 - 1) * Xidot / Xi**3

        # Angular velocity ϕ̇ = dΦ/dt
        self.phidot = 2 * (1 + Xi**2) * Xidot / (1 - Xi**2 + Xi**4)

    def _load_mode_details(self):
        """
        Load GYRE detail files for each pseudo‐mode labeled by n_pg.
        Stores a list of dicts, each containing 'xi_r', 'xi_h', etc.
        """
        self.details = []
        # summary['n_pg'] gives radial mode orders (negative for g‐modes, 0 for f, positive for p)
        for n in self.summary['n_pg']:
            # Format tag like 'n+0', 'n-1', etc.
            tag = f"n{int(n):+d}"
            path = f"star-files/{self.star_name}/detail.l2.{tag}.h5"
            # Read the HDF5 and append to the details list
            self.details.append(pg.read_output(path))

    def _compute_tidal_field(self):
        """
        Assemble the tidal tensor E_ab(t) and its time derivative dE_ab/dt
        due to the SMBH's gravitational field at the star's center.
        """
        N = self.N
        # Initialize arrays: E[a,b,i] and dEdt[a,b,i]
        E = np.zeros((3, 3, N))
        dEdt = np.zeros_like(E)

        # Precompute cos(Φ) and sin(Φ)
        c, s = np.cos(self.Phi), np.sin(self.Phi)
        R, Rdot, phidot = self.R, self.Rdot, self.phidot

        # Tidal tensor components in the principal orbital frame:
        # E_xx = (1 - 3 cos^2Φ) / R^3
        E[0,0] = (1 - 3 * c**2) / R**3
        # E_yy = (1 - 3 sin^2Φ) / R^3
        E[1,1] = (1 - 3 * s**2) / R**3
        # E_zz =       1          / R^3
        E[2,2] = 1 / R**3
        # Off‐diagonal E_xy = E_yx = -3 sinΦ cosΦ / R^3
        E[0,1] = E[1,0] = -3 * s * c / R**3

        # Now compute time derivatives dE_ab/dt:
        # Use product and chain rules on R(t) and Φ(t).
        dEdt[0,0] = -3*(1 - 3*c**2)*Rdot/R**4 +  6*c*s*phidot/R**3
        dEdt[1,1] = -3*(1 - 3*s**2)*Rdot/R**4 -  6*c*s*phidot/R**3
        dEdt[2,2] = -3 * Rdot / R**4
        # Mixed derivative for E_xy:
        dEdt[0,1] = dEdt[1,0] = (9*c*s*Rdot - 3*np.cos(2*self.Phi)*R*phidot) / R**4

        # Store the results on the instance
        self.E, self.dEdt = E, dEdt


    def _compute_overlaps(self):
        """
        Compute mode overlap integrals N and D, then form quadrupole overlaps Q
        and normalization factors xi for each normal mode.
        """
        # Unpack density and radial grid
        rho, r = self.rho, self.r
        # Stellar mass M and radius R from summary (in CGS units)
        M, R   = self.summary['M_star'][0], self.summary['R_star'][0]
        # Conversion factor to dimensionless GM_BH units
        conv   = 1000 * (R/100)**3 / (M/1000)

        Qs, xis = [], []
        # Loop over each mode’s detail dict and its frequency ω
        for det, omega in zip(self.details, self.omega):
            # Lagrangian eigenfunctions: radial and horizontal displacements
            xi_r = det['xi_r'].real    # shape (nr,)
            xi_h = det['xi_h'].real    # shape (nr,)

            # Compute the normalization integral N = ∫ ρ (r^2 ξ_r^2 + 6 r^2 ξ_h^2) dr
            N_ = np.trapz(rho * (r**2 * xi_r**2 + 6 * (r * xi_h)**2), r) * conv

            # Compute the coupling integral D = ∫ ρ (r^3 ξ_r + 3 r^3 ξ_h) dr
            D_ = np.trapz(rho * (r**3 * xi_r + 3 * r**3 * xi_h), r) * conv

            # Quadrupole overlap Q = D^2 / N
            Qs.append(D_**2 / N_)
            # Mode normalization xi = D / N
            xis.append(D_ / N_)

        # Store arrays of length n_modes
        self.overlap_Q  = np.array(Qs)   # Q[m] for each mode m
        self.overlap_xi = np.array(xis)  # ξ[m] for each mode m

    def _compute_greens_functions(self):
        """
        Build retarded Green’s functions for the quadrupole response:
          - Gret_Q : full set of modes
          - Gret_Q_g : g-modes only
          - Gret_Q_f : f-mode only
          - Gret_Q_p : p-modes only
        Also build single-mode function Gret_q(i).
        """
        omega = self.omega
        # Overall prefactor (16π/15) q R_star^2
        factor = (16 * np.pi / 15) * self.mass_ratio * self.Rstar**2

        def G_base(indices):
            """
            Return a function G(Δt) that sums over the selected mode indices.
            """
            omegam = omega[indices]             # frequencies of selected modes
            Qm = self.overlap_Q[indices]  # corresponding Q overlaps
            invomega = 1.0 / omegam
            def G(Delta_t):
                # Delta_t is an array of lags
                # sinm[k,j] = sin(ωm[k] * Delta_t[j])
                sinm = np.sin(omegam[:, None] * Delta_t[None, :])
                # Weighted sum: ∑_k Qm[k] * invω[k] * sinm[k,j]
                return (Qm[:, None] * invomega[:, None] * sinm).sum(axis=0) * factor
            return G

        # Indices for all, g-, f-, and p-modes
        all_idx = np.arange(len(omega))
        g_idx   = [i for i, n in enumerate(self.summary['n_pg']) if n < 0]
        f_idx   = [i for i, n in enumerate(self.summary['n_pg']) if n == 0]
        p_idx   = [i for i, n in enumerate(self.summary['n_pg']) if n > 0]

        # Build each Green’s function
        self.Gret_Q   = G_base(all_idx)
        self.Gret_Q_g = G_base(g_idx)
        self.Gret_Q_f = G_base(f_idx)
        self.Gret_Q_p = G_base(p_idx)

        def single(i):
            """
            Return the single-mode Green’s function G_i(Δt) for mode index i.
            G_i(Δt) = ξ[i] * sin(ω[i] Δt) / ω[i]
            """
            qi, omegai = self.overlap_xi[i], omega[i]
            return lambda Delta_t: qi * np.sin(omegai * Delta_t) / omegai

        # Store single-mode builder
        self.Gret_q = single


    def _compute_quadrupoles(self):
        """
        Convolve the retarded Green’s functions with the tidal field to
        obtain the time‐dependent quadrupole tensor Q_ab(t) for each mode set.
        """
        t, E, N = self.t, self.E, self.N

        def build_Q(G):
            # Allocate Q[a,b,i] array: 3×3 tensor at each time step
            Q = np.zeros((3, 3, N))

            # For each time index i>0, integrate over past history
            for i in range(1, N):
                # Compute time lags Δt_j = t[i] - t[j] for j=0..i
                Delta = t[i] - t[:i+1]

                # Integrate tensor components
                for a, b in [(0,0), (1,1), (2,2), (0,1)]:
                    # Multiply the Green’s response G(Δt) by the tidal drive E[a,b](t')
                    integrand = G(Delta) * E[a, b, :i+1]
                    # Numerically integrate via trapezoidal rule
                    Q_val = -np.trapz(integrand, x=t[:i+1])
                    # Store result
                    Q[a, b, i] = Q_val
                    # Enforce symmetry: Q[b,a] = Q[a,b]
                    if a != b:
                        Q[b, a, i] = Q_val

            return Q

        # Build quadrupole responses for:
        #  - all modes, g‑modes, f‑mode, and p‑modes separately
        self.Q_all = build_Q(self.Gret_Q)
        self.Q_g   = build_Q(self.Gret_Q_g)
        self.Q_f   = build_Q(self.Gret_Q_f)
        self.Q_p   = build_Q(self.Gret_Q_p)

    def _compute_mode_amplitudes(self):
        """
        Compute individual mode amplitudes q_ab(t) by convolving each
        single‐mode Green’s function with the tidal field, then build
        the total Lagrangian displacements xi_r and xi_h.
        """
        omega, t, N = self.omega, self.t, self.N
        n_modes = len(omega)

        # Allocate q[a,b,i,j]: coupling coefficients for each mode j
        self.q = np.zeros((3, 3, N, n_modes))

        # 1) Compute q_ab,i,j = -∫ G_j(Δt) E_ab(t') dt' for each mode
        for i in range(1, N):
            Delta = t[i] - t[:i+1]
            for j in range(n_modes):
                Gij = self.Gret_q(j)(Delta)  # single-mode Green’s function
                for a, b in [(0,0), (1,1), (2,2), (0,1)]:
                    val = -np.trapz(Gij * self.E[a, b, :i+1], x=t[:i+1])
                    self.q[a, b, i, j] = val
                    # symmetry q[b,a] = q[a,b]
                    self.q[b, a, i, j] = val

        # 2) Sum over modes to build xi_r and xi_h displacements
        nr = len(self.r)
        self.xi_r = np.zeros((3, 3, N, nr))
        self.xi_h = np.zeros((3, 3, N, nr))

        for i in range(1, N):
            for j, det in enumerate(self.details):
                xr = det['xi_r'].real  # radial eigenfunction
                xh = det['xi_h'].real  # horizontal eigenfunction
                for a, b in [(0,0), (1,1), (2,2), (0,1)]:
                    # Accumulate weighted eigenfunctions
                    self.xi_r[a, b, i] += self.q[a, b, i, j] * xr
                    self.xi_h[a, b, i] += self.q[a, b, i, j] * xh
                # Ensure symmetry in displacement tensors
                self.xi_r[1, 0, i] = self.xi_r[0, 1, i]
                self.xi_h[1, 0, i] = self.xi_h[0, 1, i]

    def _compute_tidal_energy(self):
        """
        Integrate the work done by the tidal field to find the cumulative
        tidal energy deposited in the star over time.
        """
        t, N = self.t, self.N

        # Instantaneous power: Q_ab * (dE_ab/dt), summed over tensor indices
        power = (
            self.Q_all[0,0] * self.dEdt[0,0]
          + self.Q_all[1,1] * self.dEdt[1,1]
          + self.Q_all[2,2] * self.dEdt[2,2]
          + 2 * self.Q_all[0,1] * self.dEdt[0,1]
        )

        # 1) Cumulative integral of power: ∫_0^t power dt'
        TE_cumulative = 0.5 * integrate.cumulative_trapezoid(power, t, initial=0)

        # 2) Subtract instantaneous potential term 0.5 Q_ab E_ab
        instant_term = 0.5 * (
            self.Q_all[0,0] * self.E[0,0]
          + self.Q_all[1,1] * self.E[1,1]
          + self.Q_all[2,2] * self.E[2,2]
          + 2 * self.Q_all[0,1] * self.E[0,1]
        )

        # Store net tidal energy
        self.TidalEnergy = TE_cumulative - instant_term

    def _detect_tde_indices(self):
        """
        Find the first time when the accumulated tidal energy exceeds the
        disruption threshold gamma * (U * q^2 / R_star).
        Interpolate to get precise t_TDE, and record R_TDE, Phi_TDE.
        """
        # Disruption criterion threshold
        threshold = self.gamma * (self.U * self.mass_ratio**2 / self.Rstar)

        # Identify indices where energy first surpasses threshold
        above = np.where(self.TidalEnergy > threshold)[0]
        if not len(above):
            raise RuntimeError("No TDE detected; tidal energy never exceeds threshold.")
        idx = above[0]

        # Linear interpolation between idx-1 and idx for precise t_TDE
        t0, t1 = self.t[idx-1], self.t[idx]
        E0, E1 = self.TidalEnergy[idx-1], self.TidalEnergy[idx]
        frac = (threshold - E0) / (E1 - E0)
        self.t_TDE = t0 + frac * (t1 - t0)

        # Record corresponding orbital radius & phase
        self.i_TDE = idx
        self.R_TDE = np.interp(self.t_TDE, self.t, self.R)
        self.Phi_TDE = np.interp(self.t_TDE, self.t, self.Phi)


    def whole_star_sample(self, N_Omega=300**2):
        """
        At the TDE moment, sample N_Omega random directions on the sphere,
        compute perturbed and unperturbed energies & orbital periods,
        and normalize them by DeltaE and DeltaT.
        Returns a dict with keys:
          'rr', 'directions', 
          'dEnergy_random', 'dT_random',
          'dEnergy_unperturbed', 'dT_unperturbed'
        """
        rho = self.rho
        # 1. Interpolate t_TDE quantities
        R_TDE   = self.R_TDE
        Phi_TDE = self.Phi_TDE
        i0, i1  = self.i_TDE - 1, self.i_TDE
        frac    = (self.t_TDE - self.t[i0]) / (self.t[i1] - self.t[i0])

        xi_r_TDE = (self.xi_r[:, :, i0]
                    + frac * (self.xi_r[:, :, i1] - self.xi_r[:, :, i0]))
        xi_h_TDE = (self.xi_h[:, :, i0]
                    + frac * (self.xi_h[:, :, i1] - self.xi_h[:, :, i0]))

        # 2. Sample random directions
        x, y, z = np.random.normal(size=(3, N_Omega))
        norm     = np.sqrt(x**2 + y**2 + z**2)
        n        = np.vstack((x, y, z)) / norm   # shape (3, N_Omega)

        # 3. Angular basis
        th = np.sqrt(1 - n[2]**2)
        dndtheta = np.vstack((
            n[0] * n[2] / th,
            n[1] * n[2] / th,
           -th
        ))
        dndphi = np.vstack((
           -n[1],
            n[0],
            np.zeros(N_Omega)
        ))

        # 4. Build radial grid and mass‐shells
        rr = (self.r[:-1] + self.r[1:]) / 2        # (nr-1,)
        dr = (self.r[1:]  - self.r[:-1])
        
        # compute dMass for each fluid element
        dMass_random = rr[:, None]**2 * dr[:, None] * (4 * np.pi / N_Omega) * (rho[:-1, None] + rho[1:, None]) / 2
        dMass_random = dMass_random * np.ones((1, N_Omega))

        # -- Displacements interpolation onto rr
        xi_r_interp = np.array([
            [np.interp(rr, self.r, xi_r_TDE[a, b])
             for b in range(3)]
            for a in range(3)
        ])  # shape (3,3,nr-1)
        xi_h_interp = np.array([
            [np.interp(rr, self.r, xi_h_TDE[a, b])
             for b in range(3)]
            for a in range(3)
        ])  # shape (3,3,nr-1)

        # 5. Compute total displacement xi[a, r, dir]
        xi = (
            np.einsum('ia,ja,ka,jkr->ira',   n,    n,    n,    xi_r_interp)
          + 2*np.einsum('ia,ja,ka,jkr->ira', dndtheta, n, dndtheta, xi_h_interp)
          + 2*np.einsum('ia,ja,ka,jkr,a->ira',
                        dndphi,    n, dndphi,    xi_h_interp,
                        1/(1 - n[2]**2))
        )  # shape (3, nr-1, N_Omega)

        # 6. Perturbed positions at TDE
        x_pos = rr[:, None] * n[0] + xi[0]
        y_pos = rr[:, None] * n[1] + xi[1]
        z_pos = rr[:, None] * n[2] + xi[2]

        # 7. Compute perturbed energy & period
        dist  = np.sqrt(
            (R_TDE * np.cos(Phi_TDE) + self.Rstar * x_pos)**2
          + (R_TDE * np.sin(Phi_TDE) + self.Rstar * y_pos)**2
          + (self.Rstar * z_pos)**2
        )
        dEnergy_random = -1/dist + 1/R_TDE
        dT_random = np.where(
            dEnergy_random < 0,
            2 * np.pi / np.abs(-2 * dEnergy_random)**1.5,
            0
        )

        # 8. Unperturbed motion
        dist0 = np.sqrt(
            (R_TDE * np.cos(Phi_TDE) + self.Rstar * rr[:, None] * n[0])**2
          + (R_TDE * np.sin(Phi_TDE) + self.Rstar * rr[:, None] * n[1])**2
          + (self.Rstar * rr[:, None] * n[2])**2
        )
        dEnergy_unperturbed = -1/dist0 + 1/R_TDE
        dT_unperturbed = np.where(
            dEnergy_unperturbed < 0,
            2 * np.pi / np.abs(-2 * dEnergy_unperturbed)**1.5,
            0
        )

        # 9. Normalize
        DeltaE = self.Rstar / self.Rp**2
        DeltaT = 1 / DeltaE**1.5

        dEnergy_random        /= DeltaE
        dEnergy_unperturbed   /= DeltaE
        dT_random             /= DeltaT
        dT_unperturbed        /= DeltaT

        return {
            'rr':                   rr,
            'directions':           n,
            'dEnergy_random':       dEnergy_random,
            'dT_random':            dT_random,
            'dEnergy_unperturbed':  dEnergy_unperturbed,
            'dT_unperturbed':       dT_unperturbed,
            'dMass':                dMass_random
        }
    
    def equator_sample(self, n_phi=3000):
        """
        Build a full 3D grid of debris positions at t_TDE, both
        unperturbed and mode‐perturbed, on a (r, θ, φ) mesh.

        Returns:
          POS_unperturbed: array shape (n_r, n_theta, n_phi, 3)
          POS:             same shape, with Lagrangian displacement added
        """
        rho = self.rho
        # --- Angular grid (cell boundaries) ---
        n_theta = 2 # makes it such that only the equator is sampled
        theta = np.linspace(0, np.pi, n_theta)
        phi   = np.linspace(0, 2*np.pi, n_phi)

        # Cell centers
        ttheta = 0.5 * (theta[:-1] + theta[1:])    # (n_theta,)
        pphi   = 0.5 * (phi[:-1] + phi[1:])        # (n_phi,)
        
        # Angular cell widths:
        dtheta = np.diff(theta)  # shape: (n_theta,)
        dphi   = np.diff(phi)    # shape: (n_phi,)

        # Radial cell centers and widths from self.r
        r_cell = 0.5 * (self.r[:-1] + self.r[1:])  # (n_r,)
        dr     = np.diff(self.r)                   # (n_r,)
        
        # Compute the average density in each radial cell.
        rho_avg = 0.5 * (rho[:-1] + rho[1:])  # shape: (n_r,)

        # Build 2D angular mesh
        T, P = np.meshgrid(ttheta, pphi, indexing='ij')  # (n_theta, n_phi)

        # Spherical unit vectors at each angular cell
        n_vec = np.stack([
            np.sin(T)*np.cos(P),
            np.sin(T)*np.sin(P),
            np.cos(T)
        ], axis=-1)                                   # (n_theta, n_phi, 3)
        dndtheta_vec = np.stack([
            np.cos(T)*np.cos(P),
            np.cos(T)*np.sin(P),
           -np.sin(T)
        ], axis=-1)                                   # (n_theta, n_phi, 3)
        dndphi_vec = np.stack([
           -np.sin(T)*np.sin(P),
            np.sin(T)*np.cos(P),
            np.zeros_like(T)
        ], axis=-1)                                   # (n_theta, n_phi, 3)

        # Interpolate xi_r and xi_h at self.t_TDE between indices i_TDE-1 and i_TDE
        i0, i1 = self.i_TDE-1, self.i_TDE
        frac   = (self.t_TDE - self.t[i0])/(self.t[i1]-self.t[i0])
        xi_r_TDE = self.xi_r[:, :, i0] + frac*(self.xi_r[:, :, i1] - self.xi_r[:, :, i0])
        xi_h_TDE = self.xi_h[:, :, i0] + frac*(self.xi_h[:, :, i1] - self.xi_h[:, :, i0])

        # Average over adjacent radial points for each mode component
        xi_r_avg = 0.5*(xi_r_TDE[..., :-1] + xi_r_TDE[..., 1:])  # (3,3,n_r)
        xi_h_avg = 0.5*(xi_h_TDE[..., :-1] + xi_h_TDE[..., 1:])  # (3,3,n_r)

        # Flatten angular and radial dims for einsum
        n_ang = T.size
        n_r   = r_cell.size
        n_vec_flat      = n_vec.reshape(n_ang, 3)
        dndtheta_flat   = dndtheta_vec.reshape(n_ang, 3)
        dndphi_flat     = dndphi_vec.reshape(n_ang, 3)
        T_flat          = T.reshape(n_ang)

        # Term1: radial displacement projection
        S1 = np.einsum('ma,mb,abi->mi', n_vec_flat, n_vec_flat, xi_r_avg)
        term1 = S1[:, :, None] * n_vec_flat[:, None, :]

        # Term2: tangential (theta) displacement
        S2 = np.einsum('ma,mb,abi->mi', n_vec_flat, dndtheta_flat, xi_h_avg)
        term2 = 2 * S2[:, :, None] * dndtheta_flat[:, None, :]

        # Term3: tangential (phi) displacement, with sin^2(θ) factor
        S3 = np.einsum('ma,mb,abi->mi', n_vec_flat, dndphi_flat, xi_h_avg)
        term3 = 2 * S3[:, :, None] * dndphi_flat[:, None, :] / (np.sin(T_flat)[:, None, None]**2)

        # Total displacement xi_flat: (n_ang, n_r, 3)
        xi_flat = term1 + term2 + term3

        # Reshape back to (n_r, n_theta, n_phi, 3)
        xi = xi_flat.reshape(ttheta.size, pphi.size, n_r, 3)
        xi = np.moveaxis(xi, 2, 0)

        # Unperturbed Cartesian positions on (r_cell, T, P) grid
        POS_unperturbed = np.empty((n_r, ttheta.size, pphi.size, 3))
        POS_unperturbed[..., 0] =   r_cell[:, None, None] * np.sin(T)[None, ...] * np.cos(P)[None, ...]
        POS_unperturbed[..., 1] =   r_cell[:, None, None] * np.sin(T)[None, ...] * np.sin(P)[None, ...]
        POS_unperturbed[..., 2] =   r_cell[:, None, None] * np.cos(T)[None, ...]

        # Add displacement to get the deformed grid
        POS = POS_unperturbed + xi
        
        # Compute dMass for each fluid element
        dMass = (r_cell**2)[:, None, None] \
        * np.sin(ttheta)[None, :, None] \
        * dr[:, None, None] \
        * dtheta[None, :, None] \
        * dphi[None, None, :] \
        * rho_avg[:, None, None]

        return {'POS_unperturbed': POS_unperturbed, "POS": POS, 'dMass': dMass}

  