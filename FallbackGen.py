import pygyre as pg
import numpy as np
import scipy as cp
from scipy import integrate
import matplotlib.pyplot as plt

class FallbackGen:
    "Does dT calculation offline"

    def __init__(self, a, rtde, rp, E, Lz, Q, dE, dLz, dQ):
        self.dE = dE
        self.dLz = dLz
        self.dQ = dQ
        self.a = a
        self.R_TDE = rtde
        self.Rp = rp
        self.OrbitEnergy = E
        self.mom = Lz
        self.Carter = Q

        self._compute_rel_dT()

    def geodesic_kerr_s2(self, τ,  y):
        """
        Testing new geodesic function based off of Kesden 2012.
        """
        t, r, phi, psi = y

        q = self.bound_Q
        E = self.bound_E
        a = self.a
        Lz = self.bound_Lz
        theta = np.pi / 2

        sigma = r**2 + a**2 * np.cos(theta)**2
        delta = r**2 + a**2 - 2 * r
        alpha = (r**2 + a**2)**2 - delta * a**2 * np.sin(theta)**2

        dt_dτ = ((alpha * E - 2 * a * r * Lz) / delta) / sigma
        dr_dτ = np.sqrt((E * (r**2 + a**2) - a * Lz)**2 - delta * (r**2 + (Lz - a * E)**2 + q)) / sigma
        # dφ_dτ = (Lz * np.csc(theta)**2 + (2 * a * r * E - a**2 * Lz) / delta) / sigma
        # dθ_dτ = np.sqrt(q - Lz**2 * np.cot(theta)**2 - a**2 * (1 - E**2) * np.cos(theta)**2) / sigma
        # dpsi_dτ = np.abs(a - Lz) * (((r**2 + a**2) - a * Lz) / ((a - Lz)**2 + r**2) + a * (Lz - a) / (a - Lz)**2) / r**2

        return [dt_dτ, dr_dτ]
    
    def R_of_r(self, r):
        a = self.a
        E = self.OrbitEnergy
        Lz = self.mom
        Q = self.Carter
        delta = r**2 - 2*r + a**2
        
        return (E*(r**2 + a**2) - a*Lz)**2 - delta*(r**2 + (Lz - a*E)**2 + Q)
    
    def find_ra(self):
        rp = self.Rp

        r1 = rp * (1 + 1e-6)
        r2 = r1

        while self.R_of_r(r2) > 0:
            r2 *= 1.3

        return cp.optimize.brentq(self.R_of_r, r1, r2)

    def _compute_rel_dT(self):
        """
        For stage two, compute dT.
        """

        dE = self.dE 
        dLz = self.dLz
        dq = self.dQ
        rp = self.Rp
        Lz = self.mom
        E = self.OrbitEnergy
        Q = self.Carter
        ε = 1e-6

        total_Energy = E + dE
        total_Q = Q + dq
        total_Lz = Lz + dLz

        # integration
        r = self.R_TDE
        tau_max = np.max(self.t)

        self.bound_E = total_Energy
        self.bound_Q = total_Q
        self.bound_Lz = total_Lz

        y0_out = [0.0, r, 0.0, 0.0]
        sol_out = cp.integrate.solve_ivp(
            self.geodesic_kerr_s2,
            (0, tau_max),
            y0_out,
            t_eval=self.t[self.t >= 0]
        )   

        tau = sol_out.t
        time = sol_out.y[0]
        radius = sol_out.y[1]

        # find tf when R reaches apocenter of orbit
        self.bound_R = cp.interpolate.interp1d(tau, radius, kind='cubic', fill_value='extrapolate')
        self.bound_obs_t = cp.interpolate.interp1d(tau, time, kind='cubic', fill_value='extrapolate')

        ra = self.find_ra()

        fp = lambda τ: self.bound_R(τ) - rp
        τ_rp = cp.optimize.brentq(fp, tau[0], tau[-1])

        fa = lambda τ: self.bound_R(τ) - ra
        τ_ra = cp.optimize.brentq(fa, tau[0], tau[-1])

        tf = self.bound_obs_t(τ_ra) - self.bound_obs_t(τ_rp)
        self.dT = 2 * tf

        # only bound orbits?
        # total_Energy = E * np.ones(dE.shape) + dE
        # total_Energy = np.where(total_Energy >= 1, np.nan, total_Energy)
        # total_Q = Q * np.ones(dq.shape) + dq
        # total_Lz = Lz * np.ones(dLz.shape) + dLz

        # bound_E = total_Energy[~np.isnan(total_Energy)]
        # bound_Q = total_Q[~np.isnan(total_Energy)]
        # bound_Lz = total_Lz[~np.isnan(total_Energy)]
    
        # for i in range(total_Energy.size):
        #     self.bound_E = bound_E[i]
        #     self.bound_Q = bound_Q[i]
        #     self.bound_Lz = bound_Lz[i]

        #     y0_out = [0.0, r, 0.0, 0.0]
        #     sol_out = cp.integrate.solve_ivp(
        #         self.geodesic_kerr_s2,
        #         (0, tau_max),
        #         y0_out,
        #         t_eval=self.t[self.t >= 0]
        #     )   

        #     tau = sol_out.t
        #     time = sol_out.y[0]
        #     radius = sol_out.y[1]

        #     # find tf when R reaches apocenter of orbit
        #     self.bound_R = cp.interpolate.interp1d(tau, radius, kind='cubic', fill_value='extrapolate')
        #     self.bound_obs_t = cp.interpolate.interp1d(tau, time, kind='cubic', fill_value='extrapolate')

        #     ra = self.find_ra()

        #     fp = lambda τ: self.bound_R(τ) - rp
        #     τ_rp = cp.optimize.brentq(fp, tau[0], tau[-1])

        #     fa = lambda τ: self.bound_R(τ) - ra
        #     τ_ra = cp.optimize.brentq(fa, tau[0], tau[-1])

        #     tf = self.bound_obs_t(τ_ra) - self.bound_obs_t(τ_rp)
        #     dT = 2 * tf

        #     dTs.append(dT)

        # self.dTs = dTs
    
    # def to_text(self):
    #     # take dTs and write file
    #     dE = self.dE 
    #     dLz = self.dLz 
    #     dQ = self.dQ
    #     dT = self.dT

    #     data = np.zeros(len(dE), dtype=[
    #         ("dE", "f8"),
    #         ("dLz", "f8"),
    #         ("dQ", "f8"),
    #         ("dT", "f8")
    #     ])

    #     data["dE"] = dE
    #     data[""] = dLz
    #     data["mass"] = dQ
    #     data["radius"] = dT

    #     np.save("mydata.npy", data)