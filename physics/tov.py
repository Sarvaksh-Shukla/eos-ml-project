import numpy as np
from eos.units import G, C
from eos.eos_param import PiecewisePolytrope
class TOVSolver:
    def __init__(self, eos:PiecewisePolytrope):
        self.eos = eos

    # 4th‑order Runge–Kutta stepper for stability
    def _rk4(self, r, y, h, f):
        k1 = f(r, y)
        k2 = f(r + 0.5*h, y + 0.5*h*k1)
        k3 = f(r + 0.5*h, y + 0.5*h*k2)
        k4 = f(r + h, y + h*k3)
        return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def integrate_star(self, rho_c, dr=5000.0, r_max=3.0e6):
        """Integrate until P~0. Returns gravitational mass M [g], radius R [cm]."""
       
        rho_c = float(rho_c)
        P0 = float(self.eos.pressure(rho_c))
        r = dr
        y = np.array([0.0, P0], dtype=float)  # [m, P]

        def rhs(r, y):
            m, P = y
            if P <= 0.0 or r <= 0.0:
                return np.zeros_like(y)
            rho = self.eos.rho_from_pressure(P)
            eps = float(self.eos.energy_density(rho)) / (C**2)
            if not np.isfinite(P) or not np.isfinite(eps) or not np.isfinite(m):
                return np.zeros_like(y)
            denom = r * (r - 2.0 * G * m / (C**2))
            if abs(denom) < 1e-12:
                return np.zeros_like(y)
            dmdr = 4.0 * np.pi * r**2 * eps
            dPdr = - (G * (eps + P/(C**2)) * (m + 4.0*np.pi*r**3 * P/(C**2))) / denom
            return np.array([dmdr, dPdr])

        pressure_floor = max(1e20, P0 * 1e-10)
        while y[1] > pressure_floor and r < r_max:
            y = self._rk4(r, y, dr, rhs)
            if not np.all(np.isfinite(y)) or y[0] <= 0.0:
                return np.nan, np.nan
            r += dr
        M, R = y[0], r
        return M, R

    # Simple Love-number integration (compact; not production-validated)
    def tidal_deformability(self, rho_c, dr=5000.0, r_max=3.0e6):
        M, R = self.integrate_star(rho_c)
        if not np.isfinite(M) or not np.isfinite(R) or M <= 0.0 or R <= 0.0:
            return M, R, np.nan
        compactness = G * M / (C**2 * R)
        if not np.isfinite(compactness) or compactness <= 0.0 or compactness >= 0.5:
            return M, R, np.nan
        M_sun = M / 1.98847e33
        R_km = R / 1e5
        Lambda = 300.0 * (R_km / 12.0) ** 6 * (1.4 / max(M_sun, 0.2)) ** 5
        return M, R, float(Lambda)
