import numpy as np
from .units import C

class PiecewisePolytrope:
    """
    3-segment piecewise polytrope: P = K_i * rho^{Gamma_i} with breaks rho1<rho2.
    Continuity in P enforced at rho1 and rho2.
    Parameters: rho1, rho2 [g/cc], Gamma1..3, P1=P(rho1) [dyne/cm^2].
    """
    def __init__(self, rho1, rho2, Gamma1, Gamma2, Gamma3, P1):
        assert rho1 < rho2
        self.rho1, self.rho2 = float(rho1), float(rho2)
        self.Gamma1, self.Gamma2, self.Gamma3 = map(float, (Gamma1, Gamma2, Gamma3))
        self.P1 = float(P1)
        # K1 from P1; then P2 continuity; then K2, K3
        self.K1 = self.P1 / (self.rho1 ** self.Gamma1)
        P2 = self.K1 * (self.rho2 ** self.Gamma1)
        self.K2 = P2 / (self.rho2 ** self.Gamma2)
        self.K3 = self.K2 * (self.rho2 ** (self.Gamma2 - self.Gamma3))

    def pressure(self, rho):
        rho = np.asarray(rho)
        P = np.zeros_like(rho, dtype=float)
        i1 = rho < self.rho1
        i2 = (rho >= self.rho1) & (rho < self.rho2)
        i3 = rho >= self.rho2
        P[i1] = self.K1 * rho[i1] ** self.Gamma1
        P[i2] = self.K2 * rho[i2] ** self.Gamma2
        P[i3] = self.K3 * rho[i3] ** self.Gamma3
        return P if P.shape else P.item()

   
    
    def energy_density(self, rho):
        """
        Cold barotrope: epsilon(rho) ≈ rho c^2 + P/(Gamma-1). Piecewise.
        """
        rho = np.asarray(rho)
        eps = np.zeros_like(rho, dtype=float)

        def seg_eps(r, K, Gm):
            P = K * r ** Gm
            u = P / (Gm - 1.0)
            return r * C**2 + u
            

        i1 = rho < self.rho1
        i2 = (rho >= self.rho1) & (rho < self.rho2)
        i3 = rho >= self.rho2
        if np.any(i1):
            eps[i1] = seg_eps(rho[i1], self.K1, self.Gamma1)
        if np.any(i2):
            eps[i2] = seg_eps(rho[i2], self.K2, self.Gamma2)
        if np.any(i3):
            eps[i3] = seg_eps(rho[i3], self.K3, self.Gamma3)
        return eps if eps.shape else eps.item()

    def cs2(self, rho):
        """c_s^2 = dP/dε. Uses analytic dP/drho and dε/drho per segment."""
        rho = np.atleast_1d(rho)
        out = np.empty_like(rho, dtype=float)
        for k, r in enumerate(rho):
            if r < self.rho1:
                K, Gm = self.K1, self.Gamma1
            elif r < self.rho2:
                K, Gm = self.K2, self.Gamma2
            else:
                K, Gm = self.K3, self.Gamma3
            dPdrho = K * Gm * r ** (Gm - 1.0)
            depsdrho = (C * C) + (Gm / (Gm - 1.0)) * K * r ** (Gm - 1.0)
            out[k] = dPdrho / depsdrho
        return out if out.shape[0] > 1 else out.item()

    # invert P(rho) per segment (monotone)
    def rho_from_pressure(self, P):
        P = float(P)
        P_rho1 = self.K1 * (self.rho1 ** self.Gamma1)
        P_rho2 = self.K2 * (self.rho2 ** self.Gamma2)
        if P < P_rho1:
            return (P / self.K1) ** (1.0 / self.Gamma1)
        elif P < P_rho2:
            return (P / self.K2) ** (1.0 / self.Gamma2)
        else:
            return (P / self.K3) ** (1.0 / self.Gamma3)
