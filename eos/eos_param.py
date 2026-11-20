import numpy as np

from .units import C


class PiecewisePolytrope:
    """Three-segment cold polytrope in CGS units.

    Pressure is ``K_i * rho**Gamma_i``. ``rho`` is rest-mass density in
    g/cm^3, pressure is dyn/cm^2, and energy density is erg/cm^3.
    Integration constants make the energy density continuous at both breaks.
    """

    def __init__(self, rho1, rho2, Gamma1, Gamma2, Gamma3, P1):
        parameters = np.asarray(
            [rho1, rho2, Gamma1, Gamma2, Gamma3, P1], dtype=float
        )
        if not np.all(np.isfinite(parameters)):
            raise ValueError("All EoS parameters must be finite")
        if rho1 <= 0 or rho2 <= rho1:
            raise ValueError("Density breaks must satisfy 0 < rho1 < rho2")
        if min(Gamma1, Gamma2, Gamma3) <= 1.0:
            raise ValueError("All adiabatic indices must be greater than 1")
        if P1 <= 0:
            raise ValueError("P1 must be positive")

        self.rho1, self.rho2 = float(rho1), float(rho2)
        self.Gamma1, self.Gamma2, self.Gamma3 = map(
            float, (Gamma1, Gamma2, Gamma3)
        )
        self.P1 = float(P1)
        self.K1 = self.P1 / self.rho1**self.Gamma1
        self.K2 = self.K1 * self.rho1 ** (self.Gamma1 - self.Gamma2)
        pressure2 = self.K2 * self.rho2**self.Gamma2
        self.K3 = pressure2 / self.rho2**self.Gamma3

        self._segments = (
            (self.rho1, self.K1, self.Gamma1),
            (self.rho2, self.K2, self.Gamma2),
            (np.inf, self.K3, self.Gamma3),
        )
        self._energy_offsets = self._compute_energy_offsets()

    def _compute_energy_offsets(self):
        offsets = [0.0]
        for i, density_break in enumerate((self.rho1, self.rho2)):
            _, K_prev, gamma_prev = self._segments[i]
            _, K_next, gamma_next = self._segments[i + 1]
            pressure = K_prev * density_break**gamma_prev
            energy_prev = (
                density_break * (C**2 + offsets[i])
                + pressure / (gamma_prev - 1.0)
            )
            offsets.append(
                (energy_prev - density_break * C**2
                 - pressure / (gamma_next - 1.0)) / density_break
            )
        return tuple(offsets)

    def _segment_values(self, rho):
        rho = np.asarray(rho, dtype=float)
        limits = np.array([self.rho1, self.rho2])
        indices = np.searchsorted(limits, rho, side="right")
        Ks = np.array([self.K1, self.K2, self.K3])[indices]
        gammas = np.array([self.Gamma1, self.Gamma2, self.Gamma3])[indices]
        offsets = np.asarray(self._energy_offsets)[indices]
        return rho, Ks, gammas, offsets

    def pressure(self, rho):
        rho, Ks, gammas, _ = self._segment_values(rho)
        if np.any(~np.isfinite(rho)) or np.any(rho < 0):
            raise ValueError("Density must be finite and non-negative")
        pressure = Ks * np.power(rho, gammas)
        return float(pressure) if pressure.ndim == 0 else pressure

    def energy_density(self, rho):
        """Return total energy density, including rest-mass energy [erg/cm^3]."""
        rho, Ks, gammas, offsets = self._segment_values(rho)
        if np.any(~np.isfinite(rho)) or np.any(rho < 0):
            raise ValueError("Density must be finite and non-negative")
        pressure = Ks * np.power(rho, gammas)
        energy = rho * (C**2 + offsets) + pressure / (gammas - 1.0)
        return float(energy) if energy.ndim == 0 else energy

    def cs2(self, rho):
        """Return ``dP/dε`` (sound speed squared divided by ``c**2``)."""
        rho, Ks, gammas, offsets = self._segment_values(rho)
        if np.any(~np.isfinite(rho)) or np.any(rho <= 0):
            raise ValueError("Density must be finite and positive")
        dp_drho = Ks * gammas * np.power(rho, gammas - 1.0)
        de_drho = (
            C**2 + offsets
            + gammas / (gammas - 1.0) * Ks * np.power(rho, gammas - 1.0)
        )
        result = dp_drho / de_drho
        return float(result) if result.ndim == 0 else result

    def rho_from_pressure(self, pressure):
        """Invert the monotone pressure relation for a non-negative pressure."""
        pressure = float(pressure)
        if not np.isfinite(pressure) or pressure < 0:
            raise ValueError("Pressure must be finite and non-negative")
        if pressure == 0:
            return 0.0
        p1 = self.pressure(self.rho1)
        p2 = self.pressure(self.rho2)
        if pressure < p1:
            K, gamma = self.K1, self.Gamma1
        elif pressure < p2:
            K, gamma = self.K2, self.Gamma2
        else:
            K, gamma = self.K3, self.Gamma3
        return (pressure / K) ** (1.0 / gamma)
