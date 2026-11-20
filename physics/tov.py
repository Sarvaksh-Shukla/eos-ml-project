import numpy as np
from scipy.integrate import solve_ivp

from eos.eos_param import PiecewisePolytrope
from eos.units import C, G


class TOVSolver:
    """Integrate TOV structure and quadrupolar tidal perturbation equations.

    The EoS uses CGS pressure and energy density. The Love-number equation is
    evaluated in geometrized units, with lengths in cm.
    """

    def __init__(self, eos: PiecewisePolytrope):
        self.eos = eos

    def _integrate(self, rho_c, dr, r_max, with_tides):
        rho_c = float(rho_c)
        if not np.isfinite(rho_c) or rho_c <= 0:
            raise ValueError("Central density must be finite and positive")
        if not np.isfinite(dr) or dr <= 0 or not np.isfinite(r_max) or r_max <= dr:
            raise ValueError("Require finite dr > 0 and r_max > dr")

        central_pressure = float(self.eos.pressure(rho_c))
        central_mass_density = float(self.eos.energy_density(rho_c)) / C**2
        if not np.isfinite(central_pressure) or central_pressure <= 0:
            raise ValueError("Central pressure must be finite and positive")

        r0 = min(10.0, dr * 1e-3)
        pressure_correction = (
            2.0 * np.pi * G
            * (central_mass_density + central_pressure / C**2)
            * (central_mass_density / 3.0 + central_pressure / C**2)
            * r0**2
        )
        initial_pressure = max(central_pressure - pressure_correction, central_pressure * 0.5)
        initial_mass = (4.0 / 3.0) * np.pi * central_mass_density * r0**3
        state0 = [initial_mass, initial_pressure]
        if with_tides:
            state0.append(2.0)

        def rhs(radius, state):
            mass, pressure = state[:2]
            pressure = max(float(pressure), 0.0)
            density = self.eos.rho_from_pressure(pressure)
            energy_geo = G * float(self.eos.energy_density(density)) / C**4
            pressure_geo = G * pressure / C**4
            mass_geo = G * mass / C**2
            compactness_factor = 1.0 - 2.0 * mass_geo / radius
            if compactness_factor <= 0.0:
                raise RuntimeError("Integration reached a trapped-surface condition")

            dmass_dr = 4.0 * np.pi * radius**2 * energy_geo * C**2 / G
            dpressure_dr = (
                -G
                * (energy_geo * C**2 / G + pressure / C**2)
                * (mass + 4.0 * np.pi * radius**3 * pressure / C**2)
                / (radius**2 * compactness_factor)
            )
            result = [dmass_dr, dpressure_dr]

            if with_tides:
                y = float(state[2])
                if density <= 0.0:
                    sound_speed_sq = 1.0
                else:
                    sound_speed_sq = float(self.eos.cs2(density))
                if not np.isfinite(sound_speed_sq) or sound_speed_sq <= 0.0:
                    raise RuntimeError("EoS has non-positive or non-finite sound speed")
                F = (
                    1.0
                    - 4.0 * np.pi * radius**2 * (energy_geo - pressure_geo)
                ) / compactness_factor
                Q = (
                    4.0
                    * np.pi
                    * (
                        5.0 * energy_geo
                        + 9.0 * pressure_geo
                        + (energy_geo + pressure_geo) / sound_speed_sq
                    )
                    / compactness_factor
                    - 6.0 / (radius**2 * compactness_factor)
                    - 4.0
                    * (mass_geo + 4.0 * np.pi * radius**3 * pressure_geo) ** 2
                    / (radius**4 * compactness_factor**2)
                )
                dy_dr = -(y**2 + y * F + radius**2 * Q) / radius
                result.append(dy_dr)
            return result

        def surface_event(_radius, state):
            return state[1]

        surface_event.terminal = True
        surface_event.direction = -1
        solution = solve_ivp(
            rhs,
            (r0, float(r_max)),
            state0,
            method="DOP853",
            events=surface_event,
            max_step=float(dr),
            rtol=2e-7,
            atol=[1e15, 1e15] + ([1e-9] if with_tides else []),
        )
        if not solution.success:
            raise RuntimeError(f"TOV integration failed: {solution.message}")
        if not solution.t_events[0].size:
            raise RuntimeError(
                f"Surface not reached before r_max={r_max:g} cm; "
                "check the EoS or increase r_max"
            )

        radius = float(solution.t_events[0][0])
        mass = float(solution.y_events[0][0][0])
        if not np.isfinite(mass) or mass <= 0 or not np.isfinite(radius):
            raise RuntimeError("TOV integration returned an invalid star")
        if not with_tides:
            return mass, radius

        y_surface = float(solution.y_events[0][0][2])
        compactness = G * mass / (C**2 * radius)
        if not np.isfinite(y_surface) or not 0 < compactness < 0.5:
            raise RuntimeError("Invalid surface state for tidal deformability")

        term1 = (
            (8.0 / 5.0)
            * compactness**5
            * (1.0 - 2.0 * compactness) ** 2
            * (2.0 + 2.0 * compactness * (y_surface - 1.0) - y_surface)
        )
        denominator = (
            2.0 * compactness
            * (6.0 - 3.0 * y_surface + 3.0 * compactness * (5.0 * y_surface - 8.0))
            + 4.0
            * compactness**3
            * (
                13.0
                - 11.0 * y_surface
                + compactness * (3.0 * y_surface - 2.0)
                + 2.0 * compactness**2 * (1.0 + y_surface)
            )
            + 3.0
            * (1.0 - 2.0 * compactness) ** 2
            * (2.0 - y_surface + 2.0 * compactness * (y_surface - 1.0))
            * np.log1p(-2.0 * compactness)
        )
        if not np.isfinite(denominator) or abs(denominator) < 1e-30:
            raise RuntimeError("Love-number calculation has a singular denominator")
        k2 = term1 / denominator
        tidal_deformability = (2.0 / 3.0) * k2 / compactness**5
        if not np.isfinite(tidal_deformability) or tidal_deformability <= 0:
            raise RuntimeError("Love-number calculation returned invalid deformability")
        return mass, radius, float(tidal_deformability)

    def integrate_star(self, rho_c, dr=2.0e4, r_max=2.0e7):
        """Return gravitational mass [g] and radius [cm] at the zero-pressure surface."""
        return self._integrate(rho_c, dr, r_max, with_tides=False)

    def tidal_deformability(self, rho_c, dr=2.0e4, r_max=2.0e7):
        """Return mass [g], radius [cm], and dimensionless tidal deformability."""
        return self._integrate(rho_c, dr, r_max, with_tides=True)
