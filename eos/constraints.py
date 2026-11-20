import numpy as np
from .units import MSUN
from .eos_param import PiecewisePolytrope
from physics.forward_model import ForwardModel

def causality_ok(eos: PiecewisePolytrope, rhos):
    rhos = np.asarray(rhos, dtype=float)
    if rhos.size == 0 or np.any(~np.isfinite(rhos)) or np.any(rhos <= 0):
        return False
    cs2 = np.asarray(eos.cs2(rhos), dtype=float)
    return bool(
        np.all(np.isfinite(cs2))
        and np.all(cs2 > 0.0)
        and np.all(cs2 <= 1.0 + 1e-6)
    )


def monotone_ok(eos: PiecewisePolytrope, rhos):
    rhos = np.asarray(rhos, dtype=float)
    if (
        rhos.size == 0
        or np.any(~np.isfinite(rhos))
        or np.any(rhos < 0)
        or np.any(np.diff(rhos) < 0)
    ):
        return False
    pressure = np.asarray(eos.pressure(rhos), dtype=float)
    return bool(np.all(np.isfinite(pressure)) and np.all(np.diff(pressure) >= 0))


def max_mass_ok(forward_model: ForwardModel, rho_grid, mmin_msun=2.0):
    rho_grid = np.asarray(rho_grid, dtype=float)
    if (
        rho_grid.size == 0
        or np.any(~np.isfinite(rho_grid))
        or np.any(rho_grid <= 0)
        or not np.isfinite(mmin_msun)
        or mmin_msun <= 0
    ):
        return False
    masses = np.asarray(
        [forward_model.tov.integrate_star(rho_c)[0] for rho_c in rho_grid],
        dtype=float,
    )
    return bool(np.all(np.isfinite(masses)) and masses.max() / MSUN >= mmin_msun)
