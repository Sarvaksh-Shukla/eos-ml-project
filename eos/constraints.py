import numpy as np
from .units import C, MSUN
from .eos_param import PiecewisePolytrope
from physics.forward_model import ForwardModel
def causality_ok(eos:PiecewisePolytrope, rhos):
    cs2 = eos.cs2(rhos)
    cs2 = np.asarray(cs2)
    return bool(np.all(np.isfinite(cs2)) and np.all(cs2 <= 1.0 + 1e-6) and np.all(cs2 > 0))


def monotone_ok(eos:PiecewisePolytrope, rhos:np):
    P = eos.pressure(rhos)
    return np.all(np.diff(P) >= 0)


def max_mass_ok(forward_model:ForwardModel, rho_grid, mmin_msun=2.0):
    Ms = []
    for rc in rho_grid:
        M, R = forward_model.tov.integrate_star(rc)
        Ms.append(M if np.isfinite(M) else 0.0)
    Ms = np.array(Ms)
    return (Ms.max() / MSUN) >= mmin_msun
