import logging
import os

import numpy as np

from eos.constraints import causality_ok, monotone_ok
from eos.units import MSUN
from inference.sampler import make_eos, sample_theta
from physics.forward_model import ForwardModel

logger = logging.getLogger(__name__)
THETA_KEYS = ("rho1", "rho2", "Gamma1", "Gamma2", "Gamma3", "P1")
TARGET_MASS_MSUN = 1.4


def synth_dataset(N=10, seed=0, save_path=None, max_tries=300):
    """Generate physics-filtered examples ``(M, R, Lambda)`` and EoS parameters.

    ``M`` is in solar masses, ``R`` is in km, and ``Lambda`` is dimensionless.
    Only candidates with a causal, monotone EoS, maximum mass >= 2 solar masses,
    and a stable-branch star near 1.4 solar masses are returned.
    """
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer")
    if not isinstance(max_tries, int) or max_tries < N:
        raise ValueError("max_tries must be an integer >= N")

    rng = np.random.default_rng(seed)
    observables = []
    theta_rows = []
    rejection_counts = {}
    central_density_min = 1e12
    central_density_max = 5e15
    causal_grid = np.geomspace(central_density_min, central_density_max, 128)

    def reject(reason):
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for _ in range(max_tries):
        if len(observables) == N:
            break
        theta = sample_theta(rng)
        try:
            eos = make_eos(theta)
            sound_speeds = np.asarray(eos.cs2(causal_grid), dtype=float)
            if not np.all(np.isfinite(sound_speeds)) or np.any(sound_speeds <= 0):
                reject("causality")
                continue
            acausal = np.flatnonzero(sound_speeds > 1.0 + 1e-6)
            if acausal.size:
                first_acausal = int(acausal[0])
                if first_acausal == 0:
                    reject("causality")
                    continue
                density_below = causal_grid[first_acausal - 1]
                density_above = causal_grid[first_acausal]
                speed_below = sound_speeds[first_acausal - 1]
                speed_above = sound_speeds[first_acausal]
                fraction = (1.0 - speed_below) / (speed_above - speed_below)
                causal_density_max = density_below * (
                    density_above / density_below
                ) ** fraction
            else:
                causal_density_max = central_density_max
            central_densities = np.geomspace(
                central_density_min, causal_density_max, 32
            )
            if not causality_ok(
                eos, np.geomspace(central_density_min, causal_density_max, 128)
            ):
                reject("causality")
                continue
            if not monotone_ok(eos, causal_grid):
                reject("monotonicity")
                continue

            model = ForwardModel(eos)
            sequence = [model.tov.integrate_star(rho_c) for rho_c in central_densities]
            masses = np.asarray([mass / MSUN for mass, _ in sequence])
            radii = np.asarray([radius / 1e5 for _, radius in sequence])
            if not np.all(np.isfinite(masses)) or not np.all(np.isfinite(radii)):
                reject("invalid stellar sequence")
                continue

            maximum_index = int(np.argmax(masses))
            if masses[maximum_index] < 2.0:
                reject("maximum mass")
                continue

            stable_masses = masses[: maximum_index + 1]
            stable_densities = central_densities[: maximum_index + 1]
            crossing = np.flatnonzero(
                (stable_masses[:-1] <= TARGET_MASS_MSUN)
                & (stable_masses[1:] >= TARGET_MASS_MSUN)
                & (np.diff(stable_masses) > 0)
            )
            if crossing.size == 0:
                reject("no stable 1.4-solar-mass crossing")
                continue

            index = int(crossing[0])
            fraction = (
                (TARGET_MASS_MSUN - stable_masses[index])
                / (stable_masses[index + 1] - stable_masses[index])
            )
            log_rho = (
                np.log(stable_densities[index])
                + fraction
                * (
                    np.log(stable_densities[index + 1])
                    - np.log(stable_densities[index])
                )
            )
            target_rho = float(np.exp(log_rho))
            mass, radius, tidal = model.tov.tidal_deformability(target_rho)
            mass_msun = mass / MSUN
            radius_km = radius / 1e5
            if not (
                np.isfinite(mass_msun)
                and abs(mass_msun - TARGET_MASS_MSUN) <= 0.15
                and np.isfinite(radius_km)
                and 5.0 < radius_km < 40.0
                and np.isfinite(tidal)
                and tidal > 0.0
            ):
                reject("invalid target observables")
                continue
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            logger.debug("Rejected EoS candidate: %s", exc)
            reject("integration or parameter error")
            continue

        observables.append([mass_msun, radius_km, tidal])
        theta_rows.append([theta[key] for key in THETA_KEYS])

    if len(observables) < N:
        raise RuntimeError(
            f"Generated {len(observables)} of {N} requested examples after "
            f"{max_tries} attempts; rejection counts: {rejection_counts}"
        )

    X = np.asarray(observables, dtype=np.float64)
    TH = np.asarray(theta_rows, dtype=np.float64)
    if save_path:
        directory = os.path.dirname(os.fspath(save_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(save_path, X=X, TH=TH)
    return X, TH


def load_dataset(path):
    with np.load(path) as data:
        if "X" not in data or "TH" not in data:
            raise ValueError("Dataset archive must contain 'X' and 'TH' arrays")
        X, TH = data["X"], data["TH"]
    if X.ndim != 2 or TH.ndim != 2 or X.shape[0] != TH.shape[0]:
        raise ValueError("Dataset arrays must be 2-D and have matching rows")
    return X, TH
