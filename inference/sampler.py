import json
from pathlib import Path

import numpy as np
from eos.eos_param import PiecewisePolytrope

_PRIORS_PATH = Path(__file__).resolve().parents[1] / "data" / "priors_config.json"
with _PRIORS_PATH.open(encoding="utf-8") as _priors_file:
    PRIORS = json.load(_priors_file)


def sample_theta(rng: np.random.Generator):
    """Draw one EoS parameter vector from ``data/priors_config.json``."""
    rho1 = rng.uniform(*PRIORS["rho1"])
    rho2 = rng.uniform(*PRIORS["rho2"])
    if rho1 >= rho2:
        raise ValueError("Configured prior ranges must ensure rho1 < rho2")
    G1s = rng.uniform(*PRIORS["Gamma1"])
    G2s = rng.uniform(*PRIORS["Gamma2"])
    G3s = rng.uniform(*PRIORS["Gamma3"])
    p1_min, p1_max = PRIORS["P1"]
    P1s = 10 ** rng.uniform(np.log10(p1_min), np.log10(p1_max))
    return dict(rho1=rho1, rho2=rho2, Gamma1=G1s, Gamma2=G2s, Gamma3=G3s, P1=P1s)


def make_eos(theta):
    return PiecewisePolytrope(**theta)