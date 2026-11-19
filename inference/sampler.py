import numpy as np
from eos.eos_param import PiecewisePolytrope

#Priors (broad but physical; adjust as you refine)
RHO1 = (1e14, 1.8e14)
RHO2 = (4e14, 8e14)
G1 = (1.4, 2.1)
G2 = (1.4, 2.2)
G3 = (1.4, 2.2)
P1 = (1e34, 5e34)  # dyne/cm^2



def sample_theta(rng: np.random.Generator):
    
    rho1 = rng.uniform(*RHO1)
    rho2 = rng.uniform(*RHO2)
    if rho2 <= rho1:
        rho1, rho2 = min(rho1, rho2*0.9), max(rho1*1.1, rho2)
    G1s = rng.uniform(*G1)
    G2s = rng.uniform(*G2)
    G3s = rng.uniform(*G3)
    P1s = 10 ** rng.uniform(np.log10(P1[0]), np.log10(P1[1]))
    return dict(rho1=rho1, rho2=rho2, Gamma1=G1s, Gamma2=G2s, Gamma3=G3s, P1=P1s)


def make_eos(theta):
    values = dict(theta)
    rho1 = float(values["rho1"])
    rho2 = float(values["rho2"])
    P1 = float(values["P1"])
    if not np.isfinite(rho1) or not np.isfinite(rho2) or rho1 <= 0 or rho2 <= 0:
        raise ValueError("Density parameters must be finite and positive")
    if rho1 >= rho2:
        rho1, rho2 = min(rho1, rho2), max(rho1, rho2)
        if rho1 == rho2:
            rho1 = rho2 * 0.5
    if not np.isfinite(P1):
        P1 = 1e34
    P1 = abs(P1)
    values["rho1"] = rho1
    values["rho2"] = rho2
    values["P1"] = max(P1, 1e30)
    eos = PiecewisePolytrope(**values)

    return eos