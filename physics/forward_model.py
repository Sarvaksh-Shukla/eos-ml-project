import numpy as np
from .tov import TOVSolver
from eos.eos_param import PiecewisePolytrope 
class ForwardModel:
    def __init__(self, eos:PiecewisePolytrope):
        self.tov = TOVSolver(eos)

    def observables(self, rho_c_grid):
        out = []
        for rho_c in rho_c_grid:
            M, R, Lam = self.tov.tidal_deformability(rho_c)
            out.append({"rho_c": rho_c, "M": M, "R": R, "Lambda": Lam})
        return out
