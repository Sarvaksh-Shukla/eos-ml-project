import numpy as np
import matplotlib.pyplot as plt
from eos.units import MSUN, KM

def plot_mr_curve(forward_model, rho_cs, ax=None, label=None):
    Ms, Rs = [], []
    for rc in rho_cs:
        M, R = forward_model.tov.integrate_star(rc)
        Ms.append(M/MSUN); Rs.append(R/KM)
    ax = ax or plt.gca()
    ax.plot(Rs, Ms, lw=1.5, label=label)
    ax.set_xlabel('R [km]')
    ax.set_ylabel('M [M$_\\odot$]')
    if label: ax.legend()
    return ax

def plot_lambda_vs_mass(forward_model, rho_cs=None, ax=None):
    rho_cs = rho_cs or np.geomspace(6e13, 3e15, 40)
    Ms, Ls = [], []
    for rc in rho_cs:
        M, R, Lam = forward_model.tov.tidal_deformability(rc)
        Ms.append(M/MSUN); Ls.append(Lam)
    ax = ax or plt.gca()
    ax.plot(Ms, Ls)
    ax.set_xlabel('M [M$_\\odot$]')
    ax.set_ylabel('Lambda')
    ax.set_yscale('log')
    return ax

