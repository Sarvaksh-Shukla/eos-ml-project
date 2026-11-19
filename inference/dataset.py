import numpy as np
import os

from physics.forward_model import ForwardModel
from eos.units import MSUN
from eos.constraints import causality_ok,monotone_ok,max_mass_ok
from inference.sampler import sample_theta, make_eos

def synth_dataset(N=10, seed=0, save_path=None, max_tries=300):
    
    """Generate up to N successful synthetic examples passing physics checks.

    Returns:
        X: (n, 3) array of summary observables [M_1.4 (Msun), R_1.4 (km), Lambda_1.4]
        TH: (n, 6) array of theta parameters [rho1, rho2, Gamma1, Gamma2, Gamma3, P1]
    """
    rng = np.random.default_rng(seed)
    thetas = []
    xs = []
    tries = 0
    
    if tries == 0:
        print("=== sample_theta preview (first 10 will be printed) ===")
        
    while len(xs) < N and tries < max_tries:
        
        tries += 1
        th = sample_theta(rng)
        eos = make_eos(th)
        # quick grid checks
        rho_grid = np.linspace(5e13, 2e15, 10)
        try:
            if not causality_ok(eos, rho_grid):
                print("Rejected: causality")
                continue
            if not monotone_ok(eos, rho_grid):
                print("Rejected: monotonicity")
                continue
        except Exception:
            continue
        fm = ForwardModel(eos)
        rho_star_grid = np.geomspace(1e14, 1e15, 20)
        # require max mass >= 2.0 Msun
        try:
            if not max_mass_ok(fm, rho_star_grid, mmin_msun=2.0):
                print("Rejected: max_mass")
                continue
        except Exception:
            continue
        #compute sequences and extract M~1.4
        Ms, Rs, Ls = [], [], []
        for rc in rho_star_grid:
            M, R = fm.tov.integrate_star(rc)
            #print(M,"  ",R)
            _, _, Lam = fm.tov.tidal_deformability(rc)
            Ms.append(M)
            Rs.append(R)
            Ls.append(Lam)
            
        Ms = np.array(Ms)
        
        if np.all(np.isnan(Ms)):
            continue
        idx = np.argmin(np.abs(Ms/MSUN - 1.4))
        M_target = Ms[idx]/MSUN
        R_target = Rs[idx] / 1e5  # cm -> km
        L_target = Ls[idx]
        
        print(M_target,R_target,L_target)
        
        
        # guard against unrealistic or invalid outputs
        if not np.isfinite(M_target) or not np.isfinite(R_target) or not np.isfinite(L_target):
            print("Rejected: NaN/Inf in outputs")
            continue
        if not (0.5 < M_target < 7.0):
            print("Rejected: bad mass", M_target)
            continue
        if not (5.0 < R_target < 40.0):
            print("Rejected: bad radius", R_target)
            continue
        if L_target <= 0 or L_target > 1e8:
            print("Rejected: bad lambda", L_target)
            continue

        
        # guard against unrealistic outputs
        # if not (0.5 < M_target < 3.0):
        #     continue
        # if not (5.0 < R_target < 20.0):
        #     continue
        # if np.isnan(L_target) or L_target <= 0:
        #     continue
        
        xs.append([M_target, R_target, L_target])
        thetas.append([th[k] for k in ("rho1", "rho2", "Gamma1", "Gamma2", "Gamma3", "P1")])
    X = np.array(xs, dtype=float)
    TH = np.array(thetas, dtype=float)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, X=X, TH=TH)
    return X, TH

def load_dataset(path):
    print("load_dataset")
    data = np.load(path)
    return data['X'], data['TH']
