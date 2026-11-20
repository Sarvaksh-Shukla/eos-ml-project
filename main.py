import numpy as np
import matplotlib.pyplot as plt
from inference.dataset import synth_dataset
from inference.supervised_baseline import train_inverse_regressor, predict_inverse
from inference.sbi_flows import train_flow, flow_sample_posterior
from inference.dataset_utils import train_val_split
from inference.sampler import make_eos
from physics.forward_model import ForwardModel
from verify.plots import plot_mr_curve

if __name__ == "__main__":
    print("Simulating dataset...")
    X, TH = synth_dataset(N=10, seed=42)
    
    print("Simulated:", X.shape, TH.shape)
    # quick train/val split (we only train on entire set for demo)
    X_train, X_val, TH_train, TH_val = train_val_split(X, TH, val_frac=0.2)

    # Train inverse regressor
    print("Training inverse regressor...")
    inv = train_inverse_regressor(X_train, TH_train, epochs=50, verbose=True)

    # Train conditional flow
    print("Training conditional flow (SBI)...")
    flow = train_flow(TH_train, X_train, epochs=50)

    # Pick a synthetic observation
    x_obs = X_val[0]
    print("x_obs:", x_obs)

    # MAP-like point estimate from inverse regressor
    th_map = predict_inverse(inv, x_obs.reshape(1, -1))[0]
    theta_keys = ["rho1","rho2","Gamma1","Gamma2","Gamma3","P1"]
    th_est = {k: float(v) for k, v in zip(theta_keys, th_map)}

    # plot M-R for the point estimate
    eos_est = make_eos(th_est)
    fm = ForwardModel(eos_est)
    rho_scan = np.geomspace(6e13, 3e15, 40)
    fig, ax = plt.subplots(figsize=(6,4))
    plot_mr_curve(fm, rho_scan, ax=ax, label="estimate")
    ax.set_title("M–R curve (point estimate)")
    plt.tight_layout()
    plt.show()

    # Posterior samples from flow
    print("Sampling posterior from flow...")
    S = flow_sample_posterior(flow, x_obs, n_samples=400)
    print("Posterior samples shape:", S.shape)

