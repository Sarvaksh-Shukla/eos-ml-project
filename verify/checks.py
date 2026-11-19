import numpy as np

def coverage(true_thetas, posterior_sampler_fn, xs, n_draws=200):
    """Compute empirical percentile ranks for one chosen parameter (per-datapoint).

    posterior_sampler_fn(n_draws, x) -> samples (n_draws, d)
    Returns ranks array in [0,1] for each datum (fraction of samples < truth)"""
    ranks = []
    for th_true, x in zip(true_thetas, xs):
        S = posterior_sampler_fn(n_draws, x)
        # choose one parameter to test (e.g., Gamma2 index 3)
        col = 3 if S.shape[1] > 3 else 0
        ranks.append((S[:, col] < th_true[col]).mean())
    return np.array(ranks)

def posterior_predictive(check_fn, posterior_sampler_fn, x_obs, n_draws=200):
    """Draw theta ~ posterior and produce predictive checks via check_fn(theta)
    check_fn should produce observables comparable to x_obs.
    Returns array of predictive realizations."""
    S = posterior_sampler_fn(n_draws, x_obs)
    preds = [check_fn(th) for th in S]
    return np.array(preds)
