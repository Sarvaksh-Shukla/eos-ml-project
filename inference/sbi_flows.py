import numpy as np
import torch as t
from torch.utils.data import TensorDataset, DataLoader
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

class ConditionalMAF:
    def __init__(self, dim_theta, dim_x, hidden=128, K=5, device=None):
        self.device = device or ("cuda" if t.cuda.is_available() else "cpu")
        transforms = []
        for _ in range(K):
            transforms.append(MaskedAffineAutoregressiveTransform(features=dim_theta, hidden_features=hidden, context_features=dim_x))
        transform = CompositeTransform(transforms)
        base = StandardNormal([dim_theta])
        self.flow = Flow(transform, base).to(self.device)

    def to(self, device):
        self.device = device
        self.flow.to(device)

    def log_prob(self, theta, x):
        return self.flow.log_prob(theta, context=x)

    def sample(self, num_samples, x_context):
        return self.flow.sample(num_samples, context=x_context)

def train_flow(thetas, xs, epochs=300, bs=256, lr=1e-3, device=None, verbose=True):
    device = device or ("cuda" if t.cuda.is_available() else "cpu")
    thetas = np.asarray(thetas, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    if (
        thetas.ndim != 2
        or xs.ndim != 2
        or thetas.shape[0] != xs.shape[0]
        or not thetas.shape[0]
    ):
        raise ValueError("Theta and observation arrays must be non-empty 2-D arrays")
    if not np.all(np.isfinite(thetas)) or not np.all(np.isfinite(xs)):
        raise ValueError("Training arrays must contain only finite values")
    th_mean = thetas.mean(axis=0)
    th_std = thetas.std(axis=0)
    th_std[th_std < 1e-8] = 1.0
    x_mean = xs.mean(axis=0)
    x_std = xs.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    thn = (thetas - th_mean) / th_std
    xn = (xs - x_mean) / x_std
    thetas = t.tensor(thn, dtype=t.float32).to(device)
    xs = t.tensor(xn, dtype=t.float32).to(device)
    ds = TensorDataset(thetas, xs)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    model = ConditionalMAF(dim_theta=thetas.shape[1], dim_x=xs.shape[1], device=device)
    model.th_mean = th_mean
    model.th_std = th_std
    model.x_mean = x_mean
    model.x_std = x_std
    opt = t.optim.Adam(model.flow.parameters(), lr=lr)
    for ep in range(epochs):
        total_loss = 0.0
        for th_batch, x_batch in dl:
            logp = model.log_prob(th_batch, x_batch)
            loss = - logp.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * th_batch.size(0)
        if verbose and (ep + 1) % 50 == 0:
            print(f"[flow] epoch {ep+1}/{epochs} nll={total_loss/len(ds):.6f}")
    return model

def flow_sample_posterior(flow_model, x_obs, n_samples=500):
    import torch as T
    x_obs = np.asarray(x_obs, dtype=np.float64)
    if (
        x_obs.ndim not in (1, 2)
        or (x_obs.ndim == 2 and x_obs.shape[0] != 1)
        or not np.all(np.isfinite(x_obs))
    ):
        raise ValueError("x_obs must be one finite observation vector")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    xn = (x_obs - flow_model.x_mean) / flow_model.x_std
    x = T.tensor(xn, dtype=T.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    with T.no_grad():
        samples = flow_model.sample(n_samples, x.to(flow_model.device))
    samples = samples.cpu().numpy().reshape(-1, flow_model.th_mean.shape[0])
    samples = samples * flow_model.th_std + flow_model.th_mean
    samples[:, 0] = np.clip(samples[:, 0], 1.2e14, 1.8e14)
    samples[:, 1] = np.clip(samples[:, 1], 4.5e14, 6e14)
    samples[:, 2] = np.clip(samples[:, 2], 1.3, 1.6)
    samples[:, 3] = np.clip(samples[:, 3], 1.8, 2.4)
    samples[:, 4] = np.clip(samples[:, 4], 2.8, 3.1)
    samples[:, 5] = np.clip(samples[:, 5], 2e33, 5e33)
    return samples
