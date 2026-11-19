import torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class MLP(nn.Module):
    def __init__(self, din, dout, width=256, depth=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(din, width))
        layers.append(nn.ReLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_inverse_regressor(X, TH, epochs=200, lr=1e-3, bs=128, device=None, verbose=True):
    device = device or ("cuda" if t.cuda.is_available() else "cpu")
    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=1e5, neginf=-1e5)
    TH = np.nan_to_num(np.asarray(TH, dtype=np.float32), nan=0.0, posinf=1e5, neginf=-1e5)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    th_mean = TH.mean(axis=0)
    th_std = TH.std(axis=0)
    th_std[th_std < 1e-8] = 1.0
    Xn = (X - x_mean) / x_std
    THn = (TH - th_mean) / th_std
    X = t.tensor(Xn, dtype=t.float32)
    TH = t.tensor(THn, dtype=t.float32)

    ds = TensorDataset(X, TH)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    model = MLP(din=X.shape[1], dout=TH.shape[1]).to(device)
    model.x_mean = x_mean
    model.x_std = x_std
    model.th_mean = th_mean
    model.th_std = th_std
    opt = t.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = lossf(pred, yb)
            loss.backward(); opt.step()
            running += loss.item() * xb.size(0)
        if verbose and (ep + 1) % 50 == 0:
            print(f"[inv] epoch {ep+1}/{epochs} mse={running/len(ds):.6e}")
    return model

def predict_inverse(model, x_obs, device=None):
    device = device or ("cuda" if t.cuda.is_available() else "cpu")
    model = model.to(device)
    import torch as T
    with T.no_grad():
        x_obs = np.asarray(x_obs, dtype=np.float32)
        xn = (x_obs - model.x_mean) / model.x_std
        tx = T.tensor(xn, dtype=T.float32).to(device)
        out = model(tx).cpu().numpy()
    out = out * model.th_std + model.th_mean
    out[:, 0] = np.clip(out[:, 0], 5e13, 2e14)
    out[:, 1] = np.clip(out[:, 1], 3e14, 1e15)
    out[:, 1] = np.maximum(out[:, 1], out[:, 0] * 1.5)
    out[:, 2:5] = np.clip(out[:, 2:5], 1.2, 3.5)
    out[:, 5] = np.clip(out[:, 5], 1e33, 1e36)
    return out
