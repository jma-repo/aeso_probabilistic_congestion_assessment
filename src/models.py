"""Surrogate models for DC-OPF outputs.

1. MeritOrderBaseline  - one system price, no network (copper plate).
2. LGBMPerBus          - one LightGBM regressor per bus (+ TreeSHAP contributions).
3. MLPSurrogate        - one PyTorch MLP predicting all buses at once.
4. LGBMCongestion      - one LightGBM classifier per line: P(line at limit).

All models are fitted on a training set, early-stopped on a validation set
and only then evaluated on the test set.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn



class MeritOrderBaseline:
    name = "baseline (copper plate)"

    fit_seconds = 0.0

    def __init__(self, buses: list[str]):
        self.buses = buses

    def fit(self, *a, **k):
        return self  # nothing to learn: the merit order is computed analytically

    def predict(self, X: pd.DataFrame, **kw) -> pd.DataFrame:
        """Uses the ``merit_price`` feature (see features.merit_order_price)."""
        p = X["merit_price"].to_numpy()
        return pd.DataFrame({f"lmp_{b}": p for b in self.buses}, index=X.index)


class LGBMPerBus:
    name = "LightGBM (per bus)"

    def __init__(self, params: dict, num_boost_round: int = 3000, early_stopping: int = 100):
        self.params = {"objective": "regression", "verbosity": -1, **params}
        self.rounds, self.es = num_boost_round, early_stopping
        self.boosters: dict[str, lgb.Booster] = {}
        self.fit_seconds = 0.0

    def fit(self, Xtr, Ytr, Xva, Yva):
        t0 = time.perf_counter()
        for col in Ytr:
            mtr, mva = Ytr[col].notna().to_numpy(), Yva[col].notna().to_numpy()
            dtr = lgb.Dataset(Xtr[mtr], Ytr[col][mtr])
            dva = lgb.Dataset(Xva[mva], Yva[col][mva], reference=dtr)
            self.boosters[col] = lgb.train(self.params, dtr, self.rounds, valid_sets=[dva],
                                           callbacks=[lgb.early_stopping(self.es, verbose=False)])
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict(self, X, threads: int | None = None) -> pd.DataFrame:
        kw = {"num_threads": threads} if threads else {}
        return pd.DataFrame({c: b.predict(X, num_iteration=b.best_iteration, **kw) for c, b in self.boosters.items()},
                            index=X.index)

    def shap_values(self, X, col: str) -> pd.DataFrame:
        """Exact TreeSHAP (Lundberg et al.) as implemented inside LightGBM."""
        b = self.boosters[col]
        contrib = b.predict(X, num_iteration=b.best_iteration, pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], index=X.index, columns=X.columns)


class LGBMCongestion:
    name = "LightGBM classifier (per line)"

    def __init__(self, params: dict, num_boost_round: int = 2000, early_stopping: int = 100):
        self.params = {"objective": "binary", "verbosity": -1, **params}
        self.rounds, self.es = num_boost_round, early_stopping
        self.boosters: dict[str, lgb.Booster | float] = {}
        self.fit_seconds = 0.0

    def fit(self, Xtr, Btr, Xva, Bva):
        t0 = time.perf_counter()
        for line in Btr:
            pos = int(Btr[line].sum())
            if pos < 5 or pos > len(Btr) - 5:
                # (almost) never/always binding in training: predict the base rate
                self.boosters[line] = float(Btr[line].mean())
                continue
            dtr = lgb.Dataset(Xtr, Btr[line])
            dva = lgb.Dataset(Xva, Bva[line], reference=dtr)
            self.boosters[line] = lgb.train(self.params, dtr, self.rounds, valid_sets=[dva],
                                            callbacks=[lgb.early_stopping(self.es, verbose=False)])
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict_proba(self, X, threads: int | None = None) -> pd.DataFrame:
        kw = {"num_threads": threads} if threads else {}
        out = {}
        for line, b in self.boosters.items():
            out[line] = np.full(len(X), b) if isinstance(b, float) else b.predict(X, num_iteration=b.best_iteration, **kw)
        return pd.DataFrame(out, index=X.index)


class _Net(nn.Module):
    def __init__(self, n_in, n_out, hidden):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class MLPSurrogate:
    """Multi-output MLP with standardised inputs/targets, masked MSE (undefined
    LMPs of islanded buses are ignored) and early stopping on validation loss."""

    name = "MLP (multi-output)"

    def __init__(self, hidden=(256, 256, 128), lr=1e-3, weight_decay=1e-5, batch_size=512,
                 max_epochs=400, patience=25, seed=0, threads=4, loss="mse", huber_delta=1.0):
        self.hidden, self.lr, self.wd = tuple(hidden), lr, weight_decay
        self.bs, self.max_epochs, self.patience, self.seed, self.threads = batch_size, max_epochs, patience, seed, threads
        self.loss, self.huber_delta = loss, huber_delta
        self.history: list[tuple[int, float, float]] = []
        self.fit_seconds = 0.0

    def _x(self, X):
        return torch.tensor(((X.to_numpy() - self.xm) / self.xs), dtype=torch.float32)

    def fit(self, Xtr, Ytr, Xva, Yva):
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        t0 = time.perf_counter()
        self.cols = list(Ytr.columns)
        self.xm, self.xs = Xtr.mean().to_numpy(), Xtr.std().replace(0, 1).to_numpy()
        self.ym, self.ys = Ytr.mean().to_numpy(), Ytr.std().replace(0, 1).to_numpy()
        xtr, xva = self._x(Xtr), self._x(Xva)
        ytr = torch.tensor((Ytr.to_numpy() - self.ym) / self.ys, dtype=torch.float32)
        yva = torch.tensor((Yva.to_numpy() - self.ym) / self.ys, dtype=torch.float32)
        mtr, mva = ~torch.isnan(ytr), ~torch.isnan(yva)
        ytr, yva = torch.nan_to_num(ytr), torch.nan_to_num(yva)
        self.net = _Net(xtr.shape[1], ytr.shape[1], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        def loss_fn(p, y, m):
            r = (p - y).abs()
            if self.loss == "huber":  # quadratic near 0, linear in the tails (robust to $999.99 events)
                d = self.huber_delta
                e = torch.where(r < d, 0.5 * r**2, d * (r - 0.5 * d))
            else:
                e = r**2
            return (e * m).sum() / m.sum()
        best, best_state, wait = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(xtr), generator=g)
            tl = 0.0
            for i in range(0, len(xtr), self.bs):
                idx = perm[i : i + self.bs]
                opt.zero_grad()
                loss = loss_fn(self.net(xtr[idx]), ytr[idx], mtr[idx])
                loss.backward()
                opt.step()
                tl += loss.item() * len(idx)
            self.net.eval()
            with torch.no_grad():
                vl = loss_fn(self.net(xva), yva, mva).item()
            self.history.append((epoch, tl / len(xtr), vl))
            if vl < best - 1e-5:
                best, wait = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_epoch = int(np.argmin([h[2] for h in self.history]))
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict(self, X, threads: int | None = None) -> pd.DataFrame:
        if threads:
            torch.set_num_threads(threads)
        self.net.eval()
        with torch.no_grad():
            p = self.net(self._x(X)).numpy() * self.ys + self.ym
        return pd.DataFrame(p, index=X.index, columns=self.cols)


class PriceSurrogate:
    """Wraps an LMP regressor. With ``hybrid=True`` the regressor learns only the
    congestion component  LMP - merit_price  (the copper-plate price is an input
    feature computed analytically), and the price is added back at prediction."""

    def __init__(self, model, hybrid: bool, label: str | None = None):
        self.model, self.hybrid = model, hybrid
        self.name = label or f"{model.name}{' hybrid' if hybrid else ''}"

    @property
    def fit_seconds(self):
        return self.model.fit_seconds

    def _t(self, X, Y):
        return Y.sub(X["merit_price"], axis=0) if self.hybrid else Y

    def fit(self, Xtr, Ytr, Xva, Yva):
        self.model.fit(Xtr, self._t(Xtr, Ytr), Xva, self._t(Xva, Yva))
        return self

    def predict(self, X, **kw) -> pd.DataFrame:
        P = self.model.predict(X, **kw)
        return P.add(X["merit_price"], axis=0) if self.hybrid else P
