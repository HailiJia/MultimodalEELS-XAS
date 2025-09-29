# train.py

import math
import random
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# XGBoost + Hyperopt
# =========================
class BOhyper_reg(object):
    """
    Bayesian optimization wrapper (Hyperopt) for XGBoost regression.
    """

    def __init__(self, X_train, X_test, y_train, y_test, random_state=1, n_splits=5):
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.best_rmse = float("inf")
        self.best_model = None
        self.n_splits = n_splits
        self.random_state = random_state

        # Hyperopt search space (note: quniform returns floats; cast ints in objective)
        self.space = {
            'learning_rate': hp.uniform('learning_rate', 0.02, 0.5),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.quniform('subsample', 0.6, 0.9, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.9, 0.05),
            'gamma': hp.quniform('gamma', 0.0, 5.0, 0.5),
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            'reg_lambda': hp.quniform('reg_lambda', 0.0, 2.0, 0.1),
            'reg_alpha': hp.quniform('reg_alpha', 0.0, 10.0, 1.0),
            'objective': 'reg:squarederror',
            'n_estimators': hp.choice('n_estimators', [50, 100, 150, 300, 500, 800, 1000]),
            'seed': random_state
        }

    def _build_model(self, params):
        """Construct an XGBRegressor from a sampled hyperparameter dict."""
        model = xgb.XGBRegressor(
            learning_rate=float(params['learning_rate']),
            min_child_weight=int(params['min_child_weight']),
            subsample=float(params['subsample']),
            colsample_bytree=float(params['colsample_bytree']),
            gamma=float(params['gamma']),
            max_depth=int(params['max_depth']),
            reg_alpha=float(params['reg_alpha']),
            reg_lambda=float(params['reg_lambda']),
            n_estimators=int(params['n_estimators']),
            objective=params['objective'],
            random_state=self.random_state,
            tree_method="hist",          # change to "gpu_hist" if on GPU
            eval_metric="rmse",
        )
        return model

    def objective(self, params):
        """
        Hyperopt objective: k-fold CV RMSE on the training set.
        Also tracks and stores the best model.
        """
        model = self._build_model(params)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        rmse_scores = []

        for train_index, val_index in kf.split(self.X_train):
            X_tr, X_val = self.X_train[train_index], self.X_train[val_index]
            y_tr, y_val = self.y_train[train_index], self.y_train[val_index]

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=25,
                verbose=False
            )
            pred = model.predict(X_val)
            rmse_scores.append(math.sqrt(mean_squared_error(y_val, pred)))

        avg_rmse = float(np.mean(rmse_scores))
        if avg_rmse < self.best_rmse:
            self.best_rmse = avg_rmse
            self.best_model = model

        return {'loss': avg_rmse, 'status': STATUS_OK}

    def search(self, max_evals=100, show_space=False):
        """
        Run Hyperopt search and return the best fitted model.
        """
        trials = Trials()
        best_hyperparams = fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_state)
        )
        if show_space:
            print("Best hyperparameters (Hyperopt space indices where applicable):")
            print(best_hyperparams)
        return self.best_model


def evaluate(model, X_test, y_test, verbose=True):
    """
    Evaluate a fitted XGBoost model on a held-out test set.
    Returns RMSE and R^2.
    """
    preds = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    if verbose:
        print("Test RMSE & R2:", rmse, r2)
    return rmse, r2, preds


# =========================
# PyTorch: MLP & 1D-CNN
# =========================
class ArrayDataset(Dataset):
    """Wrap NumPy arrays X, y; if cnn=True, adds channel dim for 1D-CNN."""
    def __init__(self, X, y, cnn=False):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.cnn = cnn
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.cnn and x.ndim == 1:
            x = x[None, ...]  # (L,) -> (1,L)
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])


class MLPReg(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.1):
        super(MLPReg, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, x):  # x: (B,L) or (B,1,L)
        if x.ndim == 3:
            x = x.squeeze(1)
        return self.net(x)


class XANESCNNReg(nn.Module):
    """Simple 1D-CNN regressor for spectra shaped (B,1,L)."""
    def __init__(self, in_channels=1):
        super(XANESCNNReg, self).__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> (B,64,1)
        )
        self.head = nn.Linear(64, 1)
    def forward(self, x):
        z = self.fe(x).squeeze(-1)  # (B,64)
        return self.head(z)         # (B,1)


def train_torch_model(model, train_loader, val_loader,
                      lr=1e-3, weight_decay=1e-5, epochs=100, patience=15, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best = {"loss": float("inf"), "state": None, "epoch": 0}
    hist = {"train_loss": [], "val_loss": []}
    patience_left = patience

    for ep in range(1, epochs + 1):
        # train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # evaluate
        model.eval()
        with torch.no_grad():
            tr_sum, tr_n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                l = loss_fn(model(xb), yb).item()
                tr_sum += l * yb.size(0); tr_n += yb.size(0)
            tr_loss = tr_sum / max(1, tr_n)

            va_sum, va_n = 0.0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                l = loss_fn(model(xb), yb).item()
                va_sum += l * yb.size(0); va_n += yb.size(0)
            va_loss = va_sum / max(1, va_n)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)

        if va_loss < best["loss"] - 1e-6:
            best["loss"] = va_loss
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, hist, best


def torch_predict(model, loader, device="cpu"):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().ravel()
            preds.append(pred)
            ys.append(yb.numpy().ravel())
    preds = np.concatenate(preds, axis=0)
    ys = np.concatenate(ys, axis=0)
    return preds, ys


def evaluate_torch(preds, ys, verbose=True):
    rmse = math.sqrt(mean_squared_error(ys, preds))
    r2 = r2_score(ys, preds)
    if verbose:
        print("Test RMSE & R2:", rmse, r2)
    return rmse, r2


# =========================
# Main switch: XGB / MLP / CNN
# =========================
if __name__ == "__main__":
    # Load data after featurization
    # data_X, label_y = ...   # (N, F), (N,)
    # For CNN mode, supply data_X_raw shaped (N, L) (full spectra 1D).

    # ---- choose one: 'xgb', 'mlp', or 'cnn'
    mode = "xgb"    # change to "mlp" or "cnn" as needed

    # Example split (replace data_X/label_y with your actual arrays)
    # Note: keep random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, label_y, test_size=0.2, random_state=random.randint(0, 100)
    )

    if mode == "xgb":
        BO = BOhyper_reg(X_train, X_test, y_train, y_test)
        best_model = BO.search(max_evals=100, show_space=True)
        rmse, r2, preds = evaluate(best_model, X_test, y_test)

    elif mode == "mlp":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # You can plug PCA-reduced features here if you prefer
        Xtr, Xva, ytr, yva = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
        train_ds = ArrayDataset(Xtr, ytr, cnn=False)
        val_ds   = ArrayDataset(Xva, yva, cnn=False)
        test_ds  = ArrayDataset(X_test, y_test, cnn=False)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

        mlp = MLPReg(in_dim=Xtr.shape[1], hidden=128, dropout=0.1)
        mlp, hist, best = train_torch_model(mlp, train_loader, val_loader,
                                            lr=1e-3, weight_decay=1e-5,
                                            epochs=200, patience=20, device=device)
        preds, ys = torch_predict(mlp, test_loader, device=device)
        rmse, r2 = evaluate_torch(preds, ys)

    elif mode == "cnn":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Expect raw 1D spectra in data_X_raw with shape (N, L)
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            data_X_raw, label_y, test_size=0.2, random_state=0
        )
        Xtr, Xva, ytr, yva = train_test_split(X_train_raw, y_train_raw, test_size=0.2, random_state=0)

        train_ds = ArrayDataset(Xtr, ytr, cnn=True)
        val_ds   = ArrayDataset(Xva, yva, cnn=True)
        test_ds  = ArrayDataset(X_test_raw, y_test_raw, cnn=True)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

        cnn = XANESCNNReg(in_channels=1)
        cnn, hist, best = train_torch_model(cnn, train_loader, val_loader,
                                            lr=1e-3, weight_decay=1e-5,
                                            epochs=200, patience=20, device=device)
        preds, ys = torch_predict(cnn, test_loader, device=device)
        rmse, r2 = evaluate_torch(preds, ys)

    else:
        raise ValueError("mode must be one of: 'xgb', 'mlp', 'cnn'")
