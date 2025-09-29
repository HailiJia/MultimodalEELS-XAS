import math
import random
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class BOhyper_reg(object):
    """
    Bayesian optimization wrapper (Hyperopt) for XGBoost regression.

    Attributes
    ----------
    X_train, X_test, y_train, y_test : arrays
        Train/test splits.
    best_rmse : float
        Best CV RMSE observed so far.
    best_model : xgb.XGBRegressor or None
        Model corresponding to best_rmse.
    space : dict
        Hyperparameter search space for Hyperopt.
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
            tree_method="hist",          # fast default; change to "gpu_hist" if on GPU
            eval_metric="rmse",
        )
        return model

    def objective(self, params):
        """
        Hyperopt objective: 5-fold CV RMSE on the training set.
        Also tracks and stores the best model.
        """
        model = self._build_model(params)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        rmse_scores = []

        for train_index, val_index in kf.split(self.X_train):
            X_tr, X_val = self.X_train[train_index], self.X_train[val_index]
            y_tr, y_val = self.y_train[train_index], self.y_train[val_index]

            # Early stopping on the fold's validation set
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=25,
                verbose=False
            )

            pred = model.predict(X_val)
            rmse_scores.append(math.sqrt(mean_squared_error(y_val, pred)))

        avg_rmse = float(np.mean(rmse_scores))

        # Track the best model seen so far (clone current fitted model)
        if avg_rmse < self.best_rmse:
            self.best_rmse = avg_rmse
            self.best_model = model

        return {'loss': avg_rmse, 'status': STATUS_OK}

    def search(self, max_evals=100, show_space=False):
        """
        Run Hyperopt search and return the best fitted model.

        Parameters
        ----------
        max_evals : int
            Number of Hyperopt trials.
        show_space : bool
            If True, prints the best hyperparameters dict returned by Hyperopt.

        Returns
        -------
        best_model : xgb.XGBRegressor
            Best model (already fitted during CV on the best fold).
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
    Evaluate a fitted model on a held-out test set.
    Returns RMSE and R^2.
    """
    preds = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    if verbose:
        print("Test RMSE & R2:", rmse, r2)
    return rmse, r2, preds


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = train_test_split(
        data_X, label_y, test_size=0.2, random_state=random.randint(0, 100)
    )
    
    BO = BOhyper_reg(X_train, X_test, y_train, y_test)
    best_model = BO.search(max_evals=100, show_space=True)
    
    rmse, r2, preds = evaluate(best_model, X_test, y_test)
