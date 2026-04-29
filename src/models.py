import numpy as np
import GPy
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.neural_network import MLPRegressor

# Model: GPR_RBF
class GPR_RBF:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        kernel = GPy.kern.RBF(input_dim=X.shape[1],ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

    def fit_high_bias_variance(self, X, y, lengthscale_weight):
        y = y.reshape(-1, 1)
        kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

        optimized_noise = float(self.model.Gaussian_noise.variance.values[0])
        kernel = GPy.kern.RBF(
            input_dim=X.shape[1],
            ARD=True,
            variance=kernel.variance,
            lengthscale=kernel.lengthscale * lengthscale_weight)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.Gaussian_noise.variance = optimized_noise

    def predict(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=True)
        y_std = np.sqrt(y_var)
        return y_mean.flatten(), y_std.flatten()

    def predict_noiseless(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=False)
        y_std = np.sqrt(y_var)
        return y_mean.flatten(), y_std.flatten()


class GPR_Matern:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        kernel = GPy.kern.Matern52(input_dim=X.shape[1], ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

    def predict(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=True)
        y_std = np.sqrt(y_var)
        return y_mean.flatten(), y_std.flatten()

    def predict_noiseless(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=False)
        y_std = np.sqrt(y_var)
        return y_mean.flatten(), y_std.flatten()

def gpr_pred_mean_std(model_f1, model_f2, X_test, noiseless=False, verbose=True):
    if noiseless:
        mean_f1, std_f1 = model_f1.predict_noiseless(X_test)
        mean_f2, std_f2 = model_f2.predict_noiseless(X_test)
    else:
        mean_f1, std_f1 = model_f1.predict(X_test)
        mean_f2, std_f2 = model_f2.predict(X_test)

    mean_f1 = np.asarray(mean_f1).reshape(-1)
    std_f1 = np.asarray(std_f1).reshape(-1)
    mean_f2 = np.asarray(mean_f2).reshape(-1)
    std_f2 = np.asarray(std_f2).reshape(-1)

    pred_mean = np.stack([mean_f1, mean_f2], axis=1)
    pred_std = np.stack([std_f1, std_f2], axis=1)

    if verbose:
        tag = "noiseless" if noiseless else "with_noise"

    return pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2


def autogluon_qr_fit_predict(X_train, y_train, X_test, quantile_levels=None, random_state=42):
    if quantile_levels is None:
        quantile_levels = [0.5, 0.8, 0.9, 0.95]

    train_df = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(X_train.shape[1])])
    train_df["target"] = y_train

    model = TabularPredictor(
        label="target",
        problem_type="quantile",
        quantile_levels=quantile_levels
    ).fit(
        train_data=train_df,
        verbosity=0,
        ag_args_fit={"random_state": random_state}
    )

    quantile_pred = autogluon_qr_predict(model, X_test)
    return quantile_pred, model


def autogluon_qr_predict(model, X):
    test_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    pred = model.predict(test_df)
    pred.columns = [f"y_q{q}" for q in pred.columns]
    return pred


def autogluon_qr_pred_mean_quantiles(model_f1, model_f2, X_test, verbose=True):
    pred_y1 = autogluon_qr_predict(model_f1, X_test)
    pred_y2 = autogluon_qr_predict(model_f2, X_test)

    mean_q = np.stack([pred_y1["y_q0.5"].values, pred_y2["y_q0.5"].values], axis=1)
    q80 = np.stack([pred_y1["y_q0.8"].values, pred_y2["y_q0.8"].values], axis=1)
    q90 = np.stack([pred_y1["y_q0.9"].values, pred_y2["y_q0.9"].values], axis=1)
    q95 = np.stack([pred_y1["y_q0.95"].values, pred_y2["y_q0.95"].values], axis=1)

    if verbose:
        print("[QR] y_q50\n", mean_q[:5])
        print("[QR] y_q80\n", q80[:5])
        print("[QR] y_q90\n", q90[:5])
        print("[QR] y_q95\n", q95[:5])

    return mean_q, q80, q90, q95


class BNNEnsembleRegressor:
    def __init__(self, hidden_layer_sizes=(64, 64), n_estimators=5, max_iter=2000, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        self.models = []
        for i in range(self.n_estimators):
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                max_iter=self.max_iter,
                random_state=self.random_state + i
            )
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        if len(self.models) == 0:
            raise ValueError("Model is not fitted yet.")
        preds = np.stack([m.predict(X) for m in self.models], axis=1)
        mean = preds.mean(axis=1)
        std = preds.std(axis=1)
        return mean, std


def bnn_pred_mean_std(model_f1, model_f2, X_test, verbose=True):
    mean_f1, std_f1 = model_f1.predict(X_test)
    mean_f2, std_f2 = model_f2.predict(X_test)

    mean_f1 = np.asarray(mean_f1).reshape(-1)
    std_f1 = np.asarray(std_f1).reshape(-1)
    mean_f2 = np.asarray(mean_f2).reshape(-1)
    std_f2 = np.asarray(std_f2).reshape(-1)

    pred_mean = np.stack([mean_f1, mean_f2], axis=1)
    pred_std = np.stack([std_f1, std_f2], axis=1)

    if verbose:
        print("[BNN] pred_mean\n", pred_mean[:5])
        print("[BNN] pred_std\n", pred_std[:5])
        print("[BNN] Max pred_std\n", np.max(pred_std, axis=0))

    return pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2
