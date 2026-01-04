import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPRegressor
from scipy.stats import multivariate_normal

# =============================
# 0) basic settings
# =============================
n = 600     # number of observations
p = 200     # linear dimension
d = 20      # nonlinear dimension
rho = 0.3   # AR(1) correlation
scenario = "A1"
seed = 123

# =============================
# 1) hyperparams (keep minimal)
# =============================
hyperparams = {
    "scad_gamma": 3.7,
    "C_grid": np.linspace(0.1, 3.0, 30),
    "cv_folds": 5,
    "mlp_hidden_layers": (64, 128, 32),
    "mlp_max_iter": 30,
    "early_stop_patience": 10,
}

# =============================
# 2) true beta (example)
# =============================
beta0 = np.zeros(p)
beta0[:5] = 1.5
beta0[5:10] = -1.5

# =============================
# 3) helpers
# =============================
def loss_mse(y, yhat):
    return np.mean(0.5 * (y - yhat) ** 2)

def scad_threshold(h, v, lam, a=hyperparams["scad_gamma"]):
    if v == 0:
        return 0.0
    def S(x, t):
        return np.sign(x) * max(abs(x) - t, 0.0)
    if abs(h) <= lam * v:
        return S(h, lam * v) / v
    elif abs(h) <= a * lam * v:
        return S(h, a * lam * v / (a - 1.0)) / (v * (1.0 - 1.0 / (a - 1.0)))
    else:
        return h / v

def coordinate_descent_beta(Y, Z, beta, lam, max_iter=100, eps=1e-6):
    p = Z.shape[1]
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r_j = Y - Z.dot(beta) + Z[:, j] * beta[j]
            h_j = Z[:, j].dot(r_j)
            v_j = Z[:, j].dot(Z[:, j])
            beta[j] = scad_threshold(h_j, v_j, lam)
        if np.linalg.norm(beta - beta_old) < eps:
            break
    return beta

def fit_mlp(resid, X):
    mlp = MLPRegressor(
        hidden_layer_sizes=hyperparams["mlp_hidden_layers"],
        activation="relu",
        solver="adam",
        alpha=0.0,
        learning_rate_init=1e-3,
        max_iter=hyperparams["mlp_max_iter"],
        early_stopping=True,
        n_iter_no_change=hyperparams["early_stop_patience"],
        random_state=0,
    )
    mlp.fit(X, resid)
    return mlp

def generate_data(n, p, d, rho, scenario="A1", seed=123, k_nonlin=None):
    np.random.seed(seed)
    if k_nonlin is None:
        k_nonlin = min(10, d)

    D = p + d
    Sigma = rho ** np.abs(np.arange(D)[:, None] - np.arange(D)[None, :])
    W = multivariate_normal.rvs(mean=np.zeros(D), cov=Sigma, size=n)
    Z, X = W[:, :p], W[:, p:]

    k = k_nonlin
    if scenario == "A1":
        g = np.sum(np.sin(X[:, :k]), axis=1)
    elif scenario == "B1":
        g = np.sum(X[:, :k] * X[:, 1:k+1], axis=1)
    else:
        raise ValueError("Unknown scenario")

    g = (g - g.mean()) / (g.std() + 1e-12)
    eps = np.random.randn(n)
    Y = Z.dot(beta0) + g + eps
    return Y, Z, X

def fit_deep_plm_alternating(Y, Z, X, lam, max_outer=10, tol=1e-3):
    beta = np.zeros(Z.shape[1])
    for _ in range(max_outer):
        resid = Y - Z.dot(beta)
        mlp = fit_mlp(resid, X)
        ghat = mlp.predict(X)

        beta_new = coordinate_descent_beta(Y - ghat, Z, beta.copy(), lam, max_iter=200)
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    return mlp, beta

def cv_lambda(Y, Z, X, C_grid, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    p = Z.shape[1]
    best_lam, best_score = None, np.inf

    for C in C_grid:
        lam = C * np.sqrt(np.log(p) / len(Y))
        scores = []
        for tr, va in kf.split(Y):
            mlp, beta_hat = fit_deep_plm_alternating(Y[tr], Z[tr], X[tr], lam, max_outer=8)
            pred_va = mlp.predict(X[va]) + Z[va].dot(beta_hat)
            scores.append(loss_mse(Y[va], pred_va))
        score = float(np.mean(scores))
        if score < best_score:
            best_score, best_lam = score, lam

    return best_lam

# =============================
# 4) main pipeline (R-like)
# =============================
Y, Z, X = generate_data(n, p, d, rho, scenario=scenario, seed=seed)

Y_tr, Y_te, Z_tr, Z_te, X_tr, X_te = train_test_split(
    Y, Z, X, test_size=0.3, random_state=42
)

lam_hat = cv_lambda(Y_tr, Z_tr, X_tr, hyperparams["C_grid"], n_splits=hyperparams["cv_folds"])
mlp_hat, beta_hat = fit_deep_plm_alternating(Y_tr, Z_tr, X_tr, lam_hat, max_outer=hyperparams["early_stop_patience"])

pred_te = mlp_hat.predict(X_te) + Z_te.dot(beta_hat)

print("lambda_hat =", lam_hat)
print("test_mse   =", loss_mse(Y_te, pred_te))
print("||beta_hat - beta0||2 =", np.linalg.norm(beta_hat - beta0))
print("beta0[:12]   =", beta0[:12])
print("beta_hat[:12]=", beta_hat[:12])
