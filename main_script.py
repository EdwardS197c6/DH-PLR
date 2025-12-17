import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import SplineTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import RepeatedKFold

# -----------------------------
#
# -----------------------------
hyperparams = {
    'scad_gamma': 3.7,
    'scad_C_grid': list(np.linspace(0.1, 3.0, 100)),
    'mlp_hidden_layers': (64, 128, 32),
    'mlp_activation': 'relu',
    'mlp_solver': 'adam',
    'mlp_alpha': 0.0,
    'mlp_learning_rate_init': 1e-3,
    'mlp_max_iter': 30,
    'rf_n_trees': 500,
    'early_stop_patience': 10,
    'cv_folds': 5,
    'pinball_tau': 0.05,
}

# -----------------------------
# 1)
# -----------------------------
def fit_mlp_with_regularization(Y, X):
    mlp = MLPRegressor(
        hidden_layer_sizes=hyperparams['mlp_hidden_layers'],
        activation=hyperparams['mlp_activation'],
        solver=hyperparams['mlp_solver'],
        alpha=hyperparams['mlp_alpha'],
        learning_rate_init=hyperparams['mlp_learning_rate_init'],
        max_iter=hyperparams['mlp_max_iter'],
        #early_stopping=False,
        warm_start=True,
        early_stopping=True,
        n_iter_no_change=hyperparams['early_stop_patience'],
        random_state=0
    )
    mlp.fit(X, Y)
    return mlp

# -----------------------------
# 2)
# -----------------------------
def scad_threshold(h, v, lam, a=hyperparams['scad_gamma']):
    """
    Univariate SCAD solution for coordinate descent:
    h = z_j^T r_j, v = z_j^T z_j.
    """
    if v == 0:
         return 0.0
    def S(x, t): return np.sign(x) * max(abs(x) - t, 0)
    if abs(h) <= lam * v:
        return S(h, lam * v) / v
    elif abs(h) <= a * lam * v:
        return S(h, a * lam * v / (a - 1)) / (v * (1 - 1/(a - 1)))
    else:
        return h / v

# -----------------------------
# 3)
# -----------------------------
def coordinate_descent_beta(Y, Z, beta, lam, a=hyperparams['scad_gamma'],
                            max_iter=100, eps=1e-6):
    p = Z.shape[1]
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # partial residual (add back jth contribution)
            r_j = Y - Z.dot(beta) + Z[:, j] * beta[j]
            h_j = Z[:, j].dot(r_j)
            v_j = Z[:, j].dot(Z[:, j])
            beta[j] = scad_threshold(h_j, v_j, lam, a)
        if np.linalg.norm(beta - beta_old) < eps:
            break
    return beta

# -----------------------------
# 4)
# -----------------------------
def AltAdamLLA(Y, Z, beta0, theta0, lam_n,
               a=hyperparams['scad_gamma'],
               max_iter=100, eps=1e-6):
    """
    Local Linear Approximation (LLA) for SCAD via coordinate descent.
    """
    beta = beta0.copy()
    for _ in range(hyperparams['early_stop_patience']):
        beta = coordinate_descent_beta(Y, Z, beta, lam_n, a, max_iter, eps)
    return beta, None

# -----------------------------
# 5)
# -----------------------------
def generate_data(n, d, p, rho,
                  scenario='A1', noise='gaussian',
                  standardize_g=True, seed=None,
                  beta0_override=None):
    
    # 1)
    if seed is not None:
        np.random.seed(seed)

    
    # 2)
    D = p + d
    Sigma = rho ** np.abs(np.arange(D)[:, None] - np.arange(D)[None, :])
    W = multivariate_normal.rvs(mean=np.zeros(D), cov=Sigma, size=n)
    Z, X = W[:, :p], W[:, p:]

    # 3)
    if scenario == 'A1':
        # Additive: sum_{j=1}^k sin(X_j)
        g = np.sum(np.sin(X[:, :k]), axis=1)

    elif scenario == 'B1':
        # Quadratic form: sum_{j=1}^k X_j * X_{j+1}
        idx = np.arange(k)
        g = np.sum(X[:, idx] * X[:, idx + 1], axis=1)

    elif scenario == 'C1':
        # High-order interaction:
        coeff = 6 * np.pi / (k * (k + 1))
        weights = np.arange(1, k + 1)
        g = 5 * np.sin(coeff * (X[:, :k] @ weights))

    elif scenario == 'D1':
        #
        # group 1
        t1 = X[:, 0]**2 * X[:, 1]**3
        t2 = np.log1p(np.abs(X[:, 2]))
        t3 = np.sqrt(np.maximum(0, 1 + X[:, 3] * X[:, 4]))
        t4 = np.exp(X[:, 4] / 2)
        # group 2
        t5 = X[:, 5]**2 * X[:, 6]**3
        t6 = np.log1p(np.abs(X[:, 7]))
        t7 = np.sqrt(np.maximum(0, 1 + X[:, 8] * X[:, 9]))
        t8 = np.exp(X[:, 9] / 2)
        g = 2.5 * (t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # 5) standardlized g
    if standardize_g:
        gm, gs = g.mean(), g.std()
        if gs > 0:
            g = (g - gm) / gs

    # 6) noise
    eps = np.random.randn(n) if noise == 'gaussian' else np.random.standard_t(df=3, size=n)

    # 7) linear part β0
    b0 = beta0_override if (beta0_override is not None) else beta0
    if len(b0) < p:
        raise ValueError(f"beta0 length {len(b0)} < p={p}")

    # 8)
    Y = Z.dot(b0[:p]) + g + eps

    return Y, Z, X



# -----------------------------
# 6)
# -----------------------------
def fit_deep_plm_alternating(data, lam,
                             max_outer=None, tol=1e-3):
    if max_outer is None:
        max_outer = hyperparams['early_stop_patience']
    Z = np.vstack([z for _,z,_ in data])
    X = np.vstack([x for _,_,x in data])
    Y = np.array([y for y,_,_ in data])
    beta = np.zeros(Z.shape[1])
    loss_history = []
    for _ in range(max_outer):
    # 1)
        resid = Y - Z.dot(beta)
        mlp = fit_mlp_with_regularization(resid, X)
        g_hat = mlp.predict(X)
        # 2) 更新 beta
        beta_new, _ = AltAdamLLA(Y - g_hat, Z, beta, theta0, lam)
        # 3) 记录当轮损失
        total_preds = Z.dot(beta_new) + g_hat
        loss_history.append( loss_mse(Y, total_preds) )
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    return mlp, beta, loss_history

# -----------------------------
# 7)
# -----------------------------
def loss_mse(Y, preds):
    preds = preds.reshape(Y.shape)
    return np.mean(0.5 * (Y - preds)**2)

def loss_pinball(Y, preds, tau=None):
    if tau is None:
        tau = hyperparams['pinball_tau']
    u = Y - preds
    return np.mean(u * (tau - (u < 0).astype(float)))

# -----------------------------
# 8)
# -----------------------------
def compute_combined_predictions(mlp, X, beta, Z):
    g_hat = mlp.predict(X)
    return g_hat + Z.dot(beta)

# -----------------------------
# 9)
# -----------------------------

def cv_deeppdplm_lambda(Y, Z, X, Cs, n_splits=None, n_repeats=3):
    if n_splits is None:
        n_splits = hyperparams['cv_folds']
    rkf = RepeatedKFold(n_splits=n_splits,
                        n_repeats=n_repeats,
                        random_state=42)

    p = Z.shape[1]
    best_lam, best_mse = None, np.inf

    
    for C in Cs:
        lam = C * np.sqrt(np.log(p)/len(Y))
        mses = []
        
        for fold, (tr, va) in enumerate(rkf.split(Y), start=1):
            data_tr = list(zip(Y[tr], Z[tr], X[tr]))
            mlp, beta_tr, _ = fit_deep_plm_alternating(data_tr, lam)
            preds_va = compute_combined_predictions(mlp, X[va], beta_tr, Z[va])
            this_mse = loss_mse(Y[va], preds_va)
            mses.append(this_mse)
            
        avg_mse = np.mean(mses)
        
        if avg_mse < best_mse:
            best_mse, best_lam = avg_mse, lam

    
    return best_lam


# -----------------------------
# 10)
# -----------------------------
def estimation_error(beta_est, beta_true):
    return np.linalg.norm(beta_est - beta_true)

def compute_support_metrics(beta_est, beta_true, thr=0.01):
    from sklearn.metrics import confusion_matrix
    est_nz  = np.abs(beta_est)  > thr
    true_nz = np.abs(beta_true) > thr
    cm = confusion_matrix(true_nz, est_nz)
    TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
    TPR = TP/(TP+FN) if TP+FN>0 else 0
    FPR = FP/(FP+TN) if FP+TN>0 else 0
    F1  = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN>0 else 0
    return TPR, FPR, F1, FP

# -----------------------------
# 11)
# -----------------------------
beta0  = np.zeros(600)
beta0[:5]   =  1.5
beta0[5:10] = -1.5
theta0 = np.zeros(100)

# -----------------------------
#
# ----------------------------
def fit_deep_plm_quantile(data, lam, tau=hyperparams['pinball_tau'],
                          max_outer=None, tol=1e-3):
    
    if max_outer is None:
        max_outer = hyperparams['early_stop_patience']
    Z = np.vstack([z for _,z,_ in data])
    X = np.vstack([x for _,_,x in data])
    Y = np.array([y for y,_,_ in data])
    beta = np.zeros(Z.shape[1])
    for _ in range(max_outer):
        # 1)
        resid = Y - Z.dot(beta)
        #
        mlp = fit_mlp_with_regularization(resid * np.where(resid>=0, tau, 1-tau), X)
        g_hat = mlp.predict(X)
        # 2)
        beta_new = fit_scad_only_quantile(Y - g_hat, Z, lam, tau)
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    return mlp, beta


# -----------------------------
# 12)
# -----------------------------

def cv_deeppdplm_quantile(Y, Z, X, Cs, tau=hyperparams['pinball_tau'], n_splits=None):
    
    if n_splits is None:
        n_splits = hyperparams['cv_folds']
    p = Z.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_lam, best_loss = None, float('inf')
    for C in Cs:
        lam = C * np.sqrt(np.log(p) / len(Y))
        fold_losses = []
        for tr, va in kf.split(Y):
            data_tr = list(zip(Y[tr], Z[tr], X[tr]))
            mlp, beta_tr = fit_deep_plm_quantile(data_tr, lam, tau)
            preds_va = compute_combined_predictions(mlp, X[va], beta_tr, Z[va])
            fold_losses.append(loss_pinball(Y[va], preds_va, tau))
        avg_loss = np.mean(fold_losses)
        if avg_loss < best_loss:
            best_loss, best_lam = avg_loss, lam
    return best_lam
