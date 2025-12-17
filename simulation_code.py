import os, sys, time
import os, sys, time
from pathlib import Path
os.chdir('')
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

import main as sim  #
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import SplineTransformer

# ------------------------------------
n = 2000
param_list = [(100, 50), (1200, 100)]
scenarios  = ['A', 'B', 'C', 'D']
n_repeats  = 100

#
beta0_rand_dict = {}
beta_true_dict  = {}
np.random.seed(2025)
for p, d in param_list:
    beta0 = np.zeros(p)
    beta0[:10] = np.random.randn(10)
    beta0_rand_dict[(p,d)] = beta0
    beta_true_dict[(p,d)]  = beta0 * 3.0

#
all_records = []

#
for p, d in param_list:
    beta0_rand = beta0_rand_dict[(p,d)]
    beta_true  = beta_true_dict[(p,d)]
    Cs = sim.hyperparams['scad_C_grid']
    
    for scenario in scenarios:
        #
        Y0, Z0, X0 = sim.generate_data(
            n=n, d=d, p=p, rho=0.3,
            scenario=scenario, noise='gaussian',
            standardize_g=True, seed=0,
            beta0_override=beta0_rand
        )
        Y0_tr, Y0_te, Z0_tr, Z0_te, X0_tr, X0_te = train_test_split(
            Y0, Z0, X0, test_size=0.2, random_state=42
        )
        lam_scad = sim.cv_scad_lambda(Y0_tr, Z0_tr, Cs)
        lam_kr   = sim.cv_kernel_lambda(Y0_tr, Z0_tr, X0_tr, Cs)
        lam_sp   = sim.cv_spline_lambda(Y0_tr, Z0_tr, X0_tr, Cs)
        lam_pd   = sim.cv_deeppdplm_lambda(Y0_tr, Z0_tr, X0_tr, Cs)

        # ——  n_repeats  ——
        for seed in range(n_repeats):
            # 1) data generate and split
            Y, Z, X = sim.generate_data(
                n=n, d=d, p=p, rho=0.3,
                scenario=scenario, noise='gaussian',
                standardize_g=True, seed=seed,
                beta0_override=beta0_rand
            )
            Y_tr, Y_te, Z_tr, Z_te, X_tr, X_te = train_test_split(
                Y, Z, X, test_size=0.2, random_state=42
            )
            
            # 2)
            #
            def eval_method(name, fit_fn, predict_fn, lam=None, has_linear=True):
                t0 = time.time()
                model = fit_fn()
                preds = predict_fn(model)
                elapsed = time.time() - t0
                
                rec = {
                    'scenario': scenario,
                    'p': p, 'd': d,
                    'seed': seed,
                    'method': name,
                    'lambda': lam,
                    'mse': sim.loss_mse(Y_te, preds),
                    'pinball': sim.loss_pinball(Y_te, preds),
                    'time_sec': elapsed
                }
                if has_linear:
                    beta_est = getattr(model, 'coef_', None)
                    #
                    if beta_est is None and name=='SCAD-Only':
                        beta_est = model  # s
                    #
                    TPR, FPR, F1, FS = sim.compute_support_metrics(beta_est, beta_true)
                    rec.update({
                        'tpr': TPR,
                        'fpr': FPR,
                        'f1':  F1,
                        'false_sel': FS,
                        'est_error': sim.estimation_error(beta_est, beta_true)
                    })
                else:
                    rec.update({
                        'tpr': np.nan, 'fpr': np.nan,
                        'f1':  np.nan, 'false_sel': np.nan,
                        'est_error': np.nan
                    })
                return rec
            
            
            def fit_scad():
                return sim.fit_scad_only(Y_tr, Z_tr, lam_scad)
            def pred_scad(m):
                return Z_te.dot(m)
            all_records.append(eval_method('SCAD-Only', fit_scad, pred_scad, lam_scad, True))
            
            
            def fit_kr():
                return sim.fit_kernel_plm_alternating(Y_tr, Z_tr, X_tr, lam_kr)[:2]
            def pred_kr(m_beta):
                g_tr, beta_kr = m_beta
                resid_tr = Y_tr - Z_tr.dot(beta_kr)
                g_te = KernelRidge(alpha=lam_kr, kernel='rbf')\
                       .fit(X_tr, resid_tr).predict(X_te)
                return g_te + Z_te.dot(beta_kr)
            #
            def fit_kr_wrapper():
                g_tr, beta_kr, _ = sim.fit_kernel_plm_alternating(Y_tr, Z_tr, X_tr, lam_kr)
                return (g_tr, beta_kr)
            all_records.append(eval_method('Kernel-PLM', fit_kr_wrapper, pred_kr, lam_kr, True))
            
            #
            def fit_sp_wrapper():
                g_tr, beta_sp, _ = sim.fit_spline_plm_alternating(Y_tr, Z_tr, X_tr, lam_sp)
                return (g_tr, beta_sp)
            def pred_sp(m_beta):
                g_tr, beta_sp = m_beta
                st = SplineTransformer(n_knots=5, degree=3).fit(X_tr[:, :4])
                phi_te = st.transform(X_te[:, :4])
                coef = np.linalg.lstsq(st.transform(X_tr[:, :4]),
                                        Y_tr - Z_tr.dot(beta_sp), rcond=None)[0]
                g_te = phi_te.dot(coef)
                return g_te + Z_te.dot(beta_sp)
            all_records.append(eval_method('Spline-PLM', fit_sp_wrapper, pred_sp, lam_sp, True))
            
            #
            def fit_pd_wrapper():
                mlp, beta_pd, _ = sim.fit_deep_plm_alternating(
                    list(zip(Y_tr, Z_tr, X_tr)), lam_pd
                )
                return (mlp, beta_pd)
            def pred_pd(m_beta):
                mlp, beta_pd = m_beta
                return mlp.predict(X_te) + Z_te.dot(beta_pd)
            all_records.append(eval_method('DeepPDPLM', fit_pd_wrapper, pred_pd, lam_pd, True))
            
            # Deep-PLM baseline
            def fit_base():
                return sim.fit_mlp_with_regularization(
                    Y_tr, np.hstack([Z_tr, X_tr])
                )
            def pred_base(m):
                return m.predict(np.hstack([Z_te, X_te]))
            all_records.append(eval_method('Deep-PLM', fit_base, pred_base, None, False))

#
df = pd.DataFrame(all_records)

#
summary = df.groupby(['scenario','p','d','method'])\
            .agg(['mean','std'])\
            .round(4)


print(summary)

#
summary.to_csv('simulation_summary.csv')

