# ===== CELL 1 =====
!pip -q install catboost scikit-learn openpyxl scipy


# ===== CELL 2 =====
from google.colab import files as _gf
print('Upload BOTH files:')
print('  ammonia efficiency subset cleaned.xlsx')
print('  KL_a subset cleaned.xlsx\n')
uploaded = _gf.upload()

ETA_FILE, KLA_FILE = None, None
for fname in uploaded.keys():
    fl = fname.lower()
    if 'ammonia' in fl or 'efficiency' in fl:
        ETA_FILE = fname
    elif 'kl' in fl or 'kla' in fl:
        KLA_FILE = fname
if ETA_FILE is None or KLA_FILE is None:
    raise FileNotFoundError(f'Missing file. Uploaded: {list(uploaded.keys())}')

print(f'eta  file: {ETA_FILE}')
print(f'K_La file: {KLA_FILE}')


# ===== CELL 3 — data, split, DEFAULT model configuration (no tuning) =====
import warnings, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
from catboost                import CatBoostRegressor

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
FEAT = ['T', 'Pv', 'pH', 't']
TRACE_ITERS = 1000   # CatBoost default iteration count; trace = deployed model

eta_df = pd.read_excel(ETA_FILE)
kla_df = pd.read_excel(KLA_FILE, header=1)

def _rename(df):
    m = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if 'temp' in cl:                              m[c] = 'T'
        elif 'vacu' in cl or 'press' in cl:           m[c] = 'Pv'
        elif cl == 'ph':                              m[c] = 'pH'
        elif 'react' in cl or 'time' in cl:           m[c] = 't'
        elif 'effic' in cl or 'η' in cl:              m[c] = 'eta'
        elif ('kla' in cl or 'k_la' in cl
              or 'k la' in cl or 'kl a' in cl):       m[c] = 'KLa'
    return df.rename(columns=m)

eta_df = _rename(eta_df); kla_df = _rename(kla_df)
for col in ['T','Pv','pH','t','eta','KLa']:
    if col in eta_df.columns: eta_df[col] = pd.to_numeric(eta_df[col], errors='coerce')
    if col in kla_df.columns: kla_df[col] = pd.to_numeric(kla_df[col], errors='coerce')
eta_df = eta_df.dropna(subset=FEAT + ['eta']).reset_index(drop=True)
kla_df = kla_df.dropna(subset=FEAT + ['KLa']).reset_index(drop=True)
print(f'eta  rows: {len(eta_df)}  (expected 218)')
print(f'K_La rows: {len(kla_df)}  (expected 159)')

def split_data(df, tgt):
    bins = pd.qcut(df[tgt], q=4, labels=False, duplicates='drop')
    X, y = df[FEAT].values, df[tgt].values
    Xtr, Xtmp, ytr, ytmp, _, btmp = train_test_split(
        X, y, bins, test_size=0.20, random_state=SEED, stratify=bins)
    Xv, Xte, yv, yte = train_test_split(
        Xtmp, ytmp, test_size=0.50, random_state=SEED, stratify=btmp)
    return Xtr, Xv, Xte, ytr, yv, yte

Xe_tr, Xe_v, Xe_te, ye_tr, ye_v, ye_te = split_data(eta_df, 'eta')
Xk_tr, Xk_v, Xk_te, yk_tr, yk_v, yk_te = split_data(kla_df, 'KLa')
print(f'eta  train/val/test = {len(ye_tr)}/{len(ye_v)}/{len(ye_te)}')
print(f'K_La train/val/test = {len(yk_tr)}/{len(yk_v)}/{len(yk_te)}')

def metrics(yt, yp):
    return (r2_score(yt, yp),
            float(np.sqrt(mean_squared_error(yt, yp))),
            mean_absolute_error(yt, yp),
            float(np.corrcoef(yt, yp)[0, 1]))

# -------------------------------------------------------------------------
# DEFAULT CatBoost — one identical configuration for BOTH targets, no tuning.
# Library defaults: iterations = 1000, depth = 6, learning_rate = auto.
# The model that is SELECTED (Taylor, defaults) and VALIDATED (repeated CV,
# defaults) is therefore the same model used here and, subsequently, for SHAP
# and the response surfaces. Only the random seed is fixed.
# -------------------------------------------------------------------------
CB_ETA_PARAMS = dict()   # empty -> pure library defaults
CB_KLA_PARAMS = dict()   # empty -> pure library defaults


# ===== CELL 4 — convergence tracers + deployed models (all defaults) =====
def train_convergence_tracer(Xtr, Xv, ytr, yv, base_params):
    # trace the DEFAULT model over its default 1000-iteration horizon
    p = {**base_params, 'iterations': TRACE_ITERS}
    m = CatBoostRegressor(**p, random_seed=SEED, verbose=False, eval_metric='RMSE')
    m.fit(Xtr, ytr, eval_set=(Xv, yv))
    return m

print('\nConvergence tracer — eta ...')
cb_conv_e = train_convergence_tracer(Xe_tr, Xe_v, ye_tr, ye_v, CB_ETA_PARAMS)
rmse_tr_e = cb_conv_e.get_evals_result()['learn']['RMSE']
rmse_vl_e = cb_conv_e.get_evals_result()['validation']['RMSE']
best_iter_e = int(np.argmin(rmse_vl_e)) + 1

print('\nConvergence tracer — K_La ...')
cb_conv_k = train_convergence_tracer(Xk_tr, Xk_v, yk_tr, yk_v, CB_KLA_PARAMS)
rmse_tr_k = cb_conv_k.get_evals_result()['learn']['RMSE']
rmse_vl_k = cb_conv_k.get_evals_result()['validation']['RMSE']
best_iter_k = int(np.argmin(rmse_vl_k)) + 1

print(f'\nBest val iteration eta : {best_iter_e}  (of {TRACE_ITERS})')
print(f'Best val iteration K_La: {best_iter_k}  (of {TRACE_ITERS})')

Xe_tv = np.vstack([Xe_tr, Xe_v]); ye_tv = np.concatenate([ye_tr, ye_v])
Xk_tv = np.vstack([Xk_tr, Xk_v]); yk_tv = np.concatenate([yk_tr, yk_v])

# deployed models = pure default CatBoost, fit on train+val, tested on held-out
cb_f_e = CatBoostRegressor(**CB_ETA_PARAMS, random_seed=SEED, verbose=False)
cb_f_e.fit(Xe_tv, ye_tv)
yp_tr_e = cb_f_e.predict(Xe_tr); yp_vl_e = cb_f_e.predict(Xe_v)
yp_tv_e = cb_f_e.predict(Xe_tv); yp_te_e = cb_f_e.predict(Xe_te)
r2_tv_e, rmse_tv_e, mae_tv_e, r_tv_e = metrics(ye_tv, yp_tv_e)
r2_te_e, rmse_te_e, mae_te_e, r_te_e = metrics(ye_te, yp_te_e)

cb_f_k = CatBoostRegressor(**CB_KLA_PARAMS, random_seed=SEED, verbose=False)
cb_f_k.fit(Xk_tv, yk_tv)
yp_tr_k = cb_f_k.predict(Xk_tr); yp_vl_k = cb_f_k.predict(Xk_v)
yp_tv_k = cb_f_k.predict(Xk_tv); yp_te_k = cb_f_k.predict(Xk_te)
r2_tv_k, rmse_tv_k, mae_tv_k, r_tv_k = metrics(yk_tv, yp_tv_k)
r2_te_k, rmse_te_k, mae_te_k, r_te_k = metrics(yk_te, yp_te_k)

print(f'\neta  Test R2={r2_te_e:.4f}  RMSE={rmse_te_e:.4f}%')
print(f'K_La Test R2={r2_te_k:.4f}  RMSE={rmse_te_k:.4f} h^-1')


# ===== CELL 5 — boosting convergence & generalisation diagnostic figure =====
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines    import Line2D
from matplotlib.patches  import Patch
from scipy               import stats

C_TRAIN = '#555555'; C_VAL = '#C0706A'
C_TEST  = '#2E7D4F'; C_CI  = '#A8D5BC'; C_OLS = '#2E7D4F'

mpl.rcParams.update({
    'font.family'    : 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size'      : 9, 'axes.labelsize': 10, 'axes.titlesize': 10.5,
    'axes.linewidth' : 0.9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.fontsize': 8,  'figure.dpi': 150, 'pdf.fonttype': 42,
})

def ols_ci(x_obs, y_pred, ax, n_pts=200):
    slope, intercept, _, _, _ = stats.linregress(x_obs, y_pred)
    xfit = np.linspace(x_obs.min(), x_obs.max(), n_pts)
    yfit = slope*xfit + intercept
    n = len(x_obs)
    se = np.sqrt(np.sum((y_pred - (slope*x_obs + intercept))**2) / (n-2))
    t_cv = stats.t.ppf(0.975, df=n-2)
    xm = x_obs.mean()
    ci = t_cv*se*np.sqrt(1/n + (xfit-xm)**2 / np.sum((x_obs-xm)**2))
    ax.fill_between(xfit, yfit-ci, yfit+ci, color=C_CI, alpha=0.55)
    ax.plot(xfit, yfit, color=C_OLS, lw=1.7, zorder=4)

def kde_bar(ax, vals_list, colors, orient='v'):
    for vals, col in zip(vals_list, colors):
        kde = stats.gaussian_kde(vals, bw_method=0.35)
        if orient == 'v':
            xg = np.linspace(vals.min()-5, vals.max()+5, 300)
            ax.fill_between(xg, kde(xg), alpha=0.45, color=col)
            ax.plot(xg, kde(xg), color=col, lw=0.8)
        else:
            yg = np.linspace(vals.min()-5, vals.max()+5, 300)
            ax.fill_betweenx(yg, kde(yg), alpha=0.45, color=col)
            ax.plot(kde(yg), yg, color=col, lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def scatter_panel(ax, y_obs_tr, yp_tr, y_obs_vl, yp_vl,
                  xlabel, ylabel, letter, title,
                  r2, rmse_v, mae, r, n, pct=True):
    all_obs = np.concatenate([y_obs_tr, y_obs_vl])
    all_prd = np.concatenate([yp_tr,    yp_vl])
    lo = min(all_obs.min(), all_prd.min()) - 3
    hi = max(all_obs.max(), all_prd.max()) + 3
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, zorder=1, alpha=0.7)
    ax.scatter(y_obs_tr, yp_tr, s=24, color=C_TRAIN, alpha=0.70, zorder=3,
               label='Training', edgecolors='none')
    ax.scatter(y_obs_vl, yp_vl, s=28, color=C_VAL, alpha=0.85, zorder=4,
               label='Validation', edgecolors='white', linewidths=0.4)
    unit = ' %' if pct else r' h$^{-1}$'
    box = (f'$R^2$ = {r2:.3f}\n'
           f'RMSE = {rmse_v:.3f}{unit}\n'
           f'MAE  = {mae:.3f}{unit}\n'
           f'$r$     = {r:.3f}\n'
           f'$n$     = {n}')
    ax.text(0.04, 0.97, box, transform=ax.transAxes, va='top', ha='left',
            fontsize=7.8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec='#cccccc', alpha=0.9))
    ax.set_xlabel(xlabel, labelpad=3); ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(f'({letter}) {title}', loc='left', fontweight='bold', pad=6)
    ax.tick_params(length=3)
    ax.legend(loc='lower right', frameon=True, framealpha=0.85,
              edgecolor='#cccccc', fontsize=8)
    kde_bar(ax.inset_axes([0,    1.01, 1,    0.18]),
            [y_obs_tr, y_obs_vl], [C_TRAIN, C_VAL], 'v')
    kde_bar(ax.inset_axes([1.01, 0,    0.18, 1   ]),
            [yp_tr,    yp_vl   ], [C_TRAIN, C_VAL], 'h')

def conv_panel(ax, rmse_tr, rmse_vl, best_iter, letter, title, ylabel):
    iters = np.arange(1, len(rmse_tr) + 1)
    ax.plot(iters, rmse_tr, color='#222222', lw=1.4, label='Training RMSE',   zorder=3)
    ax.plot(iters, rmse_vl, color=C_VAL,    lw=1.4, label='Validation RMSE', zorder=3)
    ax.axvline(best_iter, color='#777', lw=0.8, ls=':', zorder=2)
    ax.scatter([best_iter], [rmse_vl[best_iter - 1]], s=40, color=C_VAL,
               zorder=5, ec='white', lw=0.5)
    ax.text(best_iter + len(rmse_tr) * 0.02, rmse_vl[best_iter - 1] * 1.04,
            f'best iter = {best_iter}', fontsize=7.5, color='#444')
    ax.set_xlabel('Boosting iteration', labelpad=3)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(f'({letter}) {title}', loc='left', fontweight='bold', pad=6)
    ax.tick_params(length=3)
    ax.legend(loc='upper right', frameon=True, framealpha=0.85,
              edgecolor='#cccccc', fontsize=8)

def test_panel(ax, y_obs, y_pred, xlabel, ylabel, letter, title,
               r2, rmse_v, mae, r, n, pct=True):
    lo = min(y_obs.min(), y_pred.min()) - 3
    hi = max(y_obs.max(), y_pred.max()) + 3
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.9, zorder=1, alpha=0.6)
    ols_ci(y_obs, y_pred, ax)
    ax.scatter(y_obs, y_pred, s=32, color=C_TEST, alpha=0.88, zorder=5,
               edgecolors='white', linewidths=0.4)
    unit = ' %' if pct else r' h$^{-1}$'
    box = (f'$R^2$ = {r2:.3f}\n'
           f'RMSE = {rmse_v:.3f}{unit}\n'
           f'MAE  = {mae:.3f}{unit}\n'
           f'$r$     = {r:.3f}\n'
           f'$n$     = {n}')
    ax.text(0.04, 0.97, box, transform=ax.transAxes, va='top', ha='left',
            fontsize=7.8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec='#cccccc', alpha=0.9))
    ax.set_xlabel(xlabel, labelpad=3); ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(f'({letter}) {title}', loc='left', fontweight='bold', pad=6)
    ax.tick_params(length=3)
    ax.legend([Line2D([],[],color=C_OLS, lw=1.7),
               Patch(color=C_CI, alpha=0.55),
               Line2D([],[],marker='o',color='w',mfc=C_TEST,ms=6,mec='white')],
              ['OLS fit', '95% CI', 'Test'],
              loc='lower right', frameon=True, framealpha=0.85,
              edgecolor='#cccccc', fontsize=8)

fig = plt.figure(figsize=(15, 9.5))
gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,
               left=0.07, right=0.97, top=0.93, bottom=0.07)

scatter_panel(fig.add_subplot(gs[0, 0]),
    ye_tr, yp_tr_e, ye_v, yp_vl_e,
    'Experimental η (%)', 'Predicted η (%)',
    'a', 'CatBoost — Training & Validation',
    r2_tv_e, rmse_tv_e, mae_tv_e, r_tv_e, len(ye_tv), pct=True)
conv_panel(fig.add_subplot(gs[0, 1]),
    rmse_tr_e, rmse_vl_e, best_iter_e,
    'b', 'CatBoost — Boosting Convergence', 'RMSE (%)')
test_panel(fig.add_subplot(gs[0, 2]),
    ye_te, yp_te_e,
    'Experimental η (%)', 'Predicted η (%)',
    'c', 'CatBoost — Test',
    r2_te_e, rmse_te_e, mae_te_e, r_te_e, len(ye_te), pct=True)

scatter_panel(fig.add_subplot(gs[1, 0]),
    yk_tr, yp_tr_k, yk_v, yp_vl_k,
    r'Experimental $K_La$ (h$^{-1}$)', r'Predicted $K_La$ (h$^{-1}$)',
    'd', 'CatBoost — Training & Validation',
    r2_tv_k, rmse_tv_k, mae_tv_k, r_tv_k, len(yk_tv), pct=False)
conv_panel(fig.add_subplot(gs[1, 1]),
    rmse_tr_k, rmse_vl_k, best_iter_k,
    'e', 'CatBoost — Boosting Convergence', r'RMSE (h$^{-1}$)')
test_panel(fig.add_subplot(gs[1, 2]),
    yk_te, yp_te_k,
    r'Experimental $K_La$ (h$^{-1}$)', r'Predicted $K_La$ (h$^{-1}$)',
    'f', 'CatBoost — Test',
    r2_te_k, rmse_te_k, mae_te_k, r_te_k, len(yk_te), pct=False)

for ax in [fig.axes[0], fig.axes[2]]:
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))

plt.savefig('CatBoost_convergence_diagnostic.png', dpi=300, bbox_inches='tight')
plt.savefig('CatBoost_convergence_diagnostic.pdf', bbox_inches='tight')
plt.show()


# ===== CELL 6 — plateau significance analysis (paired bootstrap) =====
import json

RNG_BOOT = np.random.default_rng(SEED)
N_BOOT   = 1000

def rmse_paired(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat))**2)))

def plateau_onset(rmse_vl, best_iter, tol=0.01):
    # first iteration whose validation RMSE is within `tol` (1%) of the minimum
    best = rmse_vl[best_iter - 1]
    thr  = best * (1.0 + tol)
    for i in range(1, best_iter + 1):
        if rmse_vl[i - 1] <= thr:
            return i
    return max(1, best_iter // 2)

def bootstrap_delta(cb_model, Xv, yv, best_iter, comp_iter,
                    n_boot=N_BOOT, rng=RNG_BOOT, label=''):
    n = len(yv)
    yp_comp = cb_model.predict(Xv, ntree_end=comp_iter)
    yp_best = cb_model.predict(Xv, ntree_end=best_iter)

    rmse_comp = rmse_paired(yv, yp_comp)
    rmse_best = rmse_paired(yv, yp_best)
    delta_pt  = rmse_comp - rmse_best

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb  = np.asarray(yv)[idx]
        deltas[b] = rmse_paired(yb, yp_comp[idx]) - rmse_paired(yb, yp_best[idx])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    sd_d   = deltas.std(ddof=1)
    cohens = deltas.mean() / sd_d if sd_d > 0 else 0.0
    rel_pct = 100.0 * delta_pt / rmse_comp if rmse_comp > 0 else np.nan
    excludes_zero = (lo > 0) or (hi < 0)

    print(f'\n{label}')
    print(f'  RMSE @ iter {comp_iter:4d}      : {rmse_comp:.4f}')
    print(f'  RMSE @ best iter {best_iter:4d}: {rmse_best:.4f}')
    print(f'  Delta (comp - best)     : {delta_pt:.4f}')
    print(f'  Relative improvement    : {rel_pct:.2f} %')
    print(f'  Bootstrap 95% CI        : [{lo:.4f}, {hi:.4f}]')
    print(f"  Cohen's d (paired)      : {cohens:+.3f}")
    print(f'  95% CI excludes zero    : {excludes_zero}')
    return dict(delta=delta_pt, rmse_comp=rmse_comp, rmse_best=rmse_best,
                rel_pct=rel_pct, ci_low=lo, ci_high=hi,
                cohens_d=cohens, excludes_zero=excludes_zero,
                comp_iter=comp_iter, best_iter=best_iter)

# plateau onset defined from the data, not hard-coded
comp_iter_e = plateau_onset(rmse_vl_e, best_iter_e)
comp_iter_k = plateau_onset(rmse_vl_k, best_iter_k)

res_boot_e = bootstrap_delta(cb_conv_e, Xe_v, ye_v, best_iter_e, comp_iter_e,
                             label='eta (Removal Efficiency, %)')
res_boot_k = bootstrap_delta(cb_conv_k, Xk_v, yk_v, best_iter_k, comp_iter_k,
                             label='K_La (Mass-transfer coefficient, h^-1)')

print('\nSummary')
hdr = f'{"Target":11s} | {"comp":>5s} | {"best":>5s} | {"Delta":>10s} | {"rel %":>6s} | {"95% CI":>22s} | zero-excl'
print(hdr); print('-'*len(hdr))
for label, r in [('eta (%)', res_boot_e), ('K_La (h^-1)', res_boot_k)]:
    print(f'{label:11s} | {r["comp_iter"]:5d} | {r["best_iter"]:5d} | '
          f'{r["delta"]:+10.4f} | {r["rel_pct"]:6.2f} | '
          f'[{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}] | '
          f'{"Yes" if r["excludes_zero"] else "No"}')

with open('plateau_bootstrap_summary.json', 'w') as f:
    def cast(v):
        if isinstance(v, (bool, np.bool_)): return bool(v)
        if isinstance(v, (int, float, np.floating)): return float(v)
        return v
    json.dump({'eta'         : {k: cast(v) for k, v in res_boot_e.items()},
               'K_La'        : {k: cast(v) for k, v in res_boot_k.items()},
               'n_bootstrap' : N_BOOT,
               'random_seed' : SEED,
               'trace_iters' : TRACE_ITERS}, f, indent=2)
