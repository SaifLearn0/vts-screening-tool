# =============================================================
#  Taylor Diagram — Fair Comparison (ALL DEFAULT HYPERPARAMETERS)
#
#  Purpose: level-playing-field comparison of 11 models where
#  none has an Optuna-tuning advantage. Whichever model wins here
#  wins on merit.
#
#  Uses:
#    • Same data files as before
#    • Same stratified 80/10/10 split, same seed (SEED=42)
#    • Default hyperparameters for ALL 11 models
#    • Model trained on train+val, evaluated on test
# =============================================================


# =============================================================
#  CELL 1 — Install (run once per Colab session)
# =============================================================
!pip -q install xgboost lightgbm catboost scikit-learn openpyxl


# =============================================================
#  CELL 2 — Upload the two data files
# =============================================================
from google.colab import files as _gf
print('Upload BOTH files:')
print('  • ammonia efficiency subset cleaned.xlsx')
print('  • KL_a subset cleaned.xlsx\n')
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

print(f'\nη    file : {ETA_FILE}')
print(f'K_La file : {KLA_FILE}')


# =============================================================
#  CELL 3 — Train all 11 models with DEFAULT parameters
# =============================================================
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import (AdaBoostRegressor, GradientBoostingRegressor,
                                     RandomForestRegressor, ExtraTreesRegressor)
from sklearn.svm             import SVR
from sklearn.neural_network  import MLPRegressor
from sklearn.neighbors       import KNeighborsRegressor
from xgboost                 import XGBRegressor
from lightgbm                import LGBMRegressor
from catboost                import CatBoostRegressor

warnings.filterwarnings('ignore')
SEED = 42
FEAT = ['T', 'Pv', 'pH', 't']
np.random.seed(SEED)

# ── ALL 11 models on DEFAULTS ─────────────────────────────────
#     • Scaler applied to distance/kernel/neural models (this is
#       preprocessing, not tuning — required for those algorithms
#       to run sensibly at all).
#     • random_state / random_seed set for reproducibility only.
#     • verbose flags silenced.
#     • No n_estimators, depth, learning_rate, etc. overrides.
def get_models_default():
    return {
        'MLR'         : Pipeline([('sc', StandardScaler()),
                                  ('m',  LinearRegression())]),
        'AdaBoost'    : AdaBoostRegressor(random_state=SEED),
        'GBDT'        : GradientBoostingRegressor(random_state=SEED),
        'XGBoost'     : XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
        'LightGBM'    : LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1),
        'CatBoost'    : CatBoostRegressor(random_seed=SEED, verbose=False),
        'RandomForest': RandomForestRegressor(random_state=SEED, n_jobs=-1),
        'ExtraTrees'  : ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
        'SVR'         : Pipeline([('sc', StandardScaler()),
                                  ('m',  SVR())]),
        'MLP'         : Pipeline([('sc', StandardScaler()),
                                  ('m',  MLPRegressor(random_state=SEED,
                                                      max_iter=3000))]),
        'KNN'         : Pipeline([('sc', StandardScaler()),
                                  ('m',  KNeighborsRegressor())]),
    }

# ── Load and clean data (same rename as GBDT___CAT_code.txt) ──
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

print(f'η    rows : {len(eta_df)}  (expected 218)')
print(f'K_La rows : {len(kla_df)}  (expected 159)')

# ── Stratified 80/10/10 split ─────────────────────────────────
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
print(f'\nη    train/val/test = {len(ye_tr)}/{len(ye_v)}/{len(ye_te)}')
print(f'K_La train/val/test = {len(yk_tr)}/{len(yk_v)}/{len(yk_te)}')

# ── Train each model on train+val, score on test ──────────────
def run_all(Xtr, Xv, Xte, ytr, yv, yte, label):
    Xfit = np.vstack([Xtr, Xv])
    yfit = np.concatenate([ytr, yv])
    print(f'\n── Training 11 models on defaults ({label}) ──')
    results = {}
    for name, mdl in get_models_default().items():
        mdl.fit(Xfit, yfit)
        yp = mdl.predict(Xte)
        r2   = r2_score(yte, yp)
        rmse = float(np.sqrt(mean_squared_error(yte, yp)))
        mae  = mean_absolute_error(yte, yp)
        r    = float(np.corrcoef(yte, yp)[0, 1])
        std_ref  = float(np.std(yte, ddof=1))
        std_mdl  = float(np.std(yp,  ddof=1))
        results[name] = dict(R2=r2, RMSE=rmse, MAE=mae, r=r,
                             std_ref=std_ref, std_model=std_mdl)
        print(f'  {name:12s}: R²={r2:.4f}  RMSE={rmse:.4f}  r={r:.4f}')
    return results

res_eta = run_all(Xe_tr, Xe_v, Xe_te, ye_tr, ye_v, ye_te, 'η')
res_kla = run_all(Xk_tr, Xk_v, Xk_te, yk_tr, yk_v, yk_te, 'K_La')

# ── Ranking table ─────────────────────────────────────────────
def ranking(res, label):
    df = pd.DataFrame(res).T.sort_values('R2', ascending=False)
    print(f'\n══ Final ranking (defaults) — {label} ══')
    print(df[['R2','RMSE','MAE','r']].to_string(float_format='%.4f'))
    return df

rank_eta = ranking(res_eta, 'η')
rank_kla = ranking(res_kla, 'K_La')


# =============================================================
#  CELL 4 — Taylor Diagram (both targets)
# =============================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from matplotlib import gridspec
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder   as GF

mpl.rcParams.update({
    'font.family'     : 'sans-serif',
    'font.sans-serif' : ['Arial', 'DejaVu Sans'],
    'font.size'       : 10,
    'axes.labelsize'  : 10,
    'axes.titlesize'  : 11,
    'axes.linewidth'  : 0.9,
    'legend.fontsize' : 8.5,
    'figure.dpi'      : 130,
    'pdf.fonttype'    : 42,
})

MARKERS = {
    'MLR': 'o', 'AdaBoost': 's', 'GBDT': '*', 'XGBoost': 'D',
    'LightGBM': '^', 'CatBoost': 'P', 'RandomForest': 'v',
    'ExtraTrees': 'h', 'SVR': 'X', 'MLP': '<', 'KNN': '>',
}
COLORS = {
    'MLR': '#3B7BB0',       'AdaBoost': '#8E44AD',
    'GBDT': '#8B6A55',      'XGBoost': '#F39C12',
    'LightGBM': '#27AE60',  'CatBoost': '#C0392B',
    'RandomForest': '#16A085', 'ExtraTrees': '#B8B233',
    'SVR': '#E74C8C',       'MLP': '#7F8C8D',
    'KNN': '#5DADE2',
}

def taylor_axes(fig, subplot_spec, std_ref, title):
    """Build one Taylor diagram polar axis."""
    smax = max(1.5 * std_ref, 1.0)
    tr = PolarAxes.PolarTransform()

    # Correlation coefficient tick angles
    rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    tlocs = np.arccos(rlocs)
    gl1 = GF.FixedLocator(tlocs)
    tf1 = GF.DictFormatter(dict(zip(tlocs, [f'{v:.2f}' for v in rlocs])))

    ghelper = FA.GridHelperCurveLinear(
        tr, extremes=(0, np.pi/2, 0, smax),
        grid_locator1=gl1, tick_formatter1=tf1,
    )
    ax = FA.FloatingSubplot(fig, subplot_spec, grid_helper=ghelper)
    fig.add_subplot(ax)

    ax.axis['top'].set_axis_direction('bottom')
    ax.axis['top'].toggle(ticklabels=True, label=True)
    ax.axis['top'].major_ticklabels.set_axis_direction('top')
    ax.axis['top'].label.set_axis_direction('top')
    ax.axis['top'].label.set_text('Correlation Coefficient')

    ax.axis['left'].set_axis_direction('bottom')
    ax.axis['left'].label.set_text('Standard Deviation')

    ax.axis['right'].set_axis_direction('top')
    ax.axis['right'].toggle(ticklabels=True)
    ax.axis['right'].major_ticklabels.set_axis_direction('left')

    ax.axis['bottom'].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, pad=18, fontweight='bold')

    polar = ax.get_aux_axes(tr)

    # Reference point (observation)
    polar.plot([0], [std_ref], marker='*', ms=15, color='black',
               mec='black', zorder=10, label='Reference (obs.)')

    # RMSE arcs centred on reference
    rs, ts = np.meshgrid(np.linspace(0, smax, 200),
                         np.linspace(0, np.pi/2, 200))
    rms = np.sqrt(std_ref**2 + rs**2 - 2 * std_ref * rs * np.cos(ts))
    cs = polar.contour(ts, rs, rms, levels=5, colors='#888',
                       linestyles=':', linewidths=0.7)
    polar.clabel(cs, inline=True, fontsize=7, fmt='%.2f')

    # Correlation ray lines
    for r in [0.6, 0.8, 0.9, 0.95, 0.99]:
        polar.plot([np.arccos(r), np.arccos(r)], [0, smax],
                   color='#aaa', ls=':', lw=0.5)

    # STD circle at reference
    theta_arc = np.linspace(0, np.pi/2, 200)
    polar.plot(theta_arc, [std_ref]*200, color='#666', ls='--', lw=0.6)

    return ax, polar

def plot_models(polar, results, ranking_df):
    """Place each model as a point; ★ marks the winner."""
    winner = ranking_df.index[0]
    for name, r in results.items():
        theta = np.arccos(min(max(r['r'], -1), 1))
        radius = r['std_model']
        is_win = (name == winner)
        polar.plot([theta], [radius],
                   marker=MARKERS[name],
                   ms=(16 if is_win else 10),
                   mfc=COLORS[name], mec='black',
                   mew=(1.4 if is_win else 0.6),
                   zorder=(9 if is_win else 5),
                   label=(f'★ {name}' if is_win else name))

# ── Build figure ──
fig = plt.figure(figsize=(16, 8))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                        left=0.05, right=0.97, top=0.90, bottom=0.10)

# η panel
std_ref_e = float(np.std(ye_te, ddof=1))
ax_e, pol_e = taylor_axes(fig, gs[0, 0], std_ref_e,
                          '(a) Taylor Diagram — η (Removal Efficiency)')
plot_models(pol_e, res_eta, rank_eta)
pol_e.legend(loc='upper right', bbox_to_anchor=(1.32, 1.02),
             frameon=True, framealpha=0.92, edgecolor='#ccc',
             fontsize=8.5, ncol=1)

# K_La panel
std_ref_k = float(np.std(yk_te, ddof=1))
ax_k, pol_k = taylor_axes(fig, gs[0, 1], std_ref_k,
                          '(b) Taylor Diagram — $K_La$ (Mass Transfer Coefficient)')
plot_models(pol_k, res_kla, rank_kla)
pol_k.legend(loc='upper right', bbox_to_anchor=(1.32, 1.02),
             frameon=True, framealpha=0.92, edgecolor='#ccc',
             fontsize=8.5, ncol=1)

fig.suptitle(
    'Fair Model Comparison — All Default Hyperparameters (no tuning)',
    fontsize=13.5, fontweight='bold', y=0.985)

plt.savefig('Taylor_defaults.png', dpi=300, bbox_inches='tight')
plt.savefig('Taylor_defaults.pdf', bbox_inches='tight')
plt.show()

print('\n✓ Saved: Taylor_defaults.png / .pdf')


# =============================================================
#  CELL 5 — Export ranking tables
# =============================================================
rank_eta.to_csv('ranking_defaults_eta.csv',
                index_label='model', float_format='%.4f')
rank_kla.to_csv('ranking_defaults_kla.csv',
                index_label='model', float_format='%.4f')

print('\n✓ Saved:')
print('   ranking_defaults_eta.csv')
print('   ranking_defaults_kla.csv')

print('\n══════════════════════════════════════════════════════')
print('  WHAT TO CHECK NEXT')
print('══════════════════════════════════════════════════════')
print('For each target, look at:')
print('  1. Which model has R² closest to 1?')
print('  2. Which model sits closest to the reference ★ on the Taylor?')
print('  3. Is the winner clearly ahead, or is it a tie (< 0.02 gap in R²)?')
print('')
print('If GBDT wins for η under defaults  → keep GBDT, honest and clean.')
print('If a different model wins clearly  → switch the champion (Option A).')
print('If tie (< 0.02 gap)                → tie-language + smallest overfit gap.')
