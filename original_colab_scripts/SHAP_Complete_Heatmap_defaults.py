# ===== CELL 1 =====
!pip -q install shap catboost scikit-learn openpyxl scipy


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


# ===== CELL 3 — data, split, train CatBoost champions, SHAP values =====
import warnings, numpy as np, pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error
from catboost                import CatBoostRegressor

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
FEAT       = ['T', 'Pv', 'pH', 't']
FEAT_LABEL = {'T': 'Temperature (°C)', 'Pv': r'$P_v$ (kPa)',
              'pH': 'pH', 't': 't (min)'}

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

# -------------------------------------------------------------------------
# DEFAULT CatBoost — one identical configuration for BOTH targets, no tuning.
# Library defaults: iterations = 1000, depth = 6, learning_rate = auto.
# This is the same model that is selected (Taylor, defaults), validated
# (repeated CV, defaults) and diagnosed for convergence, so the SHAP values
# below describe exactly the deployed model. Only the random seed is fixed.
# -------------------------------------------------------------------------
CB_ETA_PARAMS = dict()   # empty -> pure library defaults
CB_KLA_PARAMS = dict()   # empty -> pure library defaults

Xe_tv = np.vstack([Xe_tr, Xe_v]); ye_tv = np.concatenate([ye_tr, ye_v])
Xk_tv = np.vstack([Xk_tr, Xk_v]); yk_tv = np.concatenate([yk_tr, yk_v])

cb_e = CatBoostRegressor(**CB_ETA_PARAMS, random_seed=SEED, verbose=False)
cb_e.fit(Xe_tv, ye_tv)
cb_k = CatBoostRegressor(**CB_KLA_PARAMS, random_seed=SEED, verbose=False)
cb_k.fit(Xk_tv, yk_tv)

Xe_full = eta_df[FEAT].values
Xk_full = kla_df[FEAT].values
sv_e = shap.TreeExplainer(cb_e).shap_values(Xe_full)
sv_k = shap.TreeExplainer(cb_k).shap_values(Xk_full)

mabs_e = np.abs(sv_e).mean(0)
mabs_k = np.abs(sv_k).mean(0)

def print_importance(mabs, label, unit):
    order = np.argsort(mabs)[::-1]
    print(f'\nMean |SHAP| — {label}')
    for i in order:
        print(f'  {FEAT[i]:4s}: {mabs[i]:8.4f} {unit}  ({100*mabs[i]/mabs.sum():5.1f} %)')

print_importance(mabs_e, 'eta (%)', '%')
print_importance(mabs_k, 'K_La (h^-1)', 'h^-1')


# ===== CELL 4 — shared style + figure functions =====
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors   import Normalize, LinearSegmentedColormap
from matplotlib.cm       import ScalarMappable

mpl.rcParams.update({
    'font.family'    : 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size'      : 11,
    'axes.labelsize' : 12,
    'axes.titlesize' : 12,
    'axes.linewidth' : 1.0,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi'     : 130,
    'pdf.fonttype'   : 42,
})

SHAP_CMAP = LinearSegmentedColormap.from_list(
    'shap', ['#1E88E5', '#B0BEC5', '#D32F2F'])


# ---- Figure 1: beeswarm summary (one per target) ----
def beeswarm_figure(sv, X, mabs, target_label, savename):
    order = np.argsort(mabs)
    fig, ax = plt.subplots(figsize=(9, 5.0))
    rng = np.random.default_rng(SEED)
    for row, idx in enumerate(order):
        vals  = sv[:, idx]
        fvals = X[:, idx]
        fnorm = (fvals - fvals.min()) / (fvals.max() - fvals.min() + 1e-12)
        y = row + rng.uniform(-0.30, 0.30, size=len(vals))
        ax.scatter(vals, y, c=fnorm, cmap=SHAP_CMAP, s=18,
                   alpha=0.8, edgecolors='none', vmin=0, vmax=1, rasterized=True)
    ax.axvline(0, color='#888', lw=0.9, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FEAT_LABEL[FEAT[i]] for i in order])
    ax.set_xlabel('SHAP value (impact on model output)')
    ax.set_title(f'SHAP summary plot — {target_label}', fontweight='bold', pad=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(y=0.08)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=SHAP_CMAP),
                      ax=ax, pad=0.02, fraction=0.035)
    cb.set_ticks([0, 1]); cb.set_ticklabels(['Low', 'High'])
    cb.set_label('Feature value', rotation=270, labelpad=16)
    plt.tight_layout()
    plt.savefig(f'{savename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{savename}.pdf', bbox_inches='tight')
    plt.show()


# ---- Figure 2: feature-target heatmap (single figure, both targets) ----
def feature_target_heatmap(mabs_e, mabs_k, savename):
    # Normalise each target column to its own max so the two targets are
    # visually comparable (each column shows within-target relative importance).
    col_e = mabs_e / mabs_e.max()
    col_k = mabs_k / mabs_k.max()
    M = np.column_stack([col_e, col_k])          # rows=features, cols=targets

    # Also keep raw values + percentages for cell annotations
    pct_e = 100 * mabs_e / mabs_e.sum()
    pct_k = 100 * mabs_k / mabs_k.sum()

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(M, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([r'$\eta$', r'$K_La$'], fontsize=13)
    ax.set_yticks(range(len(FEAT)))
    ax.set_yticklabels([FEAT_LABEL[f] for f in FEAT])
    ax.set_xlabel('Prediction target', labelpad=8)
    ax.set_title('Mean |SHAP| feature importance', fontweight='bold', pad=12)

    # annotate each cell with the within-target percentage
    for r in range(len(FEAT)):
        for c, pct in enumerate([pct_e, pct_k]):
            val = M[r, c]
            txt_color = 'white' if val > 0.55 else '#222'
            ax.text(c, r, f'{pct[r]:.1f}%', ha='center', va='center',
                    color=txt_color, fontsize=11, fontweight='bold')

    # gridlines between cells
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FEAT), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Relative importance\n(normalised per target)',
                 rotation=270, labelpad=26)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['Low', 'Med', 'High'])

    plt.tight_layout()
    plt.savefig(f'{savename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{savename}.pdf', bbox_inches='tight')
    plt.show()


# ---- Figure 3: impact-of-each-feature grid (one per target) ----
def impact_grid_figure(sv, X, target_label, yunit, savename):
    fig = plt.figure(figsize=(11, 8.5))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28,
                   left=0.09, right=0.96, top=0.92, bottom=0.09)
    letters = ['(a)', '(b)', '(c)', '(d)']
    for k, feat in enumerate(FEAT):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.scatter(X[:, k], sv[:, k], s=30, color='#1E70B8', alpha=0.8,
                   edgecolors='white', linewidths=0.4)
        ax.axhline(0, color='#888', lw=0.8, ls=':')
        ax.set_xlabel(FEAT_LABEL[feat], labelpad=4)
        ax.set_ylabel(f'SHAP value for {feat}\n({yunit})', labelpad=6)
        ax.set_title(letters[k], loc='left', fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=3)
        ax.margins(x=0.04)
    fig.suptitle(f'Impact of each feature on {target_label} prediction',
                 fontsize=13, fontweight='bold', y=0.975)
    plt.savefig(f'{savename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{savename}.pdf', bbox_inches='tight')
    plt.show()


# ===== CELL 5 — generate all figures =====
# Beeswarm (per target)
beeswarm_figure(sv_e, Xe_full, mabs_e, 'η (removal efficiency)',
                'SHAP1_beeswarm_eta')
beeswarm_figure(sv_k, Xk_full, mabs_k, r'$K_La$ (mass-transfer coefficient)',
                'SHAP1_beeswarm_KLa')

# Feature-target heatmap (single figure, both targets)
feature_target_heatmap(mabs_e, mabs_k, 'SHAP2_feature_target_heatmap')

# Impact-of-each-feature grid (per target)
impact_grid_figure(sv_e, Xe_full, 'η', '%', 'SHAP3_impact_eta')
impact_grid_figure(sv_k, Xk_full, r'$K_La$', r'h$^{-1}$', 'SHAP3_impact_KLa')

# Save raw SHAP values
pd.DataFrame(sv_e, columns=FEAT).to_csv('shap_values_eta.csv', index=False)
pd.DataFrame(sv_k, columns=FEAT).to_csv('shap_values_KLa.csv', index=False)
print('\nAll SHAP figures saved: 2 beeswarm + 1 heatmap + 2 impact grids.')
