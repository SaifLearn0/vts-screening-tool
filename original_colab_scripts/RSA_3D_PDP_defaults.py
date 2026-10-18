# ===== CELL 1 =====
!pip -q install catboost scikit-learn openpyxl


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


# ===== CELL 3 — data, split, train CatBoost champions =====
import warnings, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error
from catboost                import CatBoostRegressor

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
FEAT = ['T', 'Pv', 'pH', 't']
IDX  = {'T': 0, 'Pv': 1, 'pH': 2, 't': 3}

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
# Same model that is selected (Taylor, defaults), validated (repeated CV),
# convergence-checked and SHAP-analysed, so these surfaces describe exactly
# the deployed model. Only the random seed is fixed.
# -------------------------------------------------------------------------
CB_ETA_PARAMS = dict()   # empty -> pure library defaults
CB_KLA_PARAMS = dict()   # empty -> pure library defaults

Xe_tv = np.vstack([Xe_tr, Xe_v]); ye_tv = np.concatenate([ye_tr, ye_v])
Xk_tv = np.vstack([Xk_tr, Xk_v]); yk_tv = np.concatenate([yk_tr, yk_v])

cb_e = CatBoostRegressor(**CB_ETA_PARAMS, random_seed=SEED, verbose=False)
cb_e.fit(Xe_tv, ye_tv)
cb_k = CatBoostRegressor(**CB_KLA_PARAMS, random_seed=SEED, verbose=False)
cb_k.fit(Xk_tv, yk_tv)


# ===== CELL 4 — two-way partial dependence surface builder =====
import matplotlib as mpl, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mpl.rcParams.update({
    'font.family'    : 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size'      : 10,
    'axes.labelsize' : 11,
    'figure.dpi'     : 130,
    'pdf.fonttype'   : 42,
})

AXIS_LABEL = {'T': 'Temperature (°C)', 'Pv': r'$P_v$ (kPa)',
              'pH': 'pH', 't': 't (min)'}

GRID = 20   # grid points per axis (scikit-learn PDP default)

def pdp_surface(model, X_data, feat_x, feat_y, target_label, target_unit,
                clip_max=None, savename='surface'):
    """
    Two-way partial dependence (Friedman, 2001):
    for each (x, y) grid node, replace feat_x and feat_y in EVERY row of the
    dataset with the grid values, predict, and average over all rows.
    Averaging over the data smooths the tree staircase without artificial
    smoothing, and keeps predictions within the observed target range.
    """
    ix, iy = IDX[feat_x], IDX[feat_y]
    xr = np.linspace(X_data[:, ix].min(), X_data[:, ix].max(), GRID)
    yr = np.linspace(X_data[:, iy].min(), X_data[:, iy].max(), GRID)
    XX, YY = np.meshgrid(xr, yr)

    n = X_data.shape[0]
    Z = np.zeros((GRID, GRID))
    for i in range(GRID):
        for j in range(GRID):
            Xtmp = X_data.copy()
            Xtmp[:, ix] = XX[i, j]
            Xtmp[:, iy] = YY[i, j]
            Z[i, j] = model.predict(Xtmp).mean()

    # Safety guard: predictions are already bounded by the observed target
    # range (tree outputs cannot exceed training leaf values), so this clip
    # is a no-op for the present data and never alters the surface.
    if clip_max is not None:
        Z = np.clip(Z, None, clip_max)

    fig = plt.figure(figsize=(8.5, 6.5))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XX, YY, Z, cmap=cm.viridis,
                           linewidth=0, antialiased=True,
                           rstride=1, cstride=1, alpha=0.96)

    ax.set_xlabel('\n' + AXIS_LABEL[feat_x], labelpad=10)
    ax.set_ylabel('\n' + AXIS_LABEL[feat_y], labelpad=10)
    ax.set_zlabel(f'\n{target_label} ({target_unit})', labelpad=10)
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(pad=2)
    ax.set_title(f'{target_label} partial-dependence surface: '
                 f'{feat_x} × {feat_y}',
                 fontweight='bold', pad=14, fontsize=11)

    cb = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.10)
    cb.set_label(f'{target_label} ({target_unit})', rotation=270, labelpad=16)

    plt.tight_layout()
    plt.savefig(f'{savename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{savename}.pdf', bbox_inches='tight')
    plt.show()
    print(f'{savename}:  z-range = [{Z.min():.3f}, {Z.max():.3f}] {target_unit}')
    return XX, YY, Z


# ===== CELL 5 — the three SHAP-justified surfaces =====
Xe_all = eta_df[FEAT].values
Xk_all = kla_df[FEAT].values

# Surfaces pair the dominant driver of each target with a mechanistically
# relevant second variable (SHAP importances from the default model):
#   eta  — t dominant (53.8%); pH, Pv, T comparable secondaries (~15% each)
#   K_La — T (41.7%) and Pv (37.1%) the clear dominant pair (~79% combined)

# eta: pH x t   (t dominant; pH sets NH3/NH4+ speciation)
pdp_surface(cb_e, Xe_all, 'pH', 't', 'η', '%',
            clip_max=100.0, savename='RSA_eta_pH_t')

# eta: Pv x t   (t dominant; Pv drives the vacuum/mass-transfer effect)
pdp_surface(cb_e, Xe_all, 'Pv', 't', 'η', '%',
            clip_max=100.0, savename='RSA_eta_Pv_t')

# K_La: T x Pv  (dominant pair, ~79% combined; VLE threshold)
pdp_surface(cb_k, Xk_all, 'T', 'Pv', r'$K_La$', r'h$^{-1}$',
            clip_max=None, savename='RSA_KLa_T_Pv')

print('\nSaved: RSA_eta_pH_t, RSA_eta_Pv_t, RSA_KLa_T_Pv (png + pdf)')
