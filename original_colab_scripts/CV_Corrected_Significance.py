# ===== CELL 1 — install ======================================================
!pip -q install xgboost lightgbm catboost scikit-learn openpyxl scipy


# ===== CELL 2 — upload the two cleaned data files ============================
from google.colab import files as _gf
print('Upload BOTH files:')
print('  - ammonia efficiency subset cleaned.xlsx')
print('  - KL_a subset cleaned.xlsx\n')
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
print(f'\neta  file : {ETA_FILE}')
print(f'K_La file : {KLA_FILE}')


# ===== CELL 3 — data loading (identical cleaning to the other scripts) =======
import warnings, numpy as np, pandas as pd
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics         import r2_score, mean_squared_error
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

# ---- DEFAULT models: copied verbatim from Taylor_Defaults_Colab.py so that
#      the CV validation tests EXACTLY the models used for champion selection.
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

CHAMPION = 'CatBoost'          # revised result: champion for BOTH targets

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
print(f'eta  rows : {len(eta_df)}  (expected 218)')
print(f'K_La rows : {len(kla_df)}  (expected 159)')


# ===== CELL 4 — repeated stratified k-fold CV (R^2 per fold, all models) =====
N_SPLITS  = 10        # k
N_REPEATS = 10        # J  -> 100 R^2 estimates per model  (Bouckaert & Frank)
Q_BINS    = 4         # quartile stratification of the continuous target

def repeated_cv_scores(df, target):
    """Return DataFrame: rows = 100 folds, cols = model names, values = R^2."""
    X = df[FEAT].values
    y = df[target].values
    bins = pd.qcut(y, q=Q_BINS, labels=False, duplicates='drop')

    model_names = list(get_models_default().keys())
    scores = {name: [] for name in model_names}

    fold_id = 0
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=SEED + rep)
        for tr_idx, te_idx in skf.split(X, bins):
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            # fresh model instances every fold (no state leakage)
            for name, mdl in get_models_default().items():
                mdl.fit(Xtr, ytr)
                scores[name].append(r2_score(yte, mdl.predict(Xte)))
            fold_id += 1

    n_test  = len(y) / N_SPLITS
    n_train = len(y) - n_test
    return pd.DataFrame(scores), n_train, n_test

print('\nRunning repeated CV for eta  (11 models x 100 folds) ...')
cv_eta, ntr_e, nte_e = repeated_cv_scores(eta_df, 'eta')
print('Running repeated CV for K_La (11 models x 100 folds) ...')
cv_kla, ntr_k, nte_k = repeated_cv_scores(kla_df, 'KLa')

cv_eta.to_csv('cv_scores_eta.csv', index_label='fold')
cv_kla.to_csv('cv_scores_KLa.csv', index_label='fold')

def summarise(cv, label):
    s = (cv.agg(['mean', 'std']).T
           .sort_values('mean', ascending=False)
           .rename(columns={'mean': 'CV_R2_mean', 'std': 'CV_R2_std'}))
    print(f'\n== Mean CV R^2 ({N_REPEATS}x{N_SPLITS}) — {label} ==')
    print(s.to_string(float_format='%.4f'))
    return s

sum_eta = summarise(cv_eta, 'eta')
sum_kla = summarise(cv_kla, 'K_La')


# ===== CELL 5 — significance vs the champion ================================
#   Primary  : Nadeau-Bengio corrected resampled paired t-test  (fix for c451)
#   Reported : uncorrected paired t-test + uncorrected Wilcoxon (transparency)
#   Effect   : paired Cohen's d_z  (statistical effect size only, c481)
#   Multiplicity: Holm-Bonferroni on the corrected one-sided p-values
# ---------------------------------------------------------------------------

def corrected_resampled_ttest(diffs, k, alternative='greater'):
    """
    Nadeau & Bengio (2003) corrected resampled paired t-test.
    diffs : per-fold score differences (champion - competitor)
    k     : number of folds (rho = n_test/n_train = 1/(k-1) for equal folds)
    """
    m       = len(diffs)
    mean_d  = float(np.mean(diffs))
    var_d   = float(np.var(diffs, ddof=1))
    rho     = 1.0 / (k - 1)
    var_cor = var_d * (1.0 / m + rho)
    if var_cor <= 0:
        return mean_d, np.nan, np.nan, np.nan
    t_stat  = mean_d / np.sqrt(var_cor)
    df      = m - 1
    if   alternative == 'greater': p = float(stats.t.sf(t_stat, df))
    elif alternative == 'less'   : p = float(stats.t.cdf(t_stat, df))
    else                         : p = float(2 * stats.t.sf(abs(t_stat), df))
    return mean_d, t_stat, p, df

def cohens_dz(diffs):
    sd = np.std(diffs, ddof=1)
    return float(np.mean(diffs) / sd) if sd > 0 else np.nan

def holm_bonferroni(pvals):
    """Return Holm-adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)          # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return adj

def compare_to_champion(cv, k, label):
    champ = cv[CHAMPION].values
    rows, raw_corr_p = [], []
    for name in cv.columns:
        if name == CHAMPION:
            continue
        diffs = champ - cv[name].values           # >0 => champion better
        mean_d, t_c, p_c, df = corrected_resampled_ttest(diffs, k, 'greater')
        # uncorrected references
        t_u, p_u = stats.ttest_rel(champ, cv[name].values, alternative='greater')
        try:
            w_stat, p_w = stats.wilcoxon(champ, cv[name].values,
                                         alternative='greater')
        except ValueError:                        # all-zero differences etc.
            w_stat, p_w = np.nan, np.nan
        rows.append(dict(
            comparison        = f'{CHAMPION} vs {name}',
            mean_dR2          = mean_d,
            cohens_dz         = cohens_dz(diffs),
            t_corrected       = t_c,
            p_corrected_1s    = p_c,
            p_uncorrected_ttest_1s = float(p_u),
            p_uncorrected_wilcoxon_1s = float(p_w),
        ))
        raw_corr_p.append(p_c)

    df_out = pd.DataFrame(rows)
    df_out['p_corrected_holm'] = holm_bonferroni(df_out['p_corrected_1s'].values)
    df_out['significant_holp_0.05'] = df_out['p_corrected_holm'] < 0.05
    df_out = df_out.sort_values('mean_dR2', ascending=False).reset_index(drop=True)

    print(f'\n================  {label}: {CHAMPION} vs each competitor  ================')
    show = ['comparison','mean_dR2','cohens_dz','p_corrected_1s',
            'p_corrected_holm','p_uncorrected_wilcoxon_1s','significant_holp_0.05']
    print(df_out[show].to_string(index=False, float_format='%.4g'))

    n_sig = int(df_out['significant_holp_0.05'].sum())
    n_tot = len(df_out)
    print(f'\n  -> {CHAMPION} is significantly ahead of {n_sig}/{n_tot} models '
          f'(Holm-corrected, alpha=0.05).')
    tied = df_out.loc[~df_out['significant_holp_0.05'], 'comparison'].tolist()
    if tied:
        print(f'  -> Statistically TIED with: '
              f'{", ".join(c.split("vs ")[1] for c in tied)}')
    return df_out

stats_eta = compare_to_champion(cv_eta, N_SPLITS, 'eta (Removal Efficiency)')
stats_kla = compare_to_champion(cv_kla, N_SPLITS, 'K_La (Mass-transfer coeff.)')

stats_eta.to_csv('stats_comparison_eta.csv', index=False)
stats_kla.to_csv('stats_comparison_KLa.csv', index=False)


# ===== CELL 6 — figure: CV distributions + corrected significance + effect ===
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10.5,
    'axes.linewidth': 0.9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150, 'pdf.fonttype': 42,
})
C_CHAMP = '#C0392B'; C_TREE = '#E8B4AE'; C_OTHER = '#B7C4CF'
TREE_FAMILY = {'CatBoost','XGBoost','LightGBM','GBDT',
               'RandomForest','ExtraTrees','AdaBoost'}

def bar_color(name):
    if name == CHAMPION: return C_CHAMP
    return C_TREE if name in TREE_FAMILY else C_OTHER

def panels(fig, gs_row, cv, summ, stats_df, target_title, r2_floor):
    order = summ.index.tolist()                       # sorted by mean CV R^2

    # (col 0) box plot of CV R^2
    ax0 = fig.add_subplot(gs_row[0])
    data = [np.clip(cv[m].values, r2_floor, 1.0) for m in order]
    bp = ax0.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                     showfliers=False, medianprops=dict(color='#222', lw=1.1))
    for patch, name in zip(bp['boxes'], order):
        patch.set_facecolor(bar_color(name)); patch.set_alpha(0.9)
        patch.set_edgecolor('#333'); patch.set_linewidth(0.6)
    ax0.set_yticks(range(1, len(order)+1)); ax0.set_yticklabels(order)
    ax0.invert_yaxis()
    ax0.set_xlabel(r'Cross-validated $R^2$')
    ax0.set_title(f'({target_title[0]}) {target_title[1]} — CV $R^2$ distribution',
                  loc='left', fontweight='bold')
    ax0.axvline(summ.loc[CHAMPION, 'CV_R2_mean'], color=C_CHAMP, ls=':', lw=0.9)
    ax0.grid(axis='x', alpha=0.25)

    # (col 1) corrected -log10 p vs champion
    ax1 = fig.add_subplot(gs_row[1])
    sd = stats_df.set_index('comparison')
    comp_names = [f'{CHAMPION} vs {m}' for m in order if m != CHAMPION]
    labels = [c.split('vs ')[1] for c in comp_names]
    pv = sd.loc[comp_names, 'p_corrected_holm'].clip(lower=1e-12).values
    neglog = -np.log10(pv)
    cols = [bar_color(l) for l in labels]
    ax1.barh(range(len(labels)), neglog, color=cols, edgecolor='#333', lw=0.5)
    ax1.axvline(-np.log10(0.05), color='#444', ls='--', lw=0.9)
    ax1.text(-np.log10(0.05), len(labels)-0.4, ' p=0.05', fontsize=7, color='#444')
    ax1.set_yticks(range(len(labels))); ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel(r'$-\log_{10}(p)$  (corrected, Holm)')
    ax1.set_title(f'({target_title[0]}2) Corrected significance vs {CHAMPION}',
                  loc='left', fontweight='bold')
    ax1.grid(axis='x', alpha=0.25)

    # (col 2) Cohen's d_z
    ax2 = fig.add_subplot(gs_row[2])
    dz = sd.loc[comp_names, 'cohens_dz'].values
    ax2.barh(range(len(labels)), dz, color=cols, edgecolor='#333', lw=0.5)
    for thr, lab in [(0.2,'S'), (0.5,'M'), (0.8,'L')]:
        ax2.axvline(thr, color='#999', ls=':', lw=0.7)
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"Cohen's $d_z$ (statistical effect size)")
    ax2.set_title(f'({target_title[0]}3) Effect size vs {CHAMPION}',
                  loc='left', fontweight='bold')
    ax2.grid(axis='x', alpha=0.25)

fig = plt.figure(figsize=(15, 9))
gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.42,
              left=0.10, right=0.97, top=0.93, bottom=0.08)
panels(fig, [gs[0, 0], gs[0, 1], gs[0, 2]],
       cv_eta, sum_eta, stats_eta, ('a', 'η'),      r2_floor=-0.5)
panels(fig, [gs[1, 0], gs[1, 1], gs[1, 2]],
       cv_kla, sum_kla, stats_kla, ('b', '$K_La$'), r2_floor=0.0)

fig.suptitle('Cross-validated validation of the champion with a '
             'dependency-corrected significance test '
             '(Nadeau–Bengio, 10×10 repeated CV)',
             fontsize=12, fontweight='bold', y=0.985)

plt.savefig('CV_validation_corrected.png', dpi=300, bbox_inches='tight')
plt.savefig('CV_validation_corrected.pdf', bbox_inches='tight')
plt.show()

print('\nSaved:')
print('  cv_scores_eta.csv / cv_scores_KLa.csv           (raw 100-fold R^2)')
print('  stats_comparison_eta.csv / stats_comparison_KLa.csv')
print('  CV_validation_corrected.png / .pdf')
