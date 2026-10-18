"""
Re-create the deployed CatBoost models (VTS-Screen web tool).

As described in the manuscript (Section 2, deployment), the champion
configuration -- CatBoost with library-default hyperparameters, random seed 42,
feature order [T, Pv, pH, t] -- is refitted on the COMPLETE eta (n = 218) and
KLa (n = 159) subsets. Reported accuracies remain the cross-validated values.

Usage (from the repository root):
    python scripts/export_deployed_models.py

Outputs (in models/):
    catboost_eta.cbm  / catboost_eta.json   -- removal efficiency, %
    catboost_KLa.cbm  / catboost_KLa.json   -- volumetric mass-transfer coeff., h^-1
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "models"
SEED = 42
FEAT = ["T", "Pv", "pH", "t"]


def _rename(df):
    # identical column cleaning to the analysis notebooks
    m = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if "temp" in cl:                              m[c] = "T"
        elif "vacu" in cl or "press" in cl:           m[c] = "Pv"
        elif cl == "ph":                              m[c] = "pH"
        elif "react" in cl or "time" in cl:           m[c] = "t"
        elif "effic" in cl or "η" in cl:              m[c] = "eta"
        elif ("kla" in cl or "k_la" in cl
              or "k la" in cl or "kl a" in cl):       m[c] = "KLa"
    return df.rename(columns=m)


def load(fname, target, header):
    df = _rename(pd.read_excel(DATA / fname, header=header))
    for col in FEAT + [target]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=FEAT + [target]).reset_index(drop=True)


def main():
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    jobs = [("ammonia_efficiency_subset_cleaned.xlsx", "eta", 0, 218, "%"),
            ("KL_a_subset_cleaned.xlsx",              "KLa", 1, 159, "h^-1")]
    meta = {"library": "catboost", "catboost_version": catboost.__version__,
            "hyperparameters": "library defaults (iterations=1000, depth=6, learning_rate=auto)",
            "random_seed": SEED, "feature_order": FEAT,
            "feature_units": {"T": "degC", "Pv": "kPa", "pH": "-", "t": "min"},
            "models": {}}
    for fname, tgt, hdr, n_exp, unit in jobs:
        df = load(fname, tgt, hdr)
        assert len(df) == n_exp, f"{tgt}: {len(df)} rows, expected {n_exp}"
        X, y = df[FEAT].values, df[tgt].values
        model = CatBoostRegressor(random_seed=SEED, verbose=False,
                                  allow_writing_files=False)  # no log folder; does not affect the fit
        model.fit(X, y)
        model.save_model(str(OUT / f"catboost_{tgt}.cbm"))
        model.save_model(str(OUT / f"catboost_{tgt}.json"), format="json")
        meta["models"][tgt] = {"files": [f"catboost_{tgt}.cbm", f"catboost_{tgt}.json"],
                               "target_unit": unit, "n_training_records": int(len(df)),
                               "training_data": f"data/{fname} (complete subset)",
                               "tree_count": int(model.tree_count_)}
        print(f"{tgt:4s}: fitted on n={len(df)}, trees={model.tree_count_} -> models/catboost_{tgt}.cbm/.json")
    (OUT / "model_card.json").write_text(json.dumps(meta, indent=2))
    print("Saved models/model_card.json")


if __name__ == "__main__":
    main()
