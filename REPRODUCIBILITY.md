# VTS-Screen: data, trained models and code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: CC BY 4.0](https://img.shields.io/badge/data%20%26%20models-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

This file documents the data, models and code in this repository. The web tool itself is `index.html`; see `README.md`.

This repository holds the data, trained models and code for:

> Rahman, S.U., Guo, S., Rony, M.R.H., Qiu, X., Li, Z. **From Published Experiments to a Web-Based Prediction Tool: Machine Learning of Ammonia Removal and Mass Transfer in Vacuum Thermal Stripping.**
> State Key Laboratory of Iron and Steel Industry of Environmental Protection, University of Science and Technology Beijing.

**Web tool (VTS-Screen):** https://saiflearn0.github.io/vts-screening-tool/

## Repository structure

```
data/                     Compiled VTS database (xlsx = originals, csv = same data)
  ammonia_efficiency_subset_cleaned.xlsx / .csv   removal efficiency subset, n = 218
  KL_a_subset_cleaned.xlsx / .csv                 KLa subset, n = 159
models/                   CatBoost champion models deployed in VTS-Screen
  catboost_eta.cbm / .json      removal efficiency η (%)
  catboost_KLa.cbm / .json      volumetric mass-transfer coefficient KLa (h⁻¹)
  model_card.json               training details (seed, features, library version)
notebooks/                Full analysis, one notebook per step (open in Jupyter or Colab)
  01_model_comparison_taylor.ipynb          11 models, defaults, Taylor diagrams (Fig. 1, Table S9)
  02_cv_corrected_significance.ipynb        10×10 CV + Nadeau–Bengio test (Fig. 2, Tables S10–S11)
  03_catboost_convergence_bootstrap.ipynb   convergence + bootstrap plateau test (Fig. 3)
  04_shap_interpretation.ipynb              SHAP analysis (Figs. 4–6)
  05_partial_dependence_surfaces.ipynb      partial-dependence surfaces (Figs. 7–8)
scripts/
  export_deployed_models.py   recreates models/ from data/
  predict_example.py          command-line prediction with the released models
original_colab_scripts/   The Colab scripts exactly as used for the manuscript
results/                  Tables and figures produced by the notebooks
```

## Data

| Column | Symbol | Unit | Range (η subset) | Range (KLa subset) |
|---|---|---|---|---|
| Temperature | T | °C | 45–100 | 50–100 |
| Vacuum pressure | Pv | kPa | 10–101.3 | 15–101.3 |
| pH | pH | – | 7.5–9.83 | 8.36–9.83 |
| Reaction (stripping) time | t | min | 0–720 | 0–300 |
| Removal efficiency | η | % | 0–100 | – |
| Volumetric mass-transfer coefficient | KLa | h⁻¹ | – | 0.12–1.95 |

`RowID` is a record identifier. `Paper` is the source-study ID: 1–6 in the η subset and 1–4 in the KLa subset.

**Sources.** The records were compiled from six peer-reviewed studies (Chen et al., 2025; Tao et al., 2024; Tao et al., 2019; Tao et al., 2018; Tian et al., 2019; Ukwuani & Tao, 2016). Values were taken from text, tables and supplementary material, and digitised from figures with WebPlotDigitizer. Where KLa was not reported directly, it was calculated as KL·a. **Please cite the original studies when you reuse the data.**

## Models

The released models use CatBoost with library-default hyperparameters (1000 iterations, depth 6, automatic learning rate) and random seed 42. They were refitted on the **complete** subsets for deployment. Features must be given in the order **[T, Pv, pH, t]**.

The accuracies reported in the paper come from cross-validation: R² = 0.883 ± 0.073 for η and 0.970 ± 0.026 for KLa. The refitted models were not re-scored.

```python
from catboost import CatBoostRegressor
m = CatBoostRegressor(); m.load_model("models/catboost_KLa.cbm")
m.predict([[65, 25.1, 9.0, 120]])        # T (°C), Pv (kPa), pH, t (min)
```

or `python scripts/predict_example.py --T 65 --Pv 25.1 --pH 9.0 --t 120`

Keep inputs inside the training ranges above. The models are not valid for extrapolation.

## Reproducing the analysis

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook          # run 01 → 05
```

You can also click **Open in Colab** at the top of each notebook. The data are downloaded from this repository automatically.

All five notebooks were re-run with the versions in `requirements.txt` and reproduce the manuscript results:

- CatBoost test R² of 0.9105 (η) and 0.9796 (KLa).
- Cross-validated R² of 0.883 and 0.970.
- Best boosting iterations of 711 and 914.
- SHAP shares: t = 53.8 % for η; T = 41.7 % and Pv = 37.1 % for KLa.

Newer scikit-learn releases can shift Random Forest and GBDT scores by up to ±0.003 R². For exact numbers, use the versions in Table S4 of the Supplementary Information.

## Licence

- Code (`notebooks/`, `scripts/`, `original_colab_scripts/`): MIT.
- Data, models and results: CC BY 4.0.
- See `LICENSE` (MIT) and `DATA_LICENSE.md` (CC BY 4.0).

## How to cite

Please cite the article and this archive (DOI above). Citation metadata is in `CITATION.cff`, and GitHub shows it under **"Cite this repository"**.

## Funding

- National Key R&D Program of China (2024YFD1600201)
- National Natural Science Foundation of China (52261145693)
- National Environmental and Energy Science and Technology International Cooperation Base
- Gates Foundation

## Contact

Corresponding author: Zifu Li, zifuli@ustb.edu.cn
