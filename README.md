VTS screening tool

A browser-based screening tool for ammonia removal in vacuum thermal
stripping (VTS). Enter an operating condition and the tool draws the
removal-efficiency trajectory and reports the mass-transfer coefficient.

Live tool: https://saiflearn0.github.io/vts-screening-tool/

What it does

Two gradient-boosted regression models (CatBoost, library default
hyperparameters) predict:

- removal efficiency, η (%)
- volumetric mass-transfer coefficient, K_La (h⁻¹)

from four operating variables: temperature (T), vacuum pressure (P_v),
pH, and stripping time (t).

The models were trained on a database compiled from six published
experimental VTS studies (n = 218 for η, n = 159 for K_La).

Sampled ranges

Predictions are bounded by the conditions present in the compiled data.
Inputs outside these ranges are flagged in the interface.

Variable | Range |
|---|---|
| Temperature, T | 45–100 °C |
| Vacuum pressure, P_v | 10–101.3 kPa |
| pH | 7.5–9.83 |
| Stripping time, t | 0–720 min |

Displayed trajectory

The trajectory is constrained to be physically admissible for a batch
run: removal is zero at t = 0 by definition, and cannot decrease with
time. These constraints are applied to the displayed curve only; the
trained models are unchanged.

Intended use

The tool is a screening aid for planning experiments. Estimates reflect
consistency within the compiled database. Feed matrix, liquid depth and
solids content differ between the source studies and are not model
inputs, so values should be confirmed against measurements on the
user's own system.

How it runs

The trained models are exported to JSON and evaluated in the browser.
No server, no installation, and no data leaves the page. The JavaScript
evaluator was verified against the Python CatBoost models across all
377 records (maximum absolute difference 2.3 × 10⁻⁶ for η and
3.1 × 10⁻⁸ for K_La).

Citation

Citation details will be added once the associated manuscript is published.

License

MIT
