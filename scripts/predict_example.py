"""
Minimal example: load the released CatBoost models and predict.

Usage (from the repository root):
    python scripts/predict_example.py --T 65 --Pv 25.1 --pH 9.0 --t 120

Inputs must stay inside the training ranges:
    eta model : T 45-100 degC, Pv 10-101.3 kPa, pH 7.5-9.83, t 0-720 min
    KLa model : T 50-100 degC, Pv 15-101.3 kPa, pH 8.36-9.83, t 0-300 min
"""
import argparse
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor

MODELS = Path(__file__).resolve().parents[1] / "models"


def load(name):
    m = CatBoostRegressor()
    m.load_model(str(MODELS / f"catboost_{name}.cbm"))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, required=True, help="temperature, degC")
    p.add_argument("--Pv", type=float, required=True, help="vacuum pressure, kPa")
    p.add_argument("--pH", type=float, required=True)
    p.add_argument("--t", type=float, required=True, help="stripping time, min")
    a = p.parse_args()
    x = np.array([[a.T, a.Pv, a.pH, a.t]])          # feature order: T, Pv, pH, t
    eta = float(load("eta").predict(x)[0])
    kla = float(load("KLa").predict(x)[0])
    print(f"Input: T={a.T} degC, Pv={a.Pv} kPa, pH={a.pH}, t={a.t} min")
    print(f"Predicted removal efficiency eta = {eta:.1f} %")
    print(f"Predicted KLa                    = {kla:.3f} h^-1")


if __name__ == "__main__":
    main()
