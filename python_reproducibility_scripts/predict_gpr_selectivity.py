import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_COLS = ["Hydrophilicity", "Charge", "Sigma"]
TARGET_COLS = ["FWater", "FLi", "FCl", "FMg"]


def rbf_kernel(x_a: np.ndarray, x_b: np.ndarray, lengthscales: np.ndarray, signal: float) -> np.ndarray:
    xa = x_a / lengthscales
    xb = x_b / lengthscales
    sq_norm_a = np.sum(xa * xa, axis=1)[:, None]
    sq_norm_b = np.sum(xb * xb, axis=1)[None, :]
    sq_dist = np.maximum(sq_norm_a + sq_norm_b - 2.0 * xa @ xb.T, 0.0)
    return (signal**2) * np.exp(-0.5 * sq_dist)


def predict_target(bundle, target: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_mean = bundle[f"{target}_x_mean"]
    x_std = bundle[f"{target}_x_std"]
    x_train_scaled = bundle[f"{target}_x_train_scaled"]
    alpha = bundle[f"{target}_alpha"]
    l_factor = bundle[f"{target}_l_factor"]
    y_mean = float(bundle[f"{target}_y_mean"][0])
    y_std = float(bundle[f"{target}_y_std"][0])
    lengthscales = bundle[f"{target}_lengthscales"]
    signal = float(bundle[f"{target}_signal"][0])

    x_scaled = (x - x_mean) / x_std
    k_star = rbf_kernel(x_scaled, x_train_scaled, lengthscales, signal)
    pred_log = (k_star @ alpha) * y_std + y_mean

    v = np.linalg.solve(l_factor, k_star.T)
    var_scaled = np.maximum(signal**2 - np.sum(v * v, axis=0), 0.0)
    pred_log_std = np.sqrt(var_scaled) * y_std
    return pred_log, pred_log_std


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fluxes and log-selectivity with the trained GPR model.")
    parser.add_argument("input_csv", help="CSV with Hydrophilicity, Charge, Sigma columns.")
    parser.add_argument(
        "output_csv",
        nargs="?",
        default=str(Path("outputs") / "gpr_selectivity" / "new_predictions.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model",
        default=str(Path("outputs") / "gpr_selectivity" / "gpr_selectivity_model.npz"),
        help="Path to trained .npz model.",
    )
    parser.add_argument(
        "--metadata",
        default=str(Path("outputs") / "gpr_selectivity" / "model_metadata.json"),
        help="Path to model metadata JSON.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    missing = [col for col in INPUT_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    x = df[INPUT_COLS].to_numpy(dtype=float)
    bundle = np.load(args.model)
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    out = df.copy()
    for target in TARGET_COLS:
        pred_log, pred_log_std = predict_target(bundle, target, x)
        out[f"{target}_pred"] = np.maximum(np.expm1(pred_log), 0.0)
        out[f"{target}_pred_log1p"] = pred_log
        out[f"{target}_pred_log1p_std"] = pred_log_std

    out["ln_selectivity_pred"] = out["FLi_pred_log1p"] - out["FMg_pred_log1p"]
    out["log_selectivity_pred"] = out["ln_selectivity_pred"]
    out["log10_selectivity_pred"] = out["ln_selectivity_pred"] / np.log(10.0)
    out["selectivity_pred"] = np.exp(out["ln_selectivity_pred"])
    out["selectivity_formula"] = metadata["selectivity_formula"]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
