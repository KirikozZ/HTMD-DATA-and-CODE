import json
import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_COLS = ["Hydrophilicity", "Charge", "Sigma"]
TARGET_COLS = ["FWater", "FLi", "FCl", "FMg"]
SELECTIVITY_COL = "ln_selectivity"
OUTPUT_DIR = Path("outputs") / "gpr_selectivity"
RANDOM_SEED = 42


@dataclass
class HyperParams:
    lengthscales: list[float]
    signal: float
    noise: float


@dataclass
class FittedGPR:
    target: str
    hyperparams: HyperParams
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    x_train_scaled: np.ndarray
    alpha: np.ndarray
    l_factor: np.ndarray


def make_folds(n_rows: int, n_folds: int = 5, seed: int = RANDOM_SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    return [fold.astype(int) for fold in np.array_split(indices, n_folds)]


def standardize_x(x_train: np.ndarray, x_test: np.ndarray | None = None):
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    x_train_scaled = (x_train - x_mean) / x_std
    if x_test is None:
        return x_train_scaled, x_mean, x_std
    return x_train_scaled, (x_test - x_mean) / x_std, x_mean, x_std


def standardize_y(y_train: np.ndarray):
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std == 0:
        y_std = 1.0
    return (y_train - y_mean) / y_std, y_mean, y_std


def rbf_kernel(x_a: np.ndarray, x_b: np.ndarray, params: HyperParams) -> np.ndarray:
    lengthscales = np.asarray(params.lengthscales, dtype=float)
    xa = x_a / lengthscales
    xb = x_b / lengthscales
    sq_norm_a = np.sum(xa * xa, axis=1)[:, None]
    sq_norm_b = np.sum(xb * xb, axis=1)[None, :]
    sq_dist = np.maximum(sq_norm_a + sq_norm_b - 2.0 * xa @ xb.T, 0.0)
    return (params.signal**2) * np.exp(-0.5 * sq_dist)


def fit_gpr(x_train: np.ndarray, y_train_log: np.ndarray, target: str, params: HyperParams) -> FittedGPR:
    x_train_scaled, x_mean, x_std = standardize_x(x_train)
    y_scaled, y_mean, y_std = standardize_y(y_train_log)
    k_train = rbf_kernel(x_train_scaled, x_train_scaled, params)
    k_train += (params.noise**2 + 1e-8) * np.eye(len(x_train_scaled))
    l_factor = np.linalg.cholesky(k_train)
    alpha = np.linalg.solve(l_factor.T, np.linalg.solve(l_factor, y_scaled))
    return FittedGPR(
        target=target,
        hyperparams=params,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        x_train_scaled=x_train_scaled,
        alpha=alpha,
        l_factor=l_factor,
    )


def predict_log(model: FittedGPR, x: np.ndarray) -> np.ndarray:
    x_scaled = (x - model.x_mean) / model.x_std
    k_star = rbf_kernel(x_scaled, model.x_train_scaled, model.hyperparams)
    y_scaled_pred = k_star @ model.alpha
    return y_scaled_pred * model.y_std + model.y_mean


def prediction_std_log(model: FittedGPR, x: np.ndarray) -> np.ndarray:
    x_scaled = (x - model.x_mean) / model.x_std
    k_star = rbf_kernel(x_scaled, model.x_train_scaled, model.hyperparams)
    v = np.linalg.solve(model.l_factor, k_star.T)
    prior_var = model.hyperparams.signal**2
    var_scaled = np.maximum(prior_var - np.sum(v * v, axis=0), 0.0)
    return np.sqrt(var_scaled) * model.y_std


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum(err**2) / denom)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(r2),
    }


def sample_hyperparams(n_samples: int = 120, seed: int = RANDOM_SEED) -> list[HyperParams]:
    rng = np.random.default_rng(seed)
    params: list[HyperParams] = []

    # A few hand-picked smoothness/noise anchors help stabilize the random search.
    anchors = [
        ([0.35, 0.35, 0.35], 1.0, 0.05),
        ([0.7, 0.7, 0.7], 1.0, 0.05),
        ([1.2, 1.2, 1.2], 1.0, 0.05),
        ([2.0, 2.0, 2.0], 1.0, 0.08),
        ([0.7, 1.5, 0.7], 1.2, 0.05),
        ([1.5, 0.7, 1.5], 1.2, 0.05),
        ([1.0, 2.5, 1.0], 1.0, 0.10),
    ]
    for lengthscales, signal, noise in anchors:
        params.append(HyperParams(lengthscales=lengthscales, signal=signal, noise=noise))

    for _ in range(n_samples):
        lengthscales = np.exp(rng.uniform(np.log(0.25), np.log(5.0), size=3)).tolist()
        signal = float(np.exp(rng.uniform(np.log(0.5), np.log(3.0))))
        noise = float(np.exp(rng.uniform(np.log(0.015), np.log(0.45))))
        params.append(HyperParams(lengthscales=lengthscales, signal=signal, noise=noise))
    return params


def cross_val_predict_for_params(
    x: np.ndarray,
    y_log: np.ndarray,
    target: str,
    params: HyperParams,
    folds: list[np.ndarray],
) -> np.ndarray:
    pred = np.zeros_like(y_log, dtype=float)
    all_idx = np.arange(len(y_log))
    for valid_idx in folds:
        train_idx = np.setdiff1d(all_idx, valid_idx)
        model = fit_gpr(x[train_idx], y_log[train_idx], target, params)
        pred[valid_idx] = predict_log(model, x[valid_idx])
    return pred


def optimize_target(
    x: np.ndarray,
    y_raw: np.ndarray,
    target: str,
    folds: list[np.ndarray],
    candidates: list[HyperParams],
) -> tuple[HyperParams, pd.DataFrame, np.ndarray]:
    y_log = np.log1p(y_raw)
    rows = []
    best_score = math.inf
    best_params = candidates[0]
    best_pred_log = np.zeros_like(y_log)

    for i, params in enumerate(candidates):
        try:
            pred_log = cross_val_predict_for_params(x, y_log, target, params, folds)
        except np.linalg.LinAlgError:
            continue

        log_metrics = metric_block(y_log, pred_log)
        raw_pred = np.expm1(pred_log)
        raw_pred = np.maximum(raw_pred, 0.0)
        raw_metrics = metric_block(y_raw, raw_pred)
        row = {
            "target": target,
            "candidate": i,
            "score_log_rmse": log_metrics["rmse"],
            "log_mae": log_metrics["mae"],
            "log_r2": log_metrics["r2"],
            "raw_mae": raw_metrics["mae"],
            "raw_rmse": raw_metrics["rmse"],
            "raw_r2": raw_metrics["r2"],
            "lengthscale_Hydrophilicity": params.lengthscales[0],
            "lengthscale_Charge": params.lengthscales[1],
            "lengthscale_Sigma": params.lengthscales[2],
            "signal": params.signal,
            "noise": params.noise,
        }
        rows.append(row)
        if log_metrics["rmse"] < best_score:
            best_score = log_metrics["rmse"]
            best_params = params
            best_pred_log = pred_log.copy()

    return best_params, pd.DataFrame(rows).sort_values("score_log_rmse"), best_pred_log


def optimize_selectivity_pair(
    x: np.ndarray,
    fli_raw: np.ndarray,
    fmg_raw: np.ndarray,
    folds: list[np.ndarray],
    candidates: list[HyperParams],
) -> tuple[HyperParams, HyperParams, np.ndarray, np.ndarray, dict[str, float]]:
    true_selectivity = np.log((fli_raw + 1.0) / (fmg_raw + 1.0))
    fli_log = np.log1p(fli_raw)
    fmg_log = np.log1p(fmg_raw)

    fli_predictions = []
    fmg_predictions = []
    for params in candidates:
        try:
            fli_predictions.append(cross_val_predict_for_params(x, fli_log, "FLi", params, folds))
        except np.linalg.LinAlgError:
            fli_predictions.append(None)
        try:
            fmg_predictions.append(cross_val_predict_for_params(x, fmg_log, "FMg", params, folds))
        except np.linalg.LinAlgError:
            fmg_predictions.append(None)

    best = {
        "rmse": math.inf,
        "fli_idx": 0,
        "fmg_idx": 0,
        "fli_pred": None,
        "fmg_pred": None,
        "metrics": {},
    }
    for fli_idx, fli_pred in enumerate(fli_predictions):
        if fli_pred is None:
            continue
        for fmg_idx, fmg_pred in enumerate(fmg_predictions):
            if fmg_pred is None:
                continue
            predicted_selectivity = fli_pred - fmg_pred
            metrics = metric_block(true_selectivity, predicted_selectivity)
            if metrics["rmse"] < best["rmse"]:
                best = {
                    "rmse": metrics["rmse"],
                    "fli_idx": fli_idx,
                    "fmg_idx": fmg_idx,
                    "fli_pred": fli_pred,
                    "fmg_pred": fmg_pred,
                    "metrics": metrics,
                }

    return (
        candidates[int(best["fli_idx"])],
        candidates[int(best["fmg_idx"])],
        best["fli_pred"],
        best["fmg_pred"],
        best["metrics"],
    )


def write_report(metrics: dict, best_params: dict, top_rows: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# GPR selectivity training report",
        "",
        "Final selectivity definition:",
        "",
        "`ln_selectivity = ln((FLi + 1) / (FMg + 1)) = ln(FLi + 1) - ln(FMg + 1)`",
        "",
        "`log10_selectivity = ln_selectivity / ln(10)`",
        "",
        "Model:",
        "",
        "- Inputs: `Hydrophilicity`, `Charge`, `Sigma`",
        "- Stage 1: independent exact Gaussian Process Regression models for `FWater`, `FLi`, `FCl`, `FMg`",
        "- Target transform: `log1p(target)`",
        "- Kernel: RBF with ARD lengthscales plus Gaussian noise",
        "- Hyperparameter search: fixed anchors plus reproducible random search",
        "- `FWater` and `FCl` selected by 5-fold CV ln-RMSE; `FLi` and `FMg` selected jointly by final `ln_selectivity` CV RMSE",
        "",
        "## Cross-validation metrics",
        "",
        "| target | space | R2 | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]

    for target, block in metrics["targets"].items():
        raw = block["raw"]
        logm = block["log"]
        lines.append(f"| {target} | raw | {raw['r2']:.4f} | {raw['mae']:.4f} | {raw['rmse']:.4f} |")
        lines.append(f"| {target} | log1p | {logm['r2']:.4f} | {logm['mae']:.4f} | {logm['rmse']:.4f} |")

    sel = metrics["selectivity"]
    lines.extend(
        [
            f"| ln_selectivity | formula | {sel['r2']:.4f} | {sel['mae']:.4f} | {sel['rmse']:.4f} |",
            "",
            "## Best hyperparameters",
            "",
            "| target | lengthscale_Hydrophilicity | lengthscale_Charge | lengthscale_Sigma | signal | noise |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for target, params in best_params.items():
        lines.append(
            f"| {target} | {params['lengthscales'][0]:.4f} | {params['lengthscales'][1]:.4f} | "
            f"{params['lengthscales'][2]:.4f} | {params['signal']:.4f} | {params['noise']:.4f} |"
        )

    lines.extend(["", "## Top hyperparameter candidates", ""])
    header = list(top_rows.columns)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for _, row in top_rows.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in header) + " |")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def serialize_model(model: FittedGPR) -> dict:
    return {
        "target": model.target,
        "hyperparams": asdict(model.hyperparams),
        "x_mean": model.x_mean,
        "x_std": model.x_std,
        "y_mean": model.y_mean,
        "y_std": model.y_std,
        "x_train_scaled": model.x_train_scaled,
        "alpha": model.alpha,
        "l_factor": model.l_factor,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data.csv")
    df[SELECTIVITY_COL] = np.log((df["FLi"] + 1.0) / (df["FMg"] + 1.0))

    x = df[INPUT_COLS].to_numpy(dtype=float)
    folds = make_folds(len(df), n_folds=5)
    candidates = sample_hyperparams(n_samples=140)

    best_params: dict[str, HyperParams] = {}
    best_cv_pred_log: dict[str, np.ndarray] = {}
    all_search_rows = []
    metrics = {"targets": {}, "selectivity": {}}

    for target in TARGET_COLS:
        y_raw = df[target].to_numpy(dtype=float)
        params, search_table, pred_log = optimize_target(x, y_raw, target, folds, candidates)
        best_params[target] = params
        best_cv_pred_log[target] = pred_log
        all_search_rows.append(search_table)

    fli_params, fmg_params, fli_pred_log, fmg_pred_log, selectivity_cv_metrics = optimize_selectivity_pair(
        x,
        df["FLi"].to_numpy(dtype=float),
        df["FMg"].to_numpy(dtype=float),
        folds,
        candidates,
    )
    best_params["FLi"] = fli_params
    best_params["FMg"] = fmg_params
    best_cv_pred_log["FLi"] = fli_pred_log
    best_cv_pred_log["FMg"] = fmg_pred_log

    metrics = {"targets": {}, "selectivity": selectivity_cv_metrics}
    for target in TARGET_COLS:
        y_raw = df[target].to_numpy(dtype=float)
        pred_log = best_cv_pred_log[target]
        pred_raw = np.maximum(np.expm1(pred_log), 0.0)
        metrics["targets"][target] = {
            "raw": metric_block(y_raw, pred_raw),
            "log": metric_block(np.log1p(y_raw), pred_log),
        }

    selectivity_true = df[SELECTIVITY_COL].to_numpy(dtype=float)
    selectivity_pred = best_cv_pred_log["FLi"] - best_cv_pred_log["FMg"]
    metrics["selectivity"] = metric_block(selectivity_true, selectivity_pred)

    fitted_models = {}
    full_predictions = df.copy()
    for target in TARGET_COLS:
        y_log = np.log1p(df[target].to_numpy(dtype=float))
        model = fit_gpr(x, y_log, target, best_params[target])
        fitted_models[target] = model
        pred_log = predict_log(model, x)
        pred_std_log = prediction_std_log(model, x)
        full_predictions[f"{target}_pred"] = np.maximum(np.expm1(pred_log), 0.0)
        full_predictions[f"{target}_pred_log1p"] = pred_log
        full_predictions[f"{target}_pred_log1p_std"] = pred_std_log

    cv_predictions = df.copy()
    for target in TARGET_COLS:
        cv_predictions[f"{target}_cv_pred"] = np.maximum(np.expm1(best_cv_pred_log[target]), 0.0)
        cv_predictions[f"{target}_cv_pred_log1p"] = best_cv_pred_log[target]
    cv_predictions["ln_selectivity_cv_pred"] = selectivity_pred
    cv_predictions["log_selectivity_cv_pred"] = cv_predictions["ln_selectivity_cv_pred"]
    cv_predictions["log10_selectivity_cv_pred"] = cv_predictions["ln_selectivity_cv_pred"] / np.log(10.0)
    cv_predictions["selectivity_cv_pred"] = np.exp(selectivity_pred)

    full_predictions["ln_selectivity_pred"] = (
        full_predictions["FLi_pred_log1p"] - full_predictions["FMg_pred_log1p"]
    )
    full_predictions["log_selectivity_pred"] = full_predictions["ln_selectivity_pred"]
    full_predictions["log10_selectivity_pred"] = full_predictions["ln_selectivity_pred"] / np.log(10.0)
    full_predictions["selectivity_pred"] = np.exp(full_predictions["ln_selectivity_pred"])

    search_results = pd.concat(all_search_rows, ignore_index=True)
    search_results.to_csv(OUTPUT_DIR / "hyperparameter_search.csv", index=False)
    cv_predictions.to_csv(OUTPUT_DIR / "cv_predictions.csv", index=False)
    full_predictions.to_csv(OUTPUT_DIR / "fitted_predictions.csv", index=False)

    params_json = {target: asdict(params) for target, params in best_params.items()}
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "best_hyperparameters": params_json}, f, indent=2)

    model_bundle = {
        "input_cols": INPUT_COLS,
        "target_cols": TARGET_COLS,
        "selectivity_formula": "ln((FLi + 1) / (FMg + 1))",
        "models": {target: serialize_model(model) for target, model in fitted_models.items()},
        "metrics": metrics,
        "best_hyperparameters": params_json,
    }

    with (OUTPUT_DIR / "gpr_selectivity_model.pkl").open("wb") as f:
        pickle.dump(model_bundle, f)

    npz_payload = {"x_train": x}
    for target, model in fitted_models.items():
        serialized = serialize_model(model)
        npz_payload[f"{target}_x_mean"] = serialized["x_mean"]
        npz_payload[f"{target}_x_std"] = serialized["x_std"]
        npz_payload[f"{target}_y_mean"] = np.asarray([serialized["y_mean"]])
        npz_payload[f"{target}_y_std"] = np.asarray([serialized["y_std"]])
        npz_payload[f"{target}_x_train_scaled"] = serialized["x_train_scaled"]
        npz_payload[f"{target}_alpha"] = serialized["alpha"]
        npz_payload[f"{target}_l_factor"] = serialized["l_factor"]
        npz_payload[f"{target}_lengthscales"] = np.asarray(serialized["hyperparams"]["lengthscales"])
        npz_payload[f"{target}_signal"] = np.asarray([serialized["hyperparams"]["signal"]])
        npz_payload[f"{target}_noise"] = np.asarray([serialized["hyperparams"]["noise"]])
    np.savez_compressed(OUTPUT_DIR / "gpr_selectivity_model.npz", **npz_payload)

    with (OUTPUT_DIR / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_cols": INPUT_COLS,
                "target_cols": TARGET_COLS,
                "selectivity_formula": "ln((FLi + 1) / (FMg + 1))",
                "metrics": metrics,
                "best_hyperparameters": params_json,
            },
            f,
            indent=2,
        )

    top_rows = (
        search_results.sort_values(["target", "score_log_rmse"])
        .groupby("target", as_index=False)
        .head(3)[
            [
                "target",
                "score_log_rmse",
                "raw_r2",
                "raw_mae",
                "raw_rmse",
                "lengthscale_Hydrophilicity",
                "lengthscale_Charge",
                "lengthscale_Sigma",
                "signal",
                "noise",
            ]
        ]
        .round(5)
    )
    write_report(metrics, params_json, top_rows, OUTPUT_DIR / "training_report.md")

    print(json.dumps({"metrics": metrics, "best_hyperparameters": params_json}, indent=2))


if __name__ == "__main__":
    main()
