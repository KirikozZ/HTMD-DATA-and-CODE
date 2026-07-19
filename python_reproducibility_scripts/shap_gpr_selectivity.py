import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from predict_gpr_selectivity import INPUT_COLS, TARGET_COLS, rbf_kernel


DEFAULT_OUTPUT_DIR = Path("outputs") / "gpr_selectivity" / "shap_analysis"


def model_output(bundle, x: np.ndarray, output: str) -> np.ndarray:
    fli_log = predict_target_mean_log(bundle, "FLi", x)
    fmg_log = predict_target_mean_log(bundle, "FMg", x)
    if output == "log_selectivity":
        return fli_log - fmg_log
    if output in TARGET_COLS:
        return np.maximum(np.expm1(predict_target_mean_log(bundle, output, x)), 0.0)
    if output.endswith("_log1p"):
        target = output.replace("_log1p", "")
        if target in TARGET_COLS:
            return predict_target_mean_log(bundle, target, x)
    raise ValueError(f"Unsupported output: {output}")


def predict_target_mean_log(bundle, target: str, x: np.ndarray) -> np.ndarray:
    x_mean = bundle[f"{target}_x_mean"]
    x_std = bundle[f"{target}_x_std"]
    x_train_scaled = bundle[f"{target}_x_train_scaled"]
    alpha = bundle[f"{target}_alpha"]
    y_mean = float(bundle[f"{target}_y_mean"][0])
    y_std = float(bundle[f"{target}_y_std"][0])
    lengthscales = bundle[f"{target}_lengthscales"]
    signal = float(bundle[f"{target}_signal"][0])

    x_scaled = (x - x_mean) / x_std
    k_star = rbf_kernel(x_scaled, x_train_scaled, lengthscales, signal)
    return (k_star @ alpha) * y_std + y_mean


def subset_value(
    bundle,
    x_row: np.ndarray,
    background: np.ndarray,
    subset: tuple[int, ...],
    output: str,
) -> float:
    mixed = background.copy()
    if subset:
        mixed[:, list(subset)] = x_row[list(subset)]
    return float(model_output(bundle, mixed, output).mean())


def exact_shap_values(
    bundle,
    x_explain: np.ndarray,
    background: np.ndarray,
    output: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_features = len(INPUT_COLS)
    shap = np.zeros((len(x_explain), n_features), dtype=float)
    expected_value = float(model_output(bundle, background, output).mean())
    predictions = model_output(bundle, x_explain, output)

    all_features = tuple(range(n_features))
    for row_idx, x_row in enumerate(x_explain):
        value_cache: dict[tuple[int, ...], float] = {}
        for size in range(n_features + 1):
            for subset in combinations(all_features, size):
                value_cache[subset] = subset_value(bundle, x_row, background, subset, output)

        for feature in all_features:
            others = tuple(i for i in all_features if i != feature)
            phi = 0.0
            for size in range(n_features):
                for subset in combinations(others, size):
                    subset = tuple(sorted(subset))
                    subset_with_feature = tuple(sorted((*subset, feature)))
                    weight = (
                        math.factorial(size)
                        * math.factorial(n_features - size - 1)
                        / math.factorial(n_features)
                    )
                    phi += weight * (value_cache[subset_with_feature] - value_cache[subset])
            shap[row_idx, feature] = phi

    return shap, predictions, expected_value


def importance_table(shap_values: np.ndarray) -> pd.DataFrame:
    rows = []
    total = float(np.abs(shap_values).mean(axis=0).sum())
    for i, feature in enumerate(INPUT_COLS):
        mean_abs = float(np.abs(shap_values[:, i]).mean())
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": mean_abs,
                "relative_importance": mean_abs / total if total else 0.0,
                "mean_shap": float(shap_values[:, i].mean()),
                "median_abs_shap": float(np.median(np.abs(shap_values[:, i]))),
                "min_shap": float(shap_values[:, i].min()),
                "max_shap": float(shap_values[:, i].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def dependence_summary(df: pd.DataFrame, shap_values: np.ndarray, output: str) -> pd.DataFrame:
    rows = []
    for i, feature in enumerate(INPUT_COLS):
        values = df[feature].to_numpy(dtype=float)
        shap_col = shap_values[:, i]
        corr = float(np.corrcoef(values, shap_col)[0, 1]) if values.std() and shap_col.std() else float("nan")
        rows.append(
            {
                "output": output,
                "feature": feature,
                "value_shap_corr": corr,
                "mean_shap_low_quartile": float(shap_col[values <= np.quantile(values, 0.25)].mean()),
                "mean_shap_high_quartile": float(shap_col[values >= np.quantile(values, 0.75)].mean()),
            }
        )
    return pd.DataFrame(rows)


def ascii_bar(value: float, max_value: float, width: int = 32) -> str:
    if max_value <= 0:
        return ""
    n = int(round(width * value / max_value))
    return "#" * n


def write_report(
    output_dir: Path,
    all_importance: dict[str, pd.DataFrame],
    all_dependence: pd.DataFrame,
    expected_values: dict[str, float],
    n_background: int,
    n_explained: int,
    local_table: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# SHAP parameter importance analysis",
        "",
        "Method:",
        "",
        "- Exact interventional Shapley values over the 3 input parameters.",
        "- Background distribution: training rows from `data.csv`.",
        "- Main output: `log_selectivity_pred = log1p(FLi_pred) - log1p(FMg_pred)`.",
        "- Flux outputs: raw predicted `FWater`, `FLi`, `FCl`, and `FMg`.",
        "",
        f"Background rows: {n_background}",
        f"Explained rows: {n_explained}",
        "",
    ]

    for output, table in all_importance.items():
        max_val = float(table["mean_abs_shap"].max())
        lines.extend(
            [
                f"## {output}",
                "",
                f"Expected value: {expected_values[output]:.6f}",
                "",
                "| rank | feature | mean_abs_shap | relative_importance | mean_shap | range | bar |",
                "|---:|---|---:|---:|---:|---|---|",
            ]
        )
        for rank, (_, row) in enumerate(table.iterrows(), start=1):
            bar = ascii_bar(float(row["mean_abs_shap"]), max_val)
            shap_range = f"{row['min_shap']:.4f} to {row['max_shap']:.4f}"
            lines.append(
                f"| {rank} | {row['feature']} | {row['mean_abs_shap']:.6f} | "
                f"{row['relative_importance']:.2%} | {row['mean_shap']:.6f} | {shap_range} | `{bar}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Directionality",
            "",
            "`value_shap_corr` is the correlation between a parameter value and its SHAP contribution.",
            "Positive means larger values tend to increase the model output; negative means larger values tend to reduce it.",
            "",
            "| output | feature | value_shap_corr | low_quartile_mean_shap | high_quartile_mean_shap |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in all_dependence.iterrows():
        lines.append(
            f"| {row['output']} | {row['feature']} | {row['value_shap_corr']:.4f} | "
            f"{row['mean_shap_low_quartile']:.6f} | {row['mean_shap_high_quartile']:.6f} |"
        )

    if local_table is not None and not local_table.empty:
        lines.extend(
            [
                "",
                "## Local explanation for optimized candidate",
                "",
                "| feature | value | shap_log_selectivity |",
                "|---|---:|---:|",
            ]
        )
        for _, row in local_table.iterrows():
            lines.append(f"| {row['feature']} | {row['value']:.6f} | {row['shap_log_selectivity']:.6f} |")
        total = float(local_table["shap_log_selectivity"].sum())
        lines.append("")
        lines.append(f"Local SHAP contribution sum: {total:.6f}")

    (output_dir / "shap_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact SHAP analysis for the trained GPR selectivity model.")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument(
        "--model",
        default=str(Path("outputs") / "gpr_selectivity" / "gpr_selectivity_model.npz"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    x = df[INPUT_COLS].to_numpy(dtype=float)
    background = x.copy()
    bundle = np.load(args.model)

    all_importance = {}
    all_dependence = []
    expected_values = {}
    shap_wide = df.copy()
    local_table = None

    outputs = ["log_selectivity", *TARGET_COLS, "FWater_log1p", "FLi_log1p", "FCl_log1p", "FMg_log1p"]
    for output in outputs:
        shap_values, predictions, expected_value = exact_shap_values(bundle, x, background, output)
        expected_values[output] = expected_value
        importance = importance_table(shap_values)
        importance.insert(0, "output", output)
        all_importance[output] = importance

        shap_detail = df[INPUT_COLS].copy()
        shap_detail[f"{output}_pred"] = predictions
        shap_detail[f"{output}_base_value"] = expected_value
        for i, feature in enumerate(INPUT_COLS):
            shap_detail[f"shap_{feature}"] = shap_values[:, i]
            shap_wide[f"{output}_shap_{feature}"] = shap_values[:, i]
        shap_detail.to_csv(output_dir / f"shap_values_{output}.csv", index=False)

        dep = dependence_summary(df, shap_values, output)
        all_dependence.append(dep)

    importance_all = pd.concat(all_importance.values(), ignore_index=True)
    dependence_all = pd.concat(all_dependence, ignore_index=True)
    importance_all.to_csv(output_dir / "shap_importance.csv", index=False)
    dependence_all.to_csv(output_dir / "shap_dependence_summary.csv", index=False)
    shap_wide.to_csv(output_dir / "shap_values_all_outputs.csv", index=False)

    metadata = {
        "method": "exact interventional Shapley values",
        "features": INPUT_COLS,
        "outputs": list(all_importance.keys()),
        "expected_values": expected_values,
        "n_background": int(len(background)),
        "n_explained": int(len(x)),
    }
    optimized_path = Path("outputs") / "gpr_selectivity" / "optimized_selectivity_mixed_integer.csv"
    if optimized_path.exists():
        optimized = pd.read_csv(optimized_path).head(1)
        x_opt = optimized[INPUT_COLS].to_numpy(dtype=float)
        shap_opt, pred_opt, base_opt = exact_shap_values(bundle, x_opt, background, "log_selectivity")
        local_rows = []
        for i, feature in enumerate(INPUT_COLS):
            local_rows.append(
                {
                    "feature": feature,
                    "value": float(x_opt[0, i]),
                    "shap_log_selectivity": float(shap_opt[0, i]),
                }
            )
        local_table = pd.DataFrame(local_rows).sort_values("shap_log_selectivity", ascending=False)
        local_table["base_value"] = base_opt
        local_table["prediction"] = float(pred_opt[0])
        local_table.to_csv(output_dir / "shap_local_optimized_candidate.csv", index=False)
        metadata["optimized_candidate"] = {
            "source": str(optimized_path),
            "base_value": float(base_opt),
            "prediction": float(pred_opt[0]),
            "shap_sum": float(shap_opt.sum()),
        }
        local_all_rows = []
        for output in ["log_selectivity", *TARGET_COLS]:
            shap_output, pred_output, base_output = exact_shap_values(bundle, x_opt, background, output)
            for i, feature in enumerate(INPUT_COLS):
                local_all_rows.append(
                    {
                        "output": output,
                        "feature": feature,
                        "value": float(x_opt[0, i]),
                        "base_value": float(base_output),
                        "prediction": float(pred_output[0]),
                        "shap_value": float(shap_output[0, i]),
                    }
                )
        pd.DataFrame(local_all_rows).to_csv(output_dir / "shap_local_optimized_candidate_all_outputs.csv", index=False)
    (output_dir / "shap_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(output_dir, all_importance, dependence_all, expected_values, len(background), len(x), local_table)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
