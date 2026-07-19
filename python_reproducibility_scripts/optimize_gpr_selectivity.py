import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from predict_gpr_selectivity import INPUT_COLS, TARGET_COLS, predict_target


DEFAULT_OUTPUT_DIR = Path("outputs") / "gpr_selectivity"


def predict_selectivity(bundle, x: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(x, columns=INPUT_COLS)
    for target in TARGET_COLS:
        pred_log, pred_log_std = predict_target(bundle, target, x)
        out[f"{target}_pred"] = np.maximum(np.expm1(pred_log), 0.0)
        out[f"{target}_pred_log1p"] = pred_log
        out[f"{target}_pred_log1p_std"] = pred_log_std

    out["ln_selectivity_pred"] = out["FLi_pred_log1p"] - out["FMg_pred_log1p"]
    out["log_selectivity_pred"] = out["ln_selectivity_pred"]
    out["log10_selectivity_pred"] = out["ln_selectivity_pred"] / np.log(10.0)
    # Approximate independent-model uncertainty propagation in log space.
    out["ln_selectivity_pred_std"] = np.sqrt(
        out["FLi_pred_log1p_std"] ** 2 + out["FMg_pred_log1p_std"] ** 2
    )
    out["log_selectivity_pred_std"] = out["ln_selectivity_pred_std"]
    out["log10_selectivity_pred_std"] = out["ln_selectivity_pred_std"] / np.log(10.0)
    out["selectivity_pred"] = np.exp(out["ln_selectivity_pred"])
    return out


def acquisition(df: pd.DataFrame, beta: float) -> np.ndarray:
    return (df["ln_selectivity_pred"] + beta * df["ln_selectivity_pred_std"]).to_numpy()


def discrete_candidates(data_csv: str) -> np.ndarray:
    df = pd.read_csv(data_csv)
    levels = [sorted(df[col].unique()) for col in INPUT_COLS]
    return np.asarray(list(product(*levels)), dtype=float)


def latin_hypercube(n_samples: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_dim = bounds.shape[0]
    samples = np.empty((n_samples, n_dim), dtype=float)
    for dim in range(n_dim):
        perm = rng.permutation(n_samples)
        unit = (perm + rng.random(n_samples)) / n_samples
        low, high = bounds[dim]
        samples[:, dim] = low + unit * (high - low)
    return samples


def clip_to_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, bounds[:, 0]), bounds[:, 1])


def optimize_continuous(
    bundle,
    bounds: np.ndarray,
    beta: float,
    seed: int,
    n_initial: int,
    n_starts: int,
    n_iters: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    initial = latin_hypercube(n_initial, bounds, rng)
    initial_eval = predict_selectivity(bundle, initial)
    initial_eval["acquisition"] = acquisition(initial_eval, beta)
    starts = (
        initial_eval.sort_values("acquisition", ascending=False)
        .head(n_starts)[INPUT_COLS]
        .to_numpy(dtype=float)
    )

    span = bounds[:, 1] - bounds[:, 0]
    step = span / 6.0
    candidates = []

    for start in starts:
        current = np.array(start, dtype=float, copy=True)
        current_eval = predict_selectivity(bundle, current[None, :])
        current_score = float(acquisition(current_eval, beta)[0])

        for _ in range(n_iters):
            improved = False
            trial_points = [current]
            for dim in range(len(INPUT_COLS)):
                for direction in (-1.0, 1.0):
                    trial = current.copy()
                    trial[dim] += direction * step[dim]
                    trial_points.append(clip_to_bounds(trial, bounds))

            trial_eval = predict_selectivity(bundle, np.vstack(trial_points))
            trial_scores = acquisition(trial_eval, beta)
            best_idx = int(np.argmax(trial_scores))
            if trial_scores[best_idx] > current_score + 1e-12:
                current = np.array(trial_eval.loc[best_idx, INPUT_COLS].to_numpy(dtype=float), dtype=float, copy=True)
                current_score = float(trial_scores[best_idx])
                improved = True

            if not improved:
                step *= 0.5
            if np.max(step / span) < 1e-4:
                break

        final_eval = predict_selectivity(bundle, current[None, :])
        final_eval["acquisition"] = acquisition(final_eval, beta)
        candidates.append(final_eval)

    result = pd.concat(candidates, ignore_index=True)
    return result.sort_values("acquisition", ascending=False).drop_duplicates(INPUT_COLS)


def mixed_integer_initial(
    n_initial: int,
    bounds: np.ndarray,
    charge_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    hs_bounds = bounds[[0, 2], :]
    n_per_charge = max(10, int(np.ceil(n_initial / len(charge_values))))
    blocks = []
    for charge in charge_values:
        hs = latin_hypercube(n_per_charge, hs_bounds, rng)
        charge_col = np.full((n_per_charge, 1), float(charge))
        blocks.append(np.column_stack([hs[:, 0], charge_col[:, 0], hs[:, 1]]))
    return np.vstack(blocks)


def optimize_mixed_integer(
    bundle,
    bounds: np.ndarray,
    beta: float,
    seed: int,
    n_initial: int,
    n_starts: int,
    n_iters: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    charge_values = np.arange(int(np.ceil(bounds[1, 0])), int(np.floor(bounds[1, 1])) + 1)
    initial = mixed_integer_initial(n_initial, bounds, charge_values, rng)
    initial_eval = predict_selectivity(bundle, initial)
    initial_eval["acquisition"] = acquisition(initial_eval, beta)
    starts = (
        initial_eval.sort_values("acquisition", ascending=False)
        .head(n_starts)[INPUT_COLS]
        .to_numpy(dtype=float)
    )

    span = bounds[:, 1] - bounds[:, 0]
    step = np.array([span[0] / 6.0, 1.0, span[2] / 6.0], dtype=float)
    candidates = []

    for start in starts:
        current = np.array(start, dtype=float, copy=True)
        current[1] = np.rint(current[1])
        current = clip_to_bounds(current, bounds).copy()
        current_eval = predict_selectivity(bundle, current[None, :])
        current_score = float(acquisition(current_eval, beta)[0])

        for _ in range(n_iters):
            improved = False
            trial_points = [current]

            # Continuous moves for Hydrophilicity and Sigma.
            for dim in (0, 2):
                for direction in (-1.0, 1.0):
                    trial = current.copy()
                    trial[dim] += direction * step[dim]
                    trial = clip_to_bounds(trial, bounds).copy()
                    trial[1] = np.rint(trial[1])
                    trial_points.append(trial)

            # Integer moves for Charge.
            for direction in (-1.0, 1.0):
                trial = current.copy()
                trial[1] = np.rint(trial[1] + direction)
                trial = clip_to_bounds(trial, bounds).copy()
                trial[1] = np.rint(trial[1])
                trial_points.append(trial)

            trial_eval = predict_selectivity(bundle, np.vstack(trial_points))
            trial_scores = acquisition(trial_eval, beta)
            best_idx = int(np.argmax(trial_scores))
            if trial_scores[best_idx] > current_score + 1e-12:
                current = np.array(trial_eval.loc[best_idx, INPUT_COLS].to_numpy(dtype=float), dtype=float, copy=True)
                current[1] = np.rint(current[1])
                current_score = float(trial_scores[best_idx])
                improved = True

            if not improved:
                step[[0, 2]] *= 0.5
            if max(step[0] / span[0], step[2] / span[2]) < 1e-5:
                break

        final_eval = predict_selectivity(bundle, current[None, :])
        final_eval["Charge"] = final_eval["Charge"].round().astype(int)
        final_eval["acquisition"] = acquisition(final_eval, beta)
        candidates.append(final_eval)

    result = pd.concat(candidates, ignore_index=True)
    return result.sort_values("acquisition", ascending=False).drop_duplicates(INPUT_COLS)


def summarize_top(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cols = [
        *INPUT_COLS,
        "ln_selectivity_pred",
        "log10_selectivity_pred",
        "log_selectivity_pred",
        "log_selectivity_pred_std",
        "selectivity_pred",
        "FLi_pred",
        "FMg_pred",
        "FWater_pred",
        "FCl_pred",
        "acquisition",
    ]
    return df.sort_values("acquisition", ascending=False).head(top_n)[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize predicted GPR selectivity.")
    parser.add_argument("--data", default="data.csv", help="The manuscript 196-system data CSV; defines the reported Hydrophilicity, Charge, and Sigma bounds.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_OUTPUT_DIR / "gpr_selectivity_model.npz"),
        help="Trained .npz model path.",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_OUTPUT_DIR / "model_metadata.json"),
        help="Model metadata JSON path.",
    )
    parser.add_argument(
        "--mode",
        choices=["discrete", "continuous", "mixed_integer", "both"],
        default="mixed_integer",
        help="Optimization mode. The manuscript reports the mixed-integer result.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Acquisition beta: 0 exploits mean, positive explores upper confidence bound, negative is conservative.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-initial", type=int, default=12000)
    parser.add_argument("--n-starts", type=int, default=30)
    parser.add_argument("--n-iters", type=int, default=80)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for optimization result CSV/JSON files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = np.load(args.model)
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    training_df = pd.read_csv(args.data)
    bounds = training_df[INPUT_COLS].agg(["min", "max"]).to_numpy().T.astype(float)
    summary = {
        "selectivity_formula": metadata["selectivity_formula"],
        "beta": args.beta,
        "bounds": {col: {"min": float(bounds[i, 0]), "max": float(bounds[i, 1])} for i, col in enumerate(INPUT_COLS)},
        "results": {},
    }

    if args.mode in ("discrete", "both"):
        x_discrete = discrete_candidates(args.data)
        discrete_result = predict_selectivity(bundle, x_discrete)
        discrete_result["acquisition"] = acquisition(discrete_result, args.beta)
        discrete_top = summarize_top(discrete_result, args.top_n)
        discrete_top.to_csv(output_dir / "optimized_selectivity_discrete.csv", index=False)
        summary["results"]["discrete"] = discrete_top.iloc[0].to_dict()

    if args.mode in ("continuous", "both"):
        continuous_result = optimize_continuous(
            bundle,
            bounds,
            beta=args.beta,
            seed=args.seed,
            n_initial=args.n_initial,
            n_starts=args.n_starts,
            n_iters=args.n_iters,
        )
        continuous_top = summarize_top(continuous_result, args.top_n)
        continuous_top.to_csv(output_dir / "optimized_selectivity_continuous.csv", index=False)
        summary["results"]["continuous"] = continuous_top.iloc[0].to_dict()

    if args.mode in ("mixed_integer", "both"):
        mixed_integer_result = optimize_mixed_integer(
            bundle,
            bounds,
            beta=args.beta,
            seed=args.seed,
            n_initial=args.n_initial,
            n_starts=args.n_starts,
            n_iters=args.n_iters,
        )
        mixed_integer_top = summarize_top(mixed_integer_result, args.top_n)
        mixed_integer_top.to_csv(output_dir / "optimized_selectivity_mixed_integer.csv", index=False)
        summary["results"]["mixed_integer"] = mixed_integer_top.iloc[0].to_dict()

    with open(output_dir / "optimized_selectivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
