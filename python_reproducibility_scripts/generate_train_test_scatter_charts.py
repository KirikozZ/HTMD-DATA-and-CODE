import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from train_gpr_selectivity import HyperParams, fit_gpr, predict_log


DATA_PATH = Path("data.csv")
METADATA_PATH = Path("outputs") / "gpr_selectivity" / "model_metadata.json"
OUTPUT_DIR = Path("outputs") / "gpr_selectivity" / "\u56fe\u8868\u7ed3\u679c"
RANDOM_SEED = 42
TEST_SIZE = 0.2

INPUT_COLS = ["Hydrophilicity", "Charge", "Sigma"]
TARGETS = [
    ("FWater", "\u6c34\u901a\u91cfFWater"),
    ("FLi", "\u9502\u79bb\u5b50\u901a\u91cfFLi"),
    ("FCl", "\u6c2f\u79bb\u5b50\u901a\u91cfFCl"),
    ("FMg", "\u9541\u79bb\u5b50\u901a\u91cfFMg"),
    ("log_selectivity", "\u5bf9\u6570\u9009\u62e9\u6027log_selectivity"),
]

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/arial.ttf",
]


def get_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = get_font(28)
FONT_SUBTITLE = get_font(18)
FONT_TEXT = get_font(15)
FONT_SMALL = get_font(12)


def split_indices(n_rows):
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    n_test = int(round(n_rows * TEST_SIZE))
    test_idx = np.sort(indices[:n_test])
    train_idx = np.sort(indices[n_test:])
    return train_idx, test_idx


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float("nan") if denom == 0 else 1.0 - float(np.sum(err**2) / denom)
    return {
        "R2": r2,
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
    }


def make_models(df, train_idx, metadata):
    x = df[INPUT_COLS].to_numpy(dtype=float)
    models = {}
    for target in ["FWater", "FLi", "FCl", "FMg"]:
        hp = metadata["best_hyperparameters"][target]
        params = HyperParams(
            lengthscales=list(hp["lengthscales"]),
            signal=float(hp["signal"]),
            noise=float(hp["noise"]),
        )
        y_log = np.log1p(df[target].to_numpy(dtype=float))
        models[target] = fit_gpr(x[train_idx], y_log[train_idx], target, params)
    return models


def predict_target(models, x, target):
    if target == "log_selectivity":
        fli_log = predict_log(models["FLi"], x)
        fmg_log = predict_log(models["FMg"], x)
        return fli_log - fmg_log
    pred_log = predict_log(models[target], x)
    return np.maximum(np.expm1(pred_log), 0.0)


def actual_target(df, target):
    if target == "log_selectivity":
        return np.log((df["FLi"].to_numpy(dtype=float) + 1.0) / (df["FMg"].to_numpy(dtype=float) + 1.0))
    return df[target].to_numpy(dtype=float)


def nice_range(values, pad=0.06):
    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi == lo:
        extra = abs(hi) * 0.1 + 1.0
        return lo - extra, hi + extra
    extra = (hi - lo) * pad
    return lo - extra, hi + extra


def map_points(xs, ys, rect, xlim, ylim):
    x0, y0, x1, y1 = rect
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    px = x0 + (xs - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
    py = y1 - (ys - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
    return px, py


def draw_axes(draw, rect, x_label, y_label, xlim, ylim):
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=(45, 52, 64), width=2)
    for i in range(6):
        tx = x0 + i * (x1 - x0) / 5
        ty = y1 - i * (y1 - y0) / 5
        draw.line([(tx, y1), (tx, y1 + 5)], fill=(45, 52, 64), width=1)
        draw.line([(x0 - 5, ty), (x0, ty)], fill=(45, 52, 64), width=1)
        xv = xlim[0] + i * (xlim[1] - xlim[0]) / 5
        yv = ylim[0] + i * (ylim[1] - ylim[0]) / 5
        draw.text((tx - 24, y1 + 9), f"{xv:.3g}", font=FONT_SMALL, fill=(45, 52, 64))
        draw.text((x0 - 68, ty - 8), f"{yv:.3g}", font=FONT_SMALL, fill=(45, 52, 64))
    draw.text(((x0 + x1) / 2 - 30, y1 + 38), x_label, font=FONT_TEXT, fill=(30, 38, 52))
    draw.text((x0 - 74, y0 - 30), y_label, font=FONT_TEXT, fill=(30, 38, 52))


def draw_scatter_panel(draw, rect, actual, pred, title, metric_values, color):
    lim = nice_range(np.concatenate([actual, pred]))
    draw_axes(draw, rect, "\u5b9e\u9645\u503c", "\u9884\u6d4b\u503c", lim, lim)
    x0, y0, x1, y1 = rect
    p0, q0 = map_points([lim[0]], [lim[0]], rect, lim, lim)
    p1, q1 = map_points([lim[1]], [lim[1]], rect, lim, lim)
    draw.line([(p0[0], q0[0]), (p1[0], q1[0])], fill=(239, 68, 68), width=3)
    px, py = map_points(actual, pred, rect, lim, lim)
    for a, b in zip(px, py):
        draw.ellipse((a - 4, b - 4, a + 4, b + 4), fill=color, outline=(15, 23, 42))
    draw.text((x0, y0 - 52), title, font=FONT_SUBTITLE, fill=(17, 24, 39))
    info = f"R\u00b2={metric_values['R2']:.4f}   MAE={metric_values['MAE']:.4g}   RMSE={metric_values['RMSE']:.4g}"
    draw.text((x0, y0 - 26), info, font=FONT_TEXT, fill=(71, 85, 105))


def save_chart_and_csv(df, models, train_idx, test_idx, target, label):
    x = df[INPUT_COLS].to_numpy(dtype=float)
    actual = actual_target(df, target)
    pred = predict_target(models, x, target)

    split = np.full(len(df), "\u8bad\u7ec3\u96c6", dtype=object)
    split[test_idx] = "\u6d4b\u8bd5\u96c6"
    out = df[INPUT_COLS].copy()
    out["\u6570\u636e\u96c6"] = split
    out["\u5b9e\u9645\u503c"] = actual
    out["\u9884\u6d4b\u503c"] = pred
    out["\u6b8b\u5dee_\u5b9e\u9645\u51cf\u9884\u6d4b"] = actual - pred
    out["\u7edd\u5bf9\u8bef\u5dee"] = np.abs(actual - pred)

    base_name = f"\u8bad\u7ec3\u96c6\u6d4b\u8bd5\u96c6\u6563\u70b9\u56fe_{label}"
    out.to_csv(OUTPUT_DIR / f"{base_name}.csv", index=False, encoding="utf-8-sig")

    train_m = metrics(actual[train_idx], pred[train_idx])
    test_m = metrics(actual[test_idx], pred[test_idx])

    img = Image.new("RGB", (1500, 760), "white")
    draw = ImageDraw.Draw(img)
    draw.text((42, 28), f"\u8bad\u7ec3\u96c6/\u6d4b\u8bd5\u96c6\u9884\u6d4b\u6563\u70b9\u56fe - {label}", font=FONT_TITLE, fill=(15, 23, 42))
    draw.text(
        (42, 68),
        f"\u5212\u5206\u65b9\u5f0f\uff1a80/20 \u968f\u673a\u5212\u5206\uff0cseed={RANDOM_SEED}\uff1b\u8bad\u7ec3\u96c6 n={len(train_idx)}\uff0c\u6d4b\u8bd5\u96c6 n={len(test_idx)}",
        font=FONT_TEXT,
        fill=(71, 85, 105),
    )

    draw_scatter_panel(
        draw,
        (105, 165, 690, 620),
        actual[train_idx],
        pred[train_idx],
        "\u8bad\u7ec3\u96c6",
        train_m,
        (37, 99, 235),
    )
    draw_scatter_panel(
        draw,
        (850, 165, 1435, 620),
        actual[test_idx],
        pred[test_idx],
        "\u6d4b\u8bd5\u96c6",
        test_m,
        (16, 185, 129),
    )

    img.save(OUTPUT_DIR / f"{base_name}.png")

    return [
        {"\u76ee\u6807": label, "\u6570\u636e\u96c6": "\u8bad\u7ec3\u96c6", **train_m, "\u6837\u672c\u6570": len(train_idx)},
        {"\u76ee\u6807": label, "\u6570\u636e\u96c6": "\u6d4b\u8bd5\u96c6", **test_m, "\u6837\u672c\u6570": len(test_idx)},
    ]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_idx, test_idx = split_indices(len(df))
    models = make_models(df, train_idx, metadata)

    summary_rows = []
    for target, label in TARGETS:
        summary_rows.extend(save_chart_and_csv(df, models, train_idx, test_idx, target, label))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "\u8bad\u7ec3\u96c6\u6d4b\u8bd5\u96c6\u6563\u70b9\u56fe_\u6307\u6807\u6c47\u603b.csv", index=False, encoding="utf-8-sig")

    meta = {
        "split": "80/20 random split",
        "seed": RANDOM_SEED,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "targets": [label for _, label in TARGETS],
        "metrics": ["R2", "MAE", "RMSE"],
        "note": "Flux targets are evaluated in raw units; log_selectivity is evaluated as log((FLi+1)/(FMg+1)).",
    }
    (OUTPUT_DIR / "\u8bad\u7ec3\u96c6\u6d4b\u8bd5\u96c6\u6563\u70b9\u56fe_\u8bf4\u660e.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
