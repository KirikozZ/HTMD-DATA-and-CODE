from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data.csv")
OUTPUT_DIR = Path("outputs") / "gpr_selectivity" / "分布区间"
TARGETS = {
    "FWater": "水通过个数FWater",
    "FLi": "锂离子通过个数FLi",
    "FCl": "氯离子通过个数FCl",
    "FMg": "镁离子通过个数FMg",
}

QUANTILES = [
    ("极低区间", 0.00, 0.10),
    ("偏低区间", 0.10, 0.25),
    ("中低区间", 0.25, 0.50),
    ("中高区间", 0.50, 0.75),
    ("偏高区间", 0.75, 0.90),
    ("极高区间", 0.90, 1.00),
]


def interval_mask(values: pd.Series, lower: float, upper: float, is_first: bool, is_last: bool):
    if is_first:
        left = values >= lower
    else:
        left = values > lower
    if is_last:
        right = values <= upper
    else:
        right = values <= upper
    return left & right


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    rows = []
    summary_rows = []

    for col, label in TARGETS.items():
        values = df[col].astype(float)
        n = len(values)
        qs = values.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()

        summary_rows.append(
            {
                "参数": label,
                "样本数": n,
                "最小值": values.min(),
                "P10": qs[0.1],
                "P25": qs[0.25],
                "中位数P50": qs[0.5],
                "P75": qs[0.75],
                "P90": qs[0.9],
                "最大值": values.max(),
                "均值": values.mean(),
                "标准差": values.std(),
                "零值数量": int((values == 0).sum()),
                "零值占比": float((values == 0).mean()),
            }
        )

        for idx, (name, q_low, q_high) in enumerate(QUANTILES):
            lower = float(qs[q_low])
            upper = float(qs[q_high])
            mask = interval_mask(values, lower, upper, idx == 0, idx == len(QUANTILES) - 1)
            count = int(mask.sum())
            rows.append(
                {
                    "参数": label,
                    "区间名称": name,
                    "分位范围": f"P{int(q_low * 100)}-P{int(q_high * 100)}",
                    "下界": lower,
                    "上界": upper,
                    "区间表示": f"{lower:g} - {upper:g}",
                    "样本数": count,
                    "占比": count / n,
                    "累计占比": np.nan,
                    "是否含零值": bool(((values == 0) & mask).any()),
                }
            )

        if (values == 0).any():
            rows.append(
                {
                    "参数": label,
                    "区间名称": "零值单独标注",
                    "分位范围": "value=0",
                    "下界": 0.0,
                    "上界": 0.0,
                    "区间表示": "0",
                    "样本数": int((values == 0).sum()),
                    "占比": float((values == 0).mean()),
                    "累计占比": np.nan,
                    "是否含零值": True,
                }
            )

    interval_df = pd.DataFrame(rows)
    for parameter in interval_df["参数"].unique():
        mask = interval_df["参数"].eq(parameter) & ~interval_df["区间名称"].eq("零值单独标注")
        interval_df.loc[mask, "累计占比"] = interval_df.loc[mask, "占比"].cumsum()

    summary_df = pd.DataFrame(summary_rows)
    interval_df.to_csv(OUTPUT_DIR / "水和离子通过个数_分布区间统计.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "水和离子通过个数_分布概况.csv", index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_DIR / '水和离子通过个数_分布区间统计.csv'}")
    print(f"Saved: {OUTPUT_DIR / '水和离子通过个数_分布概况.csv'}")


if __name__ == "__main__":
    main()
