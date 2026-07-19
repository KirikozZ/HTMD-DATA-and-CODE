from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DATA_PATH = Path("data.csv")
OUTPUT_DIR = Path("outputs") / "gpr_selectivity" / "分布区间图表"

TARGETS = [
    ("FWater", "水通过个数 FWater"),
    ("FLi", "锂离子通过个数 FLi"),
    ("FCl", "氯离子通过个数 FCl"),
    ("FMg", "镁离子通过个数 FMg"),
]

INTERVALS = [
    ("极低区间", 0.00, 0.10, (78, 121, 167)),
    ("偏低区间", 0.10, 0.25, (129, 178, 154)),
    ("中低区间", 0.25, 0.50, (234, 205, 143)),
    ("中高区间", 0.50, 0.75, (224, 152, 103)),
    ("偏高区间", 0.75, 0.90, (183, 91, 91)),
    ("极高区间", 0.90, 1.00, (126, 57, 80)),
]

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/arial.ttf",
]


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = get_font(34)
FONT_PANEL = get_font(27)
FONT_LABEL = get_font(21)
FONT_TEXT = get_font(19)
FONT_SMALL = get_font(16)


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def fmt_num(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def compute_interval_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_rows = []
    quantile_rows = []
    for col, label in TARGETS:
        values = df[col].astype(float)
        n = len(values)
        qs = values.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()
        quantile_rows.append(
            {
                "参数": label,
                "最小值": qs[0],
                "P10": qs[0.1],
                "P25": qs[0.25],
                "P50": qs[0.5],
                "P75": qs[0.75],
                "P90": qs[0.9],
                "最大值": qs[1.0],
                "均值": values.mean(),
                "标准差": values.std(),
                "零值数量": int((values == 0).sum()),
                "零值占比": float((values == 0).mean()),
            }
        )

        for idx, (name, q_low, q_high, color) in enumerate(INTERVALS):
            lower = float(qs[q_low])
            upper = float(qs[q_high])
            if idx == 0:
                mask = (values >= lower) & (values <= upper)
            else:
                mask = (values > lower) & (values <= upper)
            count = int(mask.sum())
            interval_rows.append(
                {
                    "参数": label,
                    "区间名称": name,
                    "分位范围": f"P{int(q_low * 100)}-P{int(q_high * 100)}",
                    "下界": lower,
                    "上界": upper,
                    "样本数": count,
                    "占比": count / n,
                    "颜色RGB": str(color),
                }
            )

    return pd.DataFrame(interval_rows), pd.DataFrame(quantile_rows)


def draw_header(draw, title: str, subtitle: str | None = None):
    draw.text((70, 48), title, font=FONT_TITLE, fill=(20, 29, 43))
    if subtitle:
        draw.text((70, 94), subtitle, font=FONT_TEXT, fill=(82, 94, 111))


def draw_legend(draw, x: int, y: int):
    cursor = x
    for name, _, _, color in INTERVALS:
        draw.rectangle((cursor, y, cursor + 24, y + 24), fill=color, outline=(65, 65, 65))
        draw.text((cursor + 32, y - 1), name, font=FONT_SMALL, fill=(51, 65, 85))
        cursor += 150


def save_stacked_bar_chart(interval_df: pd.DataFrame):
    file_base = "论文图_水和离子通过个数_分布区间占比堆叠图"
    interval_df.to_csv(OUTPUT_DIR / f"{file_base}.csv", index=False, encoding="utf-8-sig")

    img = Image.new("RGB", (2200, 1400), "white")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "水和离子通过个数的分布区间占比",
        "按 min-P10-P25-P50-P75-P90-max 分位数划分，横向堆叠条表示各区间样本占比",
    )
    draw_legend(draw, 70, 150)

    left, right = 360, 2030
    top = 280
    bar_h = 92
    gap = 82

    for row_idx, (_, label) in enumerate(TARGETS):
        y = top + row_idx * (bar_h + gap)
        draw.text((70, y + 26), label, font=FONT_LABEL, fill=(30, 41, 59))
        draw.rectangle((left, y, right, y + bar_h), outline=(71, 85, 105), width=2)

        rows = interval_df[interval_df["参数"].eq(label)]
        x = left
        for _, row in rows.iterrows():
            width = (right - left) * float(row["占比"])
            color = next(c for name, _, _, c in INTERVALS if name == row["区间名称"])
            draw.rectangle((x, y, x + width, y + bar_h), fill=color)
            if width > 85:
                pct = f"{row['占比'] * 100:.1f}%"
                tw, th = text_size(draw, pct, FONT_SMALL)
                draw.text((x + width / 2 - tw / 2, y + bar_h / 2 - th / 2), pct, font=FONT_SMALL, fill=(15, 23, 42))
            x += width

        # Axis labels.
        for tick in np.linspace(0, 1, 6):
            tx = left + tick * (right - left)
            draw.line((tx, y + bar_h, tx, y + bar_h + 10), fill=(71, 85, 105), width=2)
            tick_text = f"{tick * 100:.0f}%"
            tw, _ = text_size(draw, tick_text, FONT_SMALL)
            draw.text((tx - tw / 2, y + bar_h + 18), tick_text, font=FONT_SMALL, fill=(71, 85, 105))

    # FMg zero annotation.
    fmg = interval_df[interval_df["参数"].str.contains("镁离子")]
    zero_pct = 0.0
    if not fmg.empty:
        first = fmg.iloc[0]
        if float(first["下界"]) == 0 and float(first["上界"]) == 0:
            zero_pct = float(first["占比"])
    note = f"注：FMg 的零值样本集中在极低区间，零值占比约 {zero_pct * 100:.2f}%。"
    draw.rounded_rectangle((70, 1120, 2030, 1265), radius=10, fill=(248, 250, 252), outline=(203, 213, 225), width=2)
    draw.text((100, 1162), note, font=FONT_TEXT, fill=(51, 65, 85))
    draw.text((100, 1206), "该图适合在论文中说明不同通过个数的相对分布位置，避免 FWater 与离子通量量纲差异造成视觉压缩。", font=FONT_TEXT, fill=(51, 65, 85))

    img.save(OUTPUT_DIR / f"{file_base}.png", dpi=(300, 300))


def map_x(value: float, lo: float, hi: float, left: int, right: int) -> float:
    if hi == lo:
        return (left + right) / 2
    return left + (value - lo) / (hi - lo) * (right - left)


def save_quantile_band_chart(quantile_df: pd.DataFrame):
    file_base = "论文图_水和离子通过个数_分位数带图"
    quantile_df.to_csv(OUTPUT_DIR / f"{file_base}.csv", index=False, encoding="utf-8-sig")

    img = Image.new("RGB", (2200, 1650), "white")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "水和离子通过个数的分布区间带",
        "每一行使用独立数值轴展示 min、P10、P25、P50、P75、P90、max；深色段为四分位区间",
    )

    left, right = 520, 1970
    top = 250
    row_h = 260
    axis_y_offset = 128

    for idx, row in quantile_df.iterrows():
        y0 = top + idx * row_h
        label = row["参数"]
        lo = float(row["最小值"])
        hi = float(row["最大值"])
        p10 = float(row["P10"])
        p25 = float(row["P25"])
        p50 = float(row["P50"])
        p75 = float(row["P75"])
        p90 = float(row["P90"])
        mean = float(row["均值"])
        zero_count = int(row["零值数量"])
        zero_ratio = float(row["零值占比"])

        draw.text((70, y0 + 82), label, font=FONT_PANEL, fill=(20, 29, 43))
        axis_y = y0 + axis_y_offset
        draw.line((left, axis_y, right, axis_y), fill=(71, 85, 105), width=3)

        # Percentile bands.
        x_min = map_x(lo, lo, hi, left, right)
        x_p10 = map_x(p10, lo, hi, left, right)
        x_p25 = map_x(p25, lo, hi, left, right)
        x_p50 = map_x(p50, lo, hi, left, right)
        x_p75 = map_x(p75, lo, hi, left, right)
        x_p90 = map_x(p90, lo, hi, left, right)
        x_max = map_x(hi, lo, hi, left, right)
        x_mean = map_x(mean, lo, hi, left, right)

        draw.rounded_rectangle((x_min, axis_y - 24, x_max, axis_y + 24), radius=12, fill=(226, 232, 240), outline=(148, 163, 184))
        draw.rounded_rectangle((x_p10, axis_y - 30, x_p90, axis_y + 30), radius=14, fill=(191, 219, 209), outline=(91, 141, 124), width=2)
        draw.rounded_rectangle((x_p25, axis_y - 42, x_p75, axis_y + 42), radius=16, fill=(224, 152, 103), outline=(159, 84, 62), width=2)

        # Markers.
        markers = [
            (x_min, "min", lo),
            (x_p10, "P10", p10),
            (x_p25, "P25", p25),
            (x_p50, "P50", p50),
            (x_p75, "P75", p75),
            (x_p90, "P90", p90),
            (x_max, "max", hi),
        ]
        used_levels: list[list[float]] = [[], [], []]
        for x, name, value in markers:
            level = 0
            for candidate_level, xs_used in enumerate(used_levels):
                if all(abs(x - prev_x) > 105 for prev_x in xs_used):
                    level = candidate_level
                    break
            used_levels[level].append(x)
            draw.line((x, axis_y - 58, x, axis_y + 58), fill=(30, 41, 59), width=2)
            marker = f"{name}\n{fmt_num(value)}"
            tw, _ = text_size(draw, marker.split("\n")[0], FONT_SMALL)
            label_y = axis_y + 68 + level * 37
            draw.text((x - tw / 2, label_y), marker, font=FONT_SMALL, fill=(51, 65, 85), align="center")
            if level:
                draw.line((x, axis_y + 58, x, label_y - 4), fill=(148, 163, 184), width=1)

        # Median and mean.
        draw.ellipse((x_p50 - 10, axis_y - 10, x_p50 + 10, axis_y + 10), fill=(15, 23, 42))
        draw.polygon([(x_mean, axis_y - 62), (x_mean - 12, axis_y - 40), (x_mean + 12, axis_y - 40)], fill=(220, 38, 38))
        draw.text((x_mean - 35, axis_y - 92), f"均值 {fmt_num(mean)}", font=FONT_SMALL, fill=(185, 28, 28))

        if zero_count:
            draw.text((70, y0 + 132), f"零值：{zero_count} 个（{zero_ratio * 100:.2f}%）", font=FONT_TEXT, fill=(185, 28, 28))

    legend_y = 1395
    draw.rounded_rectangle((70, legend_y, 2030, legend_y + 145), radius=10, fill=(248, 250, 252), outline=(203, 213, 225), width=2)
    draw.rounded_rectangle((105, legend_y + 36, 245, legend_y + 76), radius=10, fill=(226, 232, 240), outline=(148, 163, 184))
    draw.text((265, legend_y + 39), "min-max 全范围", font=FONT_TEXT, fill=(51, 65, 85))
    draw.rounded_rectangle((535, legend_y + 30, 675, legend_y + 82), radius=10, fill=(191, 219, 209), outline=(91, 141, 124), width=2)
    draw.text((695, legend_y + 39), "P10-P90 主体范围", font=FONT_TEXT, fill=(51, 65, 85))
    draw.rounded_rectangle((995, legend_y + 22, 1135, legend_y + 90), radius=10, fill=(224, 152, 103), outline=(159, 84, 62), width=2)
    draw.text((1155, legend_y + 39), "P25-P75 四分位范围", font=FONT_TEXT, fill=(51, 65, 85))
    draw.ellipse((1505, legend_y + 48, 1525, legend_y + 68), fill=(15, 23, 42))
    draw.text((1545, legend_y + 39), "中位数 P50", font=FONT_TEXT, fill=(51, 65, 85))

    img.save(OUTPUT_DIR / f"{file_base}.png", dpi=(300, 300))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    interval_df, quantile_df = compute_interval_tables(df)
    interval_df.to_csv(OUTPUT_DIR / "水和离子通过个数_分布区间绘图数据.csv", index=False, encoding="utf-8-sig")
    quantile_df.to_csv(OUTPUT_DIR / "水和离子通过个数_分位数绘图数据.csv", index=False, encoding="utf-8-sig")
    save_stacked_bar_chart(interval_df)
    save_quantile_band_chart(quantile_df)
    print(f"Saved publication charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
