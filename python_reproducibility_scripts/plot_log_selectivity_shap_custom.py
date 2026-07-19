import csv
import html
from pathlib import Path


INPUT = Path("outputs") / "gpr_selectivity" / "shap_analysis" / "shap_values_all_outputs.csv"
OUTPUT = Path("outputs") / "gpr_selectivity" / "shap_figures" / "fig7_log_selectivity_shap_value_custom.svg"

FEATURES = ["Hydrophilicity", "Charge", "Sigma"]
SHAP_COLS = {
    "Hydrophilicity": "log_selectivity_shap_Hydrophilicity",
    "Charge": "log_selectivity_shap_Charge",
    "Sigma": "log_selectivity_shap_Sigma",
}

LOW_COLOR = "#93B1F0"
HIGH_COLOR = "#EE8C8E"
TEXT = "#222222"
MUTED = "#666666"
GRID = "#E6E8EF"
AXIS = "#111111"


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(value):
    return html.escape(str(value), quote=True)


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, int(round(v)))):02X}" for v in rgb)


def lerp_color(low, high, t):
    t = max(0.0, min(1.0, t))
    a = hex_to_rgb(low)
    b = hex_to_rgb(high)
    return rgb_to_hex(tuple(a[i] * (1 - t) + b[i] * t for i in range(3)))


def text(x, y, value, size=13, anchor="start", weight=400, fill=TEXT, extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}" {extra}>'
        f"{esc(value)}</text>"
    )


def line(x1, y1, x2, y2, color=AXIS, width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
    )


def circle(cx, cy, r, fill, opacity=0.82):
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'opacity="{opacity}" stroke="#FFFFFF" stroke-width="0.45"/>'
    )


def rect(x, y, w, h, fill, stroke="none", width=1):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def main():
    rows = read_rows(INPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    width, height = 980, 610
    left, right, top, bottom = 235, 145, 90, 125
    plot_w = width - left - right
    plot_h = height - top - bottom
    row_y = {
        "Hydrophilicity": top + 70,
        "Charge": top + plot_h / 2,
        "Sigma": top + plot_h - 70,
    }

    shap_values = []
    for feature in FEATURES:
        shap_values.extend(float(r[SHAP_COLS[feature]]) for r in rows)
    xmin, xmax = -1.1, 1.4

    def sx(value):
        return left + (value - xmin) / (xmax - xmin) * plot_w

    parts = []
    parts.append(rect(0, 0, width, height, "#FFFFFF"))
    parts.append(text(40, 40, "SHAP value plot for log_selectivity", 22, weight=700))
    parts.append(text(40, 63, "Color maps each parameter value from low (#93B1F0) to high (#EE8C8E)", 12, fill=MUTED))

    # Full frame: left, bottom, top, right axes.
    parts.append(line(left, top, left, top + plot_h, AXIS, 1.4))
    parts.append(line(left, top + plot_h, left + plot_w, top + plot_h, AXIS, 1.4))
    parts.append(line(left, top, left + plot_w, top, AXIS, 1.0))
    parts.append(line(left + plot_w, top, left + plot_w, top + plot_h, AXIS, 1.0))

    # Grid and x-axis ticks. Bottom tick marks point outward.
    x_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for tick in x_ticks:
        x = sx(tick)
        parts.append(line(x, top, x, top + plot_h, GRID, 1))
        parts.append(line(x, top + plot_h, x, top + plot_h + 8, AXIS, 1.2))
        parts.append(text(x, top + plot_h + 28, f"{tick:.1f}", 12, "middle", fill=TEXT))

    # Center zero reference.
    parts.append(line(sx(0), top, sx(0), top + plot_h, "#9AA3B2", 1.2, "5,5"))

    # Y-axis labels and left outward ticks.
    for feature in FEATURES:
        y = row_y[feature]
        parts.append(line(left - 8, y, left, y, AXIS, 1.2))
        parts.append(text(left - 18, y + 5, feature, 13, "end", fill=TEXT))
        parts.append(line(left, y, left + plot_w, y, "#F2F3F7", 1))

    # Beeswarm-style jittered points.
    for feature in FEATURES:
        values = [float(r[feature]) for r in rows]
        vmin, vmax = min(values), max(values)
        sorted_rows = sorted(rows, key=lambda r: float(r[SHAP_COLS[feature]]))
        for idx, row in enumerate(sorted_rows):
            shap = float(row[SHAP_COLS[feature]])
            val = float(row[feature])
            t = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
            color = lerp_color(LOW_COLOR, HIGH_COLOR, t)
            x = sx(shap)
            # Deterministic jitter, slightly wider for dense rows.
            jitter = ((idx * 37) % 41 - 20) * 0.78
            y = row_y[feature] + jitter
            parts.append(circle(x, y, 4.0, color, 0.86))

    # Axis labels.
    parts.append(text(left + plot_w / 2, height - 42, "SHAP value", 14, "middle", weight=600))
    parts.append(
        text(
            42,
            top + plot_h / 2,
            "Feature",
            14,
            "middle",
            weight=600,
            extra=f'transform="rotate(-90 42 {top + plot_h / 2:.1f})"',
        )
    )

    # Colorbar.
    cb_x, cb_y, cb_w, cb_h = left + plot_w + 42, top + 40, 20, 210
    for i in range(cb_h):
        t = 1 - i / (cb_h - 1)
        parts.append(rect(cb_x, cb_y + i, cb_w, 1.2, lerp_color(LOW_COLOR, HIGH_COLOR, t)))
    parts.append(rect(cb_x, cb_y, cb_w, cb_h, "none", AXIS, 0.8))
    parts.append(text(cb_x + cb_w + 10, cb_y + 5, "High", 12, fill=MUTED))
    parts.append(text(cb_x + cb_w + 10, cb_y + cb_h, "Low", 12, fill=MUTED))
    parts.append(text(cb_x - 26, cb_y + cb_h + 36, "Feature value", 12, fill=MUTED))

    # Small note.
    parts.append(text(left, height - 16, "Axes: top/right borders shown; bottom and left ticks point outward.", 11, fill=MUTED))

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )
    OUTPUT.write_text(svg, encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
