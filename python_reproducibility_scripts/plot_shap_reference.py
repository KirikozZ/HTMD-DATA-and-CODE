import csv
import html
import math
from pathlib import Path


ROOT = Path("outputs") / "gpr_selectivity" / "shap_analysis"
OUT = Path("outputs") / "gpr_selectivity" / "shap_figures"
FEATURES = ["Hydrophilicity", "Charge", "Sigma"]
OUTPUTS = ["FWater", "FLi", "FCl", "FMg", "log_selectivity"]
COLORS = {
    "Hydrophilicity": "#2E8B57",
    "Charge": "#2F6FBB",
    "Sigma": "#E6862E",
    "positive": "#2F6FBB",
    "negative": "#C94C4C",
    "neutral": "#667085",
    "grid": "#D0D5DD",
    "text": "#202124",
    "muted": "#667085",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(row, key):
    return float(row[key])


def esc(s):
    return html.escape(str(s), quote=True)


def save_svg(path, width, height, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#FFFFFF"/>\n'
        '<style>'
        'text{font-family:Arial,Helvetica,sans-serif;fill:#202124}'
        '.title{font-size:22px;font-weight:700}'
        '.subtitle{font-size:12px;fill:#667085}'
        '.axis{font-size:12px;fill:#475467}'
        '.label{font-size:13px}'
        '.small{font-size:11px;fill:#667085}'
        '</style>\n'
        f"{body}\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def text(x, y, content, cls="label", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{esc(content)}</text>'


def line(x1, y1, x2, y2, color="#D0D5DD", width=1, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def rect(x, y, w, h, fill, stroke="none", rx=2, opacity=1.0):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'
    )


def circle(cx, cy, r, fill, opacity=0.85, stroke="#FFFFFF"):
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" opacity="{opacity}" stroke="{stroke}" stroke-width="0.4"/>'


def color_gradient(value, vmin, vmax):
    if vmax <= vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0, min(1, t))
    # Blue to light gray to red-orange.
    if t < 0.5:
        a = t / 0.5
        c1 = (47, 111, 187)
        c2 = (240, 244, 248)
    else:
        a = (t - 0.5) / 0.5
        c1 = (240, 244, 248)
        c2 = (230, 134, 46)
    rgb = tuple(round(c1[i] * (1 - a) + c2[i] * a) for i in range(3))
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def panel_title(parts, x, y, title, subtitle=None):
    parts.append(text(x, y, title, "title"))
    if subtitle:
        parts.append(text(x, y + 20, subtitle, "subtitle"))


def plot_global_importance(importance):
    rows = [r for r in importance if r["output"] in OUTPUTS]
    lookup = {(r["output"], r["feature"]): num(r, "relative_importance") for r in rows}
    width, height = 1040, 610
    left, top = 170, 90
    panel_w, panel_h = 780, 70
    gap = 28
    parts = []
    panel_title(parts, 40, 38, "Global SHAP Importance", "Relative mean(|SHAP|) by output and input parameter")
    for i, out in enumerate(OUTPUTS):
        y = top + i * (panel_h + gap)
        parts.append(text(40, y + 35, out, "label"))
        parts.append(line(left, y + panel_h, left + panel_w, y + panel_h, COLORS["grid"]))
        for tick in [0, 0.25, 0.5, 0.75]:
            x = left + tick * panel_w
            parts.append(line(x, y - 6, x, y + panel_h, "#EAECF0"))
            parts.append(text(x, y + panel_h + 17, f"{int(tick * 100)}%", "small", "middle"))
        bar_h = 16
        for j, feat in enumerate(FEATURES):
            val = lookup[(out, feat)]
            by = y + j * 20
            bw = val * panel_w
            parts.append(rect(left, by, bw, bar_h, COLORS[feat], rx=3))
            parts.append(text(left + bw + 8, by + 12, f"{feat} {val:.1%}", "small"))
    legend_x = 690
    for k, feat in enumerate(FEATURES):
        parts.append(rect(legend_x + k * 115, 42, 12, 12, COLORS[feat], rx=2))
        parts.append(text(legend_x + k * 115 + 18, 53, feat, "small"))
    save_svg(OUT / "fig1_global_importance.svg", width, height, "\n".join(parts))


def plot_heatmap(importance):
    lookup = {(r["output"], r["feature"]): num(r, "relative_importance") for r in importance}
    width, height = 760, 520
    left, top = 185, 105
    cell_w, cell_h = 150, 62
    parts = []
    panel_title(parts, 42, 38, "SHAP Importance Heatmap", "Cell value is relative mean(|SHAP|)")
    maxv = max(lookup[(o, f)] for o in OUTPUTS for f in FEATURES)
    for j, feat in enumerate(FEATURES):
        parts.append(text(left + j * cell_w + cell_w / 2, top - 18, feat, "axis", "middle"))
    for i, out in enumerate(OUTPUTS):
        parts.append(text(left - 16, top + i * cell_h + 38, out, "axis", "end"))
        for j, feat in enumerate(FEATURES):
            val = lookup[(out, feat)]
            x = left + j * cell_w
            y = top + i * cell_h
            fill = color_gradient(val, 0, maxv)
            parts.append(rect(x, y, cell_w - 4, cell_h - 4, fill, "#FFFFFF", rx=2))
            parts.append(text(x + cell_w / 2 - 2, y + 36, f"{val:.0%}", "label", "middle"))
    # Color key
    key_x, key_y = 185, 448
    for k in range(80):
        val = maxv * k / 79
        parts.append(rect(key_x + k * 3, key_y, 3, 12, color_gradient(val, 0, maxv), rx=0))
    parts.append(text(key_x, key_y + 30, "low", "small"))
    parts.append(text(key_x + 240, key_y + 30, "high", "small", "end"))
    save_svg(OUT / "fig2_importance_heatmap.svg", width, height, "\n".join(parts))


def plot_selectivity_beeswarm(shap_rows):
    width, height = 900, 520
    left, right, top = 230, 70, 90
    plot_w = width - left - right
    row_gap = 105
    parts = []
    panel_title(parts, 42, 38, "SHAP Beeswarm: log_selectivity", "Point color encodes parameter value")
    vals = []
    for feat in FEATURES:
        vals += [num(r, f"log_selectivity_shap_{feat}") for r in shap_rows]
    xmin, xmax = min(vals), max(vals)
    pad = (xmax - xmin) * 0.08
    xmin -= pad
    xmax += pad
    zero_x = left + (0 - xmin) / (xmax - xmin) * plot_w
    parts.append(line(zero_x, top - 20, zero_x, top + row_gap * 2 + 45, "#98A2B3", 1.2, "4,4"))
    for tick in [-1.0, -0.5, 0, 0.5, 1.0, 1.5]:
        if xmin <= tick <= xmax:
            x = left + (tick - xmin) / (xmax - xmin) * plot_w
            parts.append(line(x, top - 10, x, top + row_gap * 2 + 30, "#EAECF0"))
            parts.append(text(x, top + row_gap * 2 + 55, f"{tick:.1f}", "small", "middle"))
    for i, feat in enumerate(FEATURES):
        y0 = top + i * row_gap
        parts.append(text(left - 20, y0 + 5, feat, "label", "end"))
        fvals = [num(r, feat) for r in shap_rows]
        vmin, vmax = min(fvals), max(fvals)
        sorted_rows = sorted(shap_rows, key=lambda r: num(r, f"log_selectivity_shap_{feat}"))
        for idx, r in enumerate(sorted_rows):
            sv = num(r, f"log_selectivity_shap_{feat}")
            x = left + (sv - xmin) / (xmax - xmin) * plot_w
            # Deterministic compact jitter.
            jitter = ((idx * 37) % 31 - 15) * 0.95
            color = color_gradient(num(r, feat), vmin, vmax)
            parts.append(circle(x, y0 + jitter, 3.3, color, 0.82))
    parts.append(text(left + plot_w / 2, height - 35, "SHAP value for log_selectivity", "axis", "middle"))
    parts.append(text(width - 155, 64, "low value", "small"))
    parts.append(rect(width - 95, 54, 36, 10, color_gradient(0, 0, 1), rx=0))
    parts.append(rect(width - 58, 54, 36, 10, color_gradient(1, 0, 1), rx=0))
    parts.append(text(width - 16, 64, "high value", "small", "end"))
    save_svg(OUT / "fig3_selectivity_beeswarm.svg", width, height, "\n".join(parts))


def plot_dependence(shap_rows):
    panels = [
        ("Sigma", "log_selectivity", "Sigma effect on Selectivity"),
        ("Charge", "FCl", "Charge effect on FCl"),
        ("Sigma", "FMg", "Sigma effect on FMg"),
        ("Hydrophilicity", "FLi", "Hydrophilicity effect on FLi"),
    ]
    width, height = 1040, 720
    parts = []
    panel_title(parts, 42, 38, "SHAP Dependence Plots", "Key parameter-output relationships")
    pw, ph = 410, 235
    origins = [(90, 105), (570, 105), (90, 405), (570, 405)]
    for (feat, out, title), (x0, y0) in zip(panels, origins):
        shap_key = f"{out}_shap_{feat}"
        xs = [num(r, feat) for r in shap_rows]
        ys = [num(r, shap_key) for r in shap_rows]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xpad = (xmax - xmin) * 0.05 if xmax > xmin else 1
        ypad = (ymax - ymin) * 0.08 if ymax > ymin else 1
        xmin -= xpad
        xmax += xpad
        ymin -= ypad
        ymax += ypad
        parts.append(text(x0, y0 - 18, title, "label"))
        parts.append(rect(x0, y0, pw, ph, "#FFFFFF", "#D0D5DD", rx=4))
        if ymin <= 0 <= ymax:
            zy = y0 + ph - (0 - ymin) / (ymax - ymin) * ph
            parts.append(line(x0, zy, x0 + pw, zy, "#98A2B3", 1, "4,4"))
        for t in [0, 0.25, 0.5, 0.75, 1]:
            gx = x0 + t * pw
            gy = y0 + t * ph
            parts.append(line(gx, y0, gx, y0 + ph, "#F2F4F7"))
            parts.append(line(x0, gy, x0 + pw, gy, "#F2F4F7"))
        fvals = [num(r, feat) for r in shap_rows]
        vmin, vmax = min(fvals), max(fvals)
        for k, r in enumerate(shap_rows):
            x = x0 + (num(r, feat) - xmin) / (xmax - xmin) * pw
            y = y0 + ph - (num(r, shap_key) - ymin) / (ymax - ymin) * ph
            jitter = ((k * 17) % 11 - 5) * 0.55
            parts.append(circle(x + jitter, y, 3, color_gradient(num(r, feat), vmin, vmax), 0.72))
        parts.append(text(x0 + pw / 2, y0 + ph + 32, feat, "small", "middle"))
        parts.append(text(x0 - 44, y0 + ph / 2, f"SHAP {out}", "small", "middle", 'transform="rotate(-90 %.1f %.1f)"' % (x0 - 44, y0 + ph / 2)))
        parts.append(text(x0, y0 + ph + 15, f"{min(xs):.2f}", "small"))
        parts.append(text(x0 + pw, y0 + ph + 15, f"{max(xs):.2f}", "small", "end"))
    save_svg(OUT / "fig4_dependence_key_relations.svg", width, height, "\n".join(parts))


def plot_local_waterfall(local_rows):
    rows = [r for r in local_rows if r["output"] == "log_selectivity"]
    rows = sorted(rows, key=lambda r: abs(num(r, "shap_value")), reverse=True)
    base = num(rows[0], "base_value")
    pred = num(rows[0], "prediction")
    width, height = 920, 430
    left, top, plot_w = 140, 100, 650
    vals = [base]
    running = base
    for r in rows:
        running += num(r, "shap_value")
        vals.append(running)
    xmin, xmax = min(0, min(vals), base, pred), max(vals + [pred])
    pad = (xmax - xmin) * 0.08
    xmin -= pad
    xmax += pad

    def sx(v):
        return left + (v - xmin) / (xmax - xmin) * plot_w

    parts = []
    panel_title(parts, 42, 38, "Local SHAP Waterfall: Optimized Candidate", "Explaining predicted log_selectivity")
    y = top
    bar_h = 34
    parts.append(text(left - 18, y + 23, "base", "label", "end"))
    parts.append(rect(sx(0), y, sx(base) - sx(0), bar_h, "#98A2B3", rx=3))
    parts.append(text(sx(base) + 8, y + 23, f"{base:.3f}", "small"))
    current = base
    for i, r in enumerate(rows, start=1):
        y = top + i * 58
        v = num(r, "shap_value")
        new = current + v
        x1, x2 = sx(current), sx(new)
        parts.append(line(x1, y - 19, x1, y + 34, "#D0D5DD", 1, "3,3"))
        fill = COLORS["positive"] if v >= 0 else COLORS["negative"]
        parts.append(text(left - 18, y + 23, r["feature"], "label", "end"))
        parts.append(rect(min(x1, x2), y, abs(x2 - x1), bar_h, fill, rx=3))
        parts.append(text(max(x1, x2) + 8, y + 23, f"{v:+.3f}", "small"))
        current = new
    y = top + (len(rows) + 1) * 58
    parts.append(text(left - 18, y + 23, "prediction", "label", "end"))
    parts.append(rect(sx(0), y, sx(pred) - sx(0), bar_h, "#111827", rx=3))
    parts.append(text(sx(pred) + 8, y + 23, f"{pred:.3f}", "small"))
    for tick in [0, 1, 2, 3]:
        if xmin <= tick <= xmax:
            parts.append(line(sx(tick), top - 25, sx(tick), y + 52, "#EAECF0"))
            parts.append(text(sx(tick), y + 70, str(tick), "small", "middle"))
    parts.append(text(left + plot_w / 2, height - 26, "log_selectivity", "axis", "middle"))
    save_svg(OUT / "fig5_local_waterfall_selectivity.svg", width, height, "\n".join(parts))


def plot_local_outputs(local_rows):
    width, height = 980, 600
    left, top = 180, 95
    pw, row_h = 650, 74
    rows_by_output = {}
    for r in local_rows:
        if r["output"] in OUTPUTS:
            rows_by_output.setdefault(r["output"], []).append(r)
    max_abs = max(abs(num(r, "shap_value")) for r in local_rows if r["output"] in OUTPUTS)
    parts = []
    panel_title(parts, 42, 38, "Local SHAP Contributions at Optimized Candidate", "Positive bars increase the output; negative bars decrease it")
    zero_x = left + pw / 2
    parts.append(line(zero_x, top - 20, zero_x, top + row_h * len(OUTPUTS), "#98A2B3", 1.2, "4,4"))
    for i, out in enumerate(OUTPUTS):
        y = top + i * row_h
        parts.append(text(left - 18, y + 30, out, "label", "end"))
        parts.append(line(left, y + 37, left + pw, y + 37, "#F2F4F7"))
        for r in rows_by_output[out]:
            v = num(r, "shap_value")
            w = abs(v) / max_abs * (pw / 2 - 25)
            if v >= 0:
                x = zero_x
                fill = COLORS[r["feature"]]
            else:
                x = zero_x - w
                fill = COLORS["negative"]
            # Offset bars by feature for readability.
            feat_idx = FEATURES.index(r["feature"])
            yy = y + 6 + feat_idx * 17
            parts.append(rect(x, yy, w, 13, fill, rx=2, opacity=0.9))
            label_x = x + w + 5 if v >= 0 else x - 5
            anchor = "start" if v >= 0 else "end"
            parts.append(text(label_x, yy + 10, f"{r['feature']} {v:+.2f}", "small", anchor))
    for k, feat in enumerate(FEATURES):
        parts.append(rect(650 + k * 100, 44, 12, 12, COLORS[feat], rx=2))
        parts.append(text(668 + k * 100, 55, feat, "small"))
    parts.append(rect(650, 66, 12, 12, COLORS["negative"], rx=2))
    parts.append(text(668, 77, "negative contribution", "small"))
    save_svg(OUT / "fig6_local_all_outputs.svg", width, height, "\n".join(parts))


def write_reference_note():
    note = """# SHAP figure reference

Generated SVG figures:

1. `fig1_global_importance.svg` - global relative SHAP importance by output.
2. `fig2_importance_heatmap.svg` - compact output-by-feature importance heatmap.
3. `fig3_selectivity_beeswarm.svg` - SHAP distribution for final log-selectivity.
4. `fig4_dependence_key_relations.svg` - key parameter-output dependence plots.
5. `fig5_local_waterfall_selectivity.svg` - local SHAP waterfall for the optimized point.
6. `fig6_local_all_outputs.svg` - local SHAP contributions for water, ions, and selectivity.

Main reading:

- Selectivity is dominated by Sigma, then Charge.
- FWater is mainly controlled by Charge and Hydrophilicity.
- FLi is mainly controlled by Hydrophilicity.
- FCl is mainly controlled by Charge.
- FMg is mainly controlled by Sigma.
- At the optimized point, high Sigma strongly suppresses FMg and improves selectivity.
"""
    (OUT / "figure_reference_notes.md").write_text(note, encoding="utf-8")


def main():
    importance = read_csv(ROOT / "shap_importance.csv")
    shap_rows = read_csv(ROOT / "shap_values_all_outputs.csv")
    local_rows = read_csv(ROOT / "shap_local_optimized_candidate_all_outputs.csv")
    plot_global_importance(importance)
    plot_heatmap(importance)
    plot_selectivity_beeswarm(shap_rows)
    plot_dependence(shap_rows)
    plot_local_waterfall(local_rows)
    plot_local_outputs(local_rows)
    write_reference_note()
    print(f"Saved SHAP reference figures to {OUT}")


if __name__ == "__main__":
    main()
