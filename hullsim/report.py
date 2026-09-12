"""
Reporting: drawings, charts, tables, and a self-contained HTML dashboard.

The brief for this module was 'give us all of the data in a readable fashion,
because what we gain in one design we lose in another'. So nothing here tries
to hand back a single winner. Every chart is built to show a TRADE-OFF: two
things you care about on the two axes, and the cloud of designs between them,
so the shape of the compromise is visible rather than asserted.

Everything renders to one HTML file with the images embedded, so it opens
anywhere with no server, no assets folder and nothing to break when it gets
emailed to a teammate.
"""

from __future__ import annotations

import base64
import html
import io
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import constants as C
from .evaluate import evaluate
from .geometry import panel_warp

INK = "#1b2430"
MUTED = "#7b8794"
ACCENT = "#1f6f8b"
WARN = "#c0563b"
GOOD = "#2e7d5b"
GRID = "#dfe4ea"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
})


def _png(fig) -> str:
    """Render a figure to a base64 data URI and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Hull drawings
# ---------------------------------------------------------------------------

def draw_hull(result: dict) -> str:
    """Profile, plan and body plan, with the waterline on all three.

    Worth looking at before trusting any number: a hull that scores well but
    looks wrong on paper usually means a parameter combination the geometry
    module handled legally and no one would ever build.
    """
    det = result["detail"]
    mesh, eq = det["mesh"], det["equilibrium"]
    d = mesh.design
    x = mesh.x

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.4),
                             gridspec_kw={"height_ratios": [1, 1, 1.25]})

    # -- profile ---------------------------------------------------------
    ax = axes[0]
    ax.plot(x, mesh.keel_z, color=INK, lw=1.8, label="keel (rocker)")
    ax.plot(x, mesh.gunwale_z, color=INK, lw=1.8, label="gunwale (sheer)")
    ax.plot([x[0], x[0]], [mesh.keel_z[0], mesh.gunwale_z[0]], color=INK, lw=1.8)
    ax.plot([x[-1], x[-1]], [mesh.keel_z[-1], mesh.gunwale_z[-1]], color=INK, lw=1.8)
    ax.fill_between(x, mesh.keel_z, mesh.gunwale_z, color=ACCENT, alpha=0.07)
    ax.plot(x, eq.waterline_z, color=ACCENT, lw=1.4, ls="--", label="waterline")
    ax.fill_between(x, mesh.keel_z, eq.waterline_z,
                    where=eq.waterline_z > mesh.keel_z, color=ACCENT, alpha=0.18)
    for xf, w in zip(d.crew_x_frac, d.crew_weights):
        ax.axvline(xf * d.length, color=WARN, lw=1.0, ls=":")
        ax.annotate(f"{w:.0f} lb", (xf * d.length, mesh.gunwale_z.max()),
                    color=WARN, fontsize=7.5, ha="center", va="bottom")
    ax.set_title(f"Profile  ·  {d.length:.0f} in overall, "
                 f"draft {eq.draft:.2f} in, freeboard {eq.freeboard_min:.2f} in")
    ax.set_xlabel("station from bow (in)")
    ax.set_ylabel("height (in)")
    ax.legend(loc="upper right", fontsize=7.5, ncol=3)
    ax.set_aspect("equal", adjustable="datalim")

    # -- plan ------------------------------------------------------------
    ax = axes[1]
    ax.plot(x, mesh.half_beam, color=INK, lw=1.6)
    ax.plot(x, -mesh.half_beam, color=INK, lw=1.6)
    ax.plot(x, mesh.half_bottom, color=MUTED, lw=1.1, ls="--")
    ax.plot(x, -mesh.half_bottom, color=MUTED, lw=1.1, ls="--")
    ax.fill_between(x, -mesh.half_beam, mesh.half_beam, color=ACCENT, alpha=0.07)
    ax.set_title("Plan  ·  solid = gunwale, dashed = flat bottom panel")
    ax.set_xlabel("station from bow (in)")
    ax.set_ylabel("half breadth (in)")
    ax.set_aspect("equal", adjustable="datalim")

    # -- body plan --------------------------------------------------------
    ax = axes[2]
    idx = np.linspace(0, mesh.n_stations - 1, 11).astype(int)
    cmap = plt.get_cmap("viridis")
    for j, i in enumerate(idx):
        v = mesh.verts[i]
        col = cmap(j / max(len(idx) - 1, 1))
        ax.plot(v[:, 0], v[:, 1], color=col, lw=1.1)
    wl = float(np.max(eq.waterline_z))
    ax.axhline(wl, color=ACCENT, ls="--", lw=1.3)
    ax.annotate("waterline", (ax.get_xlim()[1], wl), color=ACCENT,
                fontsize=7.5, va="bottom", ha="right")
    ax.set_title("Body plan  ·  sections bow (dark) to stern (light)")
    ax.set_xlabel("half breadth (in)")
    ax.set_ylabel("height above baseline (in)")
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return _png(fig)


def draw_stability(result: dict) -> str:
    """Righting-arm curve with the angles that matter marked on it."""
    det = result["detail"]
    sc, sea = det["stability"], det["seakeeping"]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(sc.heel_deg, sc.gz, color=ACCENT, lw=2.0, marker="o", ms=3)
    ax.axhline(0, color=MUTED, lw=0.8)

    a = sc.heel_deg[sc.heel_deg <= sc.usable_deg]
    g = np.interp(a, sc.heel_deg, sc.gz)
    ax.fill_between(a, 0, np.clip(g, 0, None), color=ACCENT, alpha=0.15)

    small = np.linspace(0, min(12, sc.heel_deg.max()), 20)
    ax.plot(small, sc.gm0 * np.sin(np.radians(small)), color=MUTED, lw=1.0,
            ls=":", label=f"GM·sin(φ), GM = {sc.gm0:.1f} in")

    if not math.isnan(sc.downflood_deg):
        ax.axvline(sc.downflood_deg, color=WARN, lw=1.3)  # noqa: E501
        ax.annotate(f"gunwale under\n{sc.downflood_deg:.0f}°",
                    (sc.downflood_deg, sc.gz_max * 0.85), color=WARN,
                    fontsize=8, ha="right")
    ax.axvline(sea.paddle_heel_deg, color=GOOD, lw=1.3, ls="--")
    ax.annotate(f"heel from a {result['paddle_lean_in']:.0f} in lean\n"
                f"{sea.paddle_heel_deg:.1f}°",
                (sea.paddle_heel_deg, sc.gz_max * 0.25), color=GOOD,
                fontsize=8, ha="left")

    ax.set_xlabel("heel angle (deg)")
    ax.set_ylabel("righting arm GZ (in)")
    ax.set_title(f"Righting arm  ·  peak {sc.gz_max:.2f} in at "
                 f"{sc.heel_at_gz_max:.0f}°, energy reserve "
                 f"{sc.area_to_downflood:.0f} in·deg")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    return _png(fig)


def draw_resistance(result: dict) -> str:
    """Drag broken into its parts, with the operating point marked."""
    det = result["detail"]
    curve = det["drag_curve"]
    v = np.array([p.v_ms for p in curve])
    rf = np.array([p.r_friction for p in curve])
    rr = np.array([p.r_residuary for p in curve])
    rw = np.array([p.r_waves for p in curve])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    ax.stackplot(v, rf, rr, rw, colors=[ACCENT, WARN, MUTED], alpha=0.8,
                 labels=["friction (ITTC + form)", "residuary / wave-making",
                         "added resistance in chop"])
    ax.axvline(result["terminal_v_ms"], color=INK, lw=1.3, ls="--")
    ax.annotate(f"settles at {result['terminal_v_ms']:.2f} m/s\n"
                f"Fr = {result['froude']:.2f}",
                (result["terminal_v_ms"], max(rf + rr + rw) * 0.6),
                fontsize=8, ha="right")
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("resistance (lbf)")
    ax.set_title("Where the drag comes from")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_ylim(0, float(np.percentile(rf + rr + rw, 92)))

    frac = rr / np.maximum(rf + rr + rw, 1e-9)
    ax2.plot(v, 100 * frac, color=WARN, lw=2.0)
    ax2.axvline(result["terminal_v_ms"], color=INK, lw=1.3, ls="--")
    ax2.set_xlabel("speed (m/s)")
    ax2.set_ylabel("residuary share of total drag (%)")
    ax2.set_title("How much of the answer rests on the weakest model")
    ax2.set_ylim(0, 100)
    fig.tight_layout()
    return _png(fig)


def draw_swamping(result: dict) -> str:
    """The flooding cascade: what each gallon aboard costs."""
    det = result["detail"]
    sw = det["swamping"]
    if sw is None or not sw.steps:
        return ""
    g = np.array([s["gallons"] for s in sw.steps])
    gm = np.array([s["gm_effective_in"] for s in sw.steps])
    gms = np.array([s["gm_solid_in"] for s in sw.steps])
    fb = np.array([s["freeboard_in"] for s in sw.steps])

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.plot(g, gms, color=MUTED, lw=1.3, ls=":", label="GM ignoring free surface")
    ax.plot(g, gm, color=ACCENT, lw=2.0, label="effective GM")
    ax.plot(g, fb, color=GOOD, lw=1.6, label="freeboard")
    ax.axhline(0, color=WARN, lw=1.0)
    if not math.isnan(sw.gallons_to_negative_gm):
        ax.axvline(sw.gallons_to_negative_gm, color=WARN, lw=1.3, ls="--")
        ax.annotate(f"loses stability\nat {sw.gallons_to_negative_gm:.0f} gal",
                    (sw.gallons_to_negative_gm, max(gms) * 0.55), color=WARN,
                    fontsize=8, ha="left")
    ax.set_xlabel("water aboard (US gallons)")
    ax.set_ylabel("inches")
    ax.set_title(f"Flooding cascade  ·  {sw.lanes} free-water lane(s)  ·  "
                 f"reserve {sw.gallons_reserve:.0f} gal")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------------------
# Sweep charts
# ---------------------------------------------------------------------------

def chart_tradeoff(df: pd.DataFrame) -> str:
    """The central picture: speed against stability, with feasibility shown.

    This is the chart the whole project exists to produce. Every design is a
    dot. Infeasible ones are still drawn, in grey, because knowing WHERE the
    feasible region ends is most of the information.
    """
    ok = df[df.feasible]
    bad = df[~df.feasible]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))

    ax = axes[0]
    ax.scatter(bad.time_s, bad.gm_in, s=7, c=GRID, alpha=0.7,
               label=f"fails a constraint ({len(bad)})")
    sc = ax.scatter(ok.time_s, ok.gm_in, s=26, c=ok.expected_score,
                    cmap="viridis", edgecolor="white", linewidth=0.4,
                    label=f"feasible ({len(ok)})")
    fig.colorbar(sc, ax=ax, label="expected score")
    ax.set_xlabel("predicted crossing time (s)")
    ax.set_ylabel("metacentric height GM (in)")
    ax.set_title("Speed against stability")
    ax.legend(fontsize=7.5, loc="upper right")
    if len(df):
        ax.set_xlim(df.time_s.quantile(0.01), df.time_s.quantile(0.97))
        ax.set_ylim(min(-5, df.gm_in.quantile(0.02)), df.gm_in.quantile(0.99))

    ax = axes[1]
    if len(ok):
        sc2 = ax.scatter(ok.time_s, ok.p_finish, s=26, c=ok.swamp_reserve_gal,
                         cmap="plasma", edgecolor="white", linewidth=0.4)
        fig.colorbar(sc2, ax=ax, label="swamping reserve (gal)")
        # Pareto front: fastest design at or above each finish probability.
        pts = ok[["time_s", "p_finish"]].sort_values("time_s").values
        front, best = [], -1.0
        for t, p in pts:
            if p > best:
                front.append((t, p))
                best = p
        if len(front) > 1:
            fx, fy = zip(*front)
            ax.plot(fx, fy, color=WARN, lw=1.5, marker="o", ms=4,
                    label="Pareto front")
            ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlabel("predicted crossing time (s)")
    ax.set_ylabel("estimated probability of finishing")
    ax.set_title("What speed costs in reliability")
    fig.tight_layout()
    return _png(fig)


def chart_sensitivity(df: pd.DataFrame, params: list[str] | None = None) -> str:
    """Which knobs actually move the score, one panel each."""
    ok = df[df.feasible]
    if len(ok) < 12:
        ok = df
    params = params or ["length", "bottom_width", "beam", "side_height",
                        "bow_rocker", "chine_frac", "n_frames",
                        "free_water_lanes", "bow_taper", "chine_tape_layers",
                        "tape_diag_density", "bow_sheer"]
    params = [p for p in params if p in ok.columns and ok[p].nunique() > 1]
    if not params:
        return ""

    ncol = 4
    nrow = int(math.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.3 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, p in enumerate(params):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        sub = ok[[p, "expected_score"]].dropna()
        ax.scatter(sub[p], sub.expected_score, s=9, c=ACCENT, alpha=0.45)
        try:
            xs = sub[p].astype(float)
            bins = np.quantile(xs, np.linspace(0, 1, 7))
            bins = np.unique(bins)
            if len(bins) > 2:
                mids, meds = [], []
                for a, b in zip(bins[:-1], bins[1:]):
                    m = (xs >= a) & (xs <= b)
                    if m.sum() >= 3:
                        mids.append(0.5 * (a + b))
                        meds.append(sub.expected_score[m].median())
                if len(mids) > 1:
                    ax.plot(mids, meds, color=WARN, lw=1.8)
            r = float(np.corrcoef(xs, sub.expected_score)[0, 1])
            ax.set_title(f"{p}   r = {r:+.2f}", fontsize=8.5)
        except Exception:
            ax.set_title(p, fontsize=8.5)
        ax.set_xlabel("")
        if i % ncol == 0:
            ax.set_ylabel("expected score", fontsize=8)
    fig.suptitle("Which parameters move the score  ·  red line is the median trend",
                 fontsize=10, y=1.005)
    fig.tight_layout()
    return _png(fig)


def chart_violations(df: pd.DataFrame) -> str:
    """What is actually stopping designs, ranked."""
    bad = df[~df.feasible]
    if not len(bad):
        return ""
    s = (bad.violations.astype(str).str.split("; ").explode()
         .str.replace(r"-?[\d.]+", "N", regex=True)
         .value_counts().head(14)[::-1])
    fig, ax = plt.subplots(figsize=(7.8, 0.32 * len(s) + 1.1))
    ax.barh(range(len(s)), s.values, color=ACCENT, alpha=0.85)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlabel(f"designs blocked (of {len(df)} evaluated, "
                  f"{len(bad)} infeasible)")
    ax.set_title("What the constraints are actually catching")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _png(fig)


def chart_frontier(df: pd.DataFrame) -> str:
    """Material and tape budgets against performance, the other hard limit."""
    ok = df[df.feasible]
    if not len(ok):
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))

    ax = axes[0]
    bad = df[~df.feasible]
    # Two passes with the feasible set drawn last. In one pass the few hundred
    # feasible points vanish underneath ten thousand grey ones.
    ax.scatter(bad.board_spare_frac * 100, bad.tape_spare_frac * 100,
               s=9, c=GRID, alpha=0.6, label=f"fails ({len(bad)})")
    ax.scatter(ok.board_spare_frac * 100, ok.tape_spare_frac * 100,
               s=16, c=ACCENT, alpha=0.9, edgecolor="white", linewidth=0.3,
               label=f"feasible ({len(ok)})")
    ax.axvline(0, color=WARN, lw=1.2)
    ax.axhline(0, color=WARN, lw=1.2)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_xlabel("cardboard left over (%)")
    ax.set_ylabel("tape left over (%)")
    ax.set_title("Material budget  ·  below zero on either axis = over budget")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)

    ax = axes[1]
    s2 = ax.scatter(ok.hull_weight_lb, ok.time_s, s=22, c=ok.all_up_lb,
                    cmap="cividis", edgecolor="white", linewidth=0.4)
    fig.colorbar(s2, ax=ax, label="all-up weight (lb)")
    ax.set_xlabel("hull weight (lb)")
    ax.set_ylabel("crossing time (s)")
    ax.set_title("What the hull's own weight costs in time")
    fig.tight_layout()
    return _png(fig)


def chart_top_designs(df: pd.DataFrame, n: int = 12) -> str:
    """Parallel coordinates of the leaders: where they agree and differ."""
    ok = df[df.feasible].head(n)
    if len(ok) < 3:
        return ""
    cols = ["length", "bottom_width", "beam", "side_height", "bow_rocker",
            "chine_frac", "n_frames", "free_water_lanes", "time_s", "gm_in",
            "expected_score"]
    cols = [c for c in cols if c in ok.columns]
    vals = ok[cols].astype(float).values
    lo, hi = vals.min(axis=0), vals.max(axis=0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    norm = (vals - lo) / span

    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    cmap = plt.get_cmap("viridis")
    for i in range(len(ok)):
        ax.plot(range(len(cols)), norm[i], color=cmap(1 - i / max(len(ok) - 1, 1)),
                lw=1.5, alpha=0.85)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c}\n{lo[j]:.3g}–{hi[j]:.3g}" for j, c in enumerate(cols)],
                       fontsize=7.5)
    ax.set_yticks([])
    ax.set_ylabel("normalised within this set")
    ax.set_title(f"Top {len(ok)} feasible designs  ·  darker = higher score")
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
:root{--ink:#1b2430;--muted:#69737f;--line:#e3e8ee;--accent:#1f6f8b;
      --warn:#c0563b;--good:#2e7d5b;--bg:#fbfcfd;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:44px 26px 100px}
h1{font-size:27px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:18px;margin:44px 0 6px;padding-top:20px;border-top:1px solid var(--line);
   letter-spacing:-.01em}
h3{font-size:14px;margin:24px 0 6px;color:var(--muted);text-transform:uppercase;
   letter-spacing:.07em;font-weight:600}
p{margin:0 0 12px;max-width:70ch}
.sub{color:var(--muted);margin-bottom:26px;font-size:14px}
img{max-width:100%;display:block;margin:14px 0;border:1px solid var(--line);
    border-radius:7px;background:#fff}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 18px;
      background:#fff;border:1px solid var(--line);border-radius:7px;overflow:hidden}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{background:#f2f5f8;font-weight:600;font-size:12px;letter-spacing:.02em}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
       gap:11px;margin:18px 0 22px}
.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card .k{font-size:11px;color:var(--muted);text-transform:uppercase;
         letter-spacing:.06em;margin-bottom:3px}
.card .v{font-size:21px;font-weight:600;font-variant-numeric:tabular-nums;
         letter-spacing:-.01em}
.card .n{font-size:11.5px;color:var(--muted);margin-top:2px}
.ok{color:var(--good)} .bad{color:var(--warn)}
.note{background:#fff;border-left:3px solid var(--accent);padding:12px 16px;
      margin:16px 0;font-size:13.5px;border-radius:0 6px 6px 0}
.note.warn{border-left-color:var(--warn)}
.note b{font-weight:600}
ul{margin:8px 0 14px;padding-left:20px} li{margin:3px 0;max-width:70ch}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:#f2f5f8;
     padding:1px 5px;border-radius:4px}
.foot{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);
      color:var(--muted);font-size:12.5px}
"""


def _card(k, v, note="", cls="") -> str:
    return (f'<div class="card"><div class="k">{html.escape(k)}</div>'
            f'<div class="v {cls}">{v}</div>'
            f'<div class="n">{html.escape(note)}</div></div>')


def _table(rows, headers, numeric_from=1) -> str:
    h = "".join(f'<th class="{"num" if i >= numeric_from else ""}">{html.escape(str(x))}</th>'
                for i, x in enumerate(headers))
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{x}</td>'
            for i, x in enumerate(r)) + "</tr>"
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def single_design_html(result: dict) -> str:
    """Full report for one hull."""
    det = result["detail"]
    d, mesh, eq = det["design"], det["mesh"], det["equilibrium"]
    sc, st, mat = det["stability"], det["structure"], det["materials"]
    sea, race, sw = det["seakeeping"], det["race"], det["swamping"]

    feas = result["feasible"]
    parts = []

    parts.append('<div class="cards">')
    parts.append(_card("Verdict", "FEASIBLE" if feas else "FAILS",
                       "all constraints pass" if feas else
                       f"{result['n_violations']} constraint(s) violated",
                       "ok" if feas else "bad"))
    parts.append(_card("Crossing time", f"{result['time_s']:.1f} s",
                       f"settles at {result['terminal_v_ms']:.2f} m/s, "
                       f"Fr {result['froude']:.2f}"))
    parts.append(_card("Expected score", f"{result['expected_score']:.1f}",
                       f"P(finish) {result['p_finish']:.0%}"))
    parts.append(_card("Draft", f"{eq.draft:.2f} in",
                       f"{result['draft_frac']:.0%} of side height"))
    parts.append(_card("Freeboard", f"{eq.freeboard_min:.2f} in",
                       f"lowest point at x = {eq.freeboard_at:.0f} in"))
    parts.append(_card("GM", f"{eq.gm_t:+.2f} in",
                       f"{result['gm_effective_in']:+.1f} with a gallon aboard"))
    parts.append(_card("Gunwale under at",
                       "never" if math.isnan(sc.downflood_deg)
                       else f"{sc.downflood_deg:.0f}°",
                       f"crew lean already uses {sea.paddle_heel_deg:.1f}°"))
    parts.append(_card("Swamping reserve", f"{sw.gallons_reserve:.0f} gal",
                       f"{d.free_water_lanes} free-water lane(s)"))
    parts.append(_card("Hull weight", f"{result['hull_weight_lb']:.1f} lb",
                       f"all-up {result['all_up_lb']:.0f} lb"))
    parts.append(_card("Cardboard", f"{mat.board_spare_frac:+.0%}",
                       "spare on one roll",
                       "bad" if mat.board_over else "ok"))
    parts.append(_card("Tape", f"{mat.tape_spare_frac:+.0%}",
                       "spare on one roll",
                       "bad" if mat.tape_over else "ok"))
    parts.append(_card("Twist under paddling", f"{st.torsion.twist_actual_deg:.1f}°",
                       st.torsion.verdict))
    parts.append("</div>")

    if not feas:
        parts.append('<div class="note warn"><b>Why it fails.</b><ul>'
                     + "".join(f"<li>{html.escape(v)}</li>"
                               for v in det["violations"]) + "</ul></div>")

    parts.append("<h2>What she looks like</h2>")
    parts.append(f'<img src="{draw_hull(result)}" alt="hull drawings">')
    warp = panel_warp(mesh).max()
    parts.append(
        f'<div class="note"><b>Buildability.</b> Worst flare-panel warp is '
        f'<b>{warp:.3f}</b>. Under 0.01 the panel folds flat from the roll; '
        f'over about 0.055 it needs darts and the flutes crush along the '
        f'chine. Rocker, plan taper and flare are each free on their own — '
        f'it is combining all three that makes a side panel non-developable, '
        f'and that is what this number measures.</div>')

    parts.append("<h2>Stability</h2>")
    parts.append(f'<img src="{draw_stability(result)}" alt="righting arm curve">')
    parts.append(_table([
        ["Centre of buoyancy above keel, KB", f"{eq.kb:.2f} in"],
        ["Metacentric radius, BM", f"{eq.bm_t:.2f} in"],
        ["Centre of gravity above keel, KG", f"{det['mass'].kg:.2f} in"],
        ["GM = KB + BM − KG", f"{eq.gm_t:+.2f} in"],
        ["Free-surface loss, one gallon aboard", f"{result['free_surface_loss_in']:.2f} in"],
        ["Peak righting arm", f"{sc.gz_max:.2f} in at {sc.heel_at_gz_max:.0f}°"],
        ["Gunwale immerses at",
         "never, within the computed heel range" if math.isnan(sc.downflood_deg)
         else f"{sc.downflood_deg:.1f}°"],
        ["Righting arm vanishes at",
         "beyond the computed range" if math.isnan(sc.vanishing_deg)
         else f"{sc.vanishing_deg:.1f}°"],
        ["Energy reserve to downflooding", f"{sc.area_to_downflood:.0f} in·deg"],
        ["Heel from one paddler leaning "
         f"{d.paddle_lean_in:.0f} in", f"{sea.paddle_heel_deg:.1f}°"],
    ], ["Stability", "Value"]))

    parts.append("<h3>Inclining test — check this on the finished boat</h3>")
    parts.append("<p>Float her with both paddlers in racing position, hang a "
                 "plumb line from a thwart, shift a known weight across the "
                 "beam and measure the swing. A bigger swing than predicted "
                 "means the real GM is lower, and the usual cause is a crew "
                 "centre of gravity higher than assumed — which is the "
                 "cheapest thing left to change.</p>")
    parts.append(_table(
        [[f"{i['shift_lb']:.0f} lb", f"{i['lever_in']:.0f} in",
          f"{i['heel_deg']:.1f}°", f"{i['plumb_swing_in']:.2f} in"]
         for i in det["inclining"]],
        ["Shift", "Lever", "Predicted heel", "Plumb swing (24 in line)"]))

    parts.append("<h2>Speed</h2>")
    parts.append(f'<img src="{draw_resistance(result)}" alt="resistance breakdown">')
    parts.append(_table([
        ["Crossing time from a standing start", f"{race.time_s:.1f} s"],
        ["Steady speed if the course were longer", f"{race.terminal_v:.2f} m/s"],
        ["Time lost to acceleration", f"{race.accel_penalty_s:.1f} s"],
        ["Distance to reach 90% of top speed",
         "n/a" if math.isnan(race.distance_to_90pct) else f"{race.distance_to_90pct:.0f} m"],
        ["Froude number at speed", f"{result['froude']:.3f}"],
        ["Form factor (1+k)", f"{result['form_factor']:.2f}"],
        ["Friction drag", f"{result['drag_friction_lb']:.1f} lbf"],
        ["Residuary drag", f"{result['drag_residuary_lb']:.1f} lbf"],
        ["Hydrodynamic power", f"{result['hydro_power_w']:.0f} W"],
        ["Crew shaft power assumed", f"{sum(d.crew_power_w):.0f} W"],
    ], ["Speed", "Value"]))
    parts.append("<h3>Pacing</h3>")
    parts.append("<p>The same total effort, spread differently across the "
                 "course. Power within the race is not assumed constant: the "
                 "crew has a sustainable output and a finite reserve to spend "
                 "above it, and once that reserve is gone they are pinned at "
                 "the sustainable level whether they like it or not.</p>")
    pac_rows = []
    for pac in ("even", "fast_start", "hard_start", "negative_split"):
        rr = evaluate(d.with_(pacing=pac), detail=False)
        nice = {"even": "Even the whole way",
                "fast_start": "Brief lift off the line",
                "hard_start": "Sprint it and hang on",
                "negative_split": "Hold back, then build"}[pac]
        pac_rows.append([
            nice + (" &nbsp;<b>&larr; planned</b>" if pac == d.pacing else ""),
            f"{rr['time_s']:.1f} s",
            f"{rr['time_s'] - result['time_s']:+.1f} s",
            f"{rr['power_start_w']:.0f} / {rr['power_finish_w']:.0f} W",
            f"{rr['fade_frac']:.0%}",
            f"{rr['v_peak_ms']:.2f} m/s",
        ])
    parts.append(_table(pac_rows, ["Strategy", "Time", "vs planned",
                                   "Power start / finish", "Fade", "Peak speed"]))
    parts.append(
        '<div class="note">Going out hard is punished here more than it would '
        'be on most boats. Residuary resistance climbs very steeply past '
        'Froude 0.4, which is right where this hull settles, so overspeeding '
        'early does not cost a little extra drag — it costs a lot, and that '
        'energy comes straight out of a reserve that does not refill inside '
        'seventy-five seconds. Start controlled and hold it.</div>')

    share = result["drag_residuary_lb"] / max(result["drag_lb"], 1e-9)
    parts.append(
        f'<div class="note warn"><b>Read the time with suspicion.</b> '
        f'Residuary resistance is <b>{share:.0%}</b> of total drag at this '
        f'speed, and residuary is the one part of the model that is a '
        f'correlation rather than a calculation. Friction comes from the '
        f'ITTC line on a wetted surface computed from the real geometry, and '
        f'is solid. Wave-making comes from a standard-series curve fitted to '
        f'hulls nothing like a cardboard shoebox. Paddle a measured distance, '
        f'time it, and run <code>scripts/calibrate.py</code> — after that the '
        f'model extrapolates from your own measurement instead of a textbook.'
        f'</div>')

    parts.append("<h2>Taking water aboard</h2>")
    img = draw_swamping(result)
    if img:
        parts.append(f'<img src="{img}" alt="swamping cascade">')
    parts.append(_table([
        ["Chop assumed", f"{sea.chop_in:.1f} in significant height"],
        ["Wavelength", f"{sea.wavelength_in:.0f} in"],
        ["Relative motion RMS", f"{sea.sigma_relative_in:.2f} in"],
        ["Static freeboard", f"{sea.static_freeboard_in:.2f} in"],
        ["Lost to crew lean", f"{sea.heel_freeboard_loss_in:.2f} in"],
        ["Effective freeboard", f"{sea.effective_freeboard_in:.2f} in"],
        ["Wave encounters over the crossing", f"{sea.n_encounters:.0f}"],
        ["Expected shipping events", f"{sea.expected_shipping_events:.3g}"],
        ["Water she can take before losing stability",
         f"{sw.gallons_reserve:.0f} gal"],
        ["Verdict", sea.verdict],
    ], ["Seakeeping", "Value"]))

    cr = det["crew"]
    parts.append("<h3>The crew</h3>")
    rows = []
    for i, (g, rr) in enumerate(zip(cr["geometries"], det["reaches"])):
        rows.append([
            f"Paddler {i + 1} &mdash; {d.crew_weights[i]:.0f} lb, "
            f"{g.height_in:.0f} in tall",
            g.posture.replace("_", " "),
            f"{g.cg_z:.1f} in", f"{g.shoulder_z:.1f} in",
            f"{rr['clearance_in']:.1f} in", f"{rr['efficiency']:.0%}",
        ])
    parts.append(_table(rows, ["Who", "Posture", "CG above floor",
                               "Shoulder", "Over the rail", "Stroke left"]))
    parts.append(
        f'<div class="note"><b>Posture is a trade, not a free win.</b> '
        f'Sitting lower drops the centre of gravity, which every stability '
        f'number rewards, and drops the shoulders toward the gunwale, which '
        f'costs stroke. Together the two give an optimum in the middle rather '
        f'than at the bottom — and a short crew has an advantage here, because '
        f'they kneel lower for the same posture. Kneeling and seating '
        f'positions are both swept, so the ranking has already weighed this. '
        f'The crew sit {det["spacing"]["gap_in"]:.0f} in apart; any closer and '
        f'knees overlap or paddles clash.</div>')

    parts.append("<h3>Time in the water</h3>")
    parts.append(_table([
        ["Afloat before the gun", f"{d.pre_race_soak_min:.1f} min"],
        ["Plus the crossing", f"{result['time_s'] / 60:.1f} min"],
        ["Total when she finishes", f"{result['minutes_afloat']:.1f} min"],
        ["Board strength left at the finish",
         f"{result['wet_factor_at_finish']:.0%} of dry"],
        ["Long-immersion floor", f"{d.wet_factor:.0%}"],
    ], ["Soaking", "Value"]))
    parts.append(
        '<div class="note"><b>Launch late.</b> Board strength decays toward '
        'its wet floor with a time constant of about six minutes, and the '
        'race itself is barely seventy-five seconds. Almost all the soaking '
        'a hull suffers happens while it sits at the line waiting for the '
        'start, so ten minutes of early launching costs far more strength '
        'than the entire crossing does. Anchored to a published figure: a '
        'coated corrugated board lost 47% of its flat crush strength after '
        'five minutes immersed.</div>')

    bd = det["boarding"]
    parts.append("<h3>Getting in</h3>")
    parts.append(_table([
        ["Heel while the second paddler boards", f"{bd.heel_deg:.1f}°"],
        ["Low gunwale drops by", f"{bd.low_gunwale_drop_in:.2f} in"],
        ["Rail left above water", f"{bd.freeboard_left_in:.2f} in"],
        ["Righting arm demanded vs available",
         f"{bd.gz_demand_in:.2f} / {bd.gz_available_in:.2f} in"],
        ["Verdict", bd.verdict],
    ], ["Boarding", "Value"]))
    parts.append(
        '<div class="note"><b>The crew is the bigger wave.</b> An inch of '
        'ripple moves the surface about a quarter inch. A paddler shifting '
        f'{d.paddle_lean_in:.0f} inches off the centreline drops the low '
        f'gunwale by <b>{sea.heel_freeboard_loss_in:.2f} in</b> — several '
        'times more, on every stroke. Freeboard enters the wetness criterion '
        'squared and inside an exponential, so it is a cliff edge rather than '
        'a linear margin: the last two inches are worth far more than the '
        'first two.</div>')

    parts.append("<h2>Structure</h2>")
    tor, gir, pan = st.torsion, st.girder, st.panel
    parts.append(_table([
        ["Torsional stiffness, decked vs open", f"{tor.stiffness_ratio:.1f}×"],
        ["Twist under paddling torque", f"{tor.twist_actual_deg:.1f}°  ({tor.verdict})"],
        ["Frame spacing", f"{tor.frame_spacing_in:.0f} in"],
        ["Peak hull-girder moment",
         f"{gir.max_moment_lbin:,.0f} lb·in at x = {gir.moment_at_in:.0f} in"],
        ["Girder stress vs allowable",
         f"{gir.stress_psi:.0f} / {gir.allowable_psi:.0f} psi "
         f"({gir.utilisation:.0%})"],
        ["Girder verdict", gir.verdict],
        ["Bottom panel deflection",
         f"{pan.deflection_water_in + pan.deflection_knee_in:.2f} in "
         f"(bending-only theory would say {pan.deflection_linear_in:.2f})"],
        ["Chine tape tension",
         f"{pan.chine_tape_tension_lb_in:.1f} lb/in "
         f"({pan.chine_tape_utilisation:.0%} of {d.chine_tape_layers} layer(s))"],
        ["Frames needed to hold half an inch", f"{pan.frames_needed_for_half_inch}"],
        ["Panel verdict", pan.verdict],
        ["Gunwale tube fails by", st.gunwale["governing_mode"]],
        ["Joint type", f"{d.joint_type} (×{st.joint_multiplier:.2f}, "
                       f"{st.joint_source} data)"],
    ], ["Structure", "Value"]))
    if st.flags:
        parts.append('<div class="note warn"><b>Structural flags.</b><ul>'
                     + "".join(f"<li>{html.escape(f)}</li>" for f in st.flags)
                     + "</ul></div>")

    parts.append("<h2>Cut list</h2>")
    parts.append(_table(
        [[n, f"{a:,.0f}", f"{a / C.SHEET_AREA_IN2 * 100:.1f}%"]
         for n, a in mat.board_items]
        + [["<b>TOTAL</b>", f"<b>{mat.board_used_in2:,.0f}</b>",
            f"<b>{mat.board_used_in2 / C.SHEET_AREA_IN2 * 100:.1f}%</b>"],
           ["Available on one 55 ft × 41 in roll",
            f"{C.SHEET_AREA_IN2:,.0f}", "100%"]],
        ["Cardboard part", "Area (in²)", "Share of roll"]))
    parts.append(_table(
        [[n, f"{v:,.0f}", f"{v / C.TAPE_ROLL_IN * 100:.1f}%"]
         for n, v in mat.tape_items]
        + [["<b>TOTAL</b>", f"<b>{mat.tape_used_in:,.0f}</b>",
            f"<b>{mat.tape_used_in / C.TAPE_ROLL_IN * 100:.1f}%</b>"],
           ["Available on one 100 m roll", f"{C.TAPE_ROLL_IN:,.0f}", "100%"]],
        ["Tape run", "Length (in)", "Share of roll"]))
    parts.append(
        f'<div class="note"><b>Weight closes the loop.</b> Cardboard '
        f'{mat.board_weight_lb:.1f} lb plus tape {mat.tape_weight_lb:.1f} lb '
        f'gives a hull of {mat.hull_weight_lb:.1f} lb, and that number went '
        f'straight back into the displacement solve above. Tape is not a '
        f'rounding error — a full roll is about two pounds. Weigh a measured '
        f'square of your actual board and put it in '
        f'<code>data/joint_tests.json</code>; it is the single measurement '
        f'that tightens the most downstream numbers.</div>')

    return "\n".join(parts)


def write_report(path: str | Path, *, title: str, subtitle: str = "",
                 sections: list[tuple[str, str]] = ()) -> Path:
    """Write a self-contained HTML report."""
    body = "".join(f"{s}" for _, s in sections)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>
<div class="wrap"><h1>{html.escape(title)}</h1>
<div class="sub">{subtitle}</div>{body}
<div class="foot">Generated {datetime.now():%Y-%m-%d %H:%M} by hullsim.
Every constant is labelled in <code>hullsim/constants.py</code> as exact,
typical or needs-measuring. The physics is only as good as the numbers you
have not measured yet.</div></div></body></html>"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p


def sweep_html(df: pd.DataFrame, *, n_top: int = 20) -> str:
    """Report body for a whole sweep."""
    ok = df[df.feasible]
    parts = []

    parts.append('<div class="cards">')
    parts.append(_card("Designs evaluated", f"{len(df):,}"))
    parts.append(_card("Feasible", f"{len(ok):,}",
                       f"{len(ok) / max(len(df), 1):.0%} of the space"))
    if len(ok):
        parts.append(_card("Fastest feasible", f"{ok.time_s.min():.1f} s"))
        parts.append(_card("Best expected score", f"{ok.expected_score.max():.1f}"))
        parts.append(_card("Best P(finish)", f"{ok.p_finish.max():.0%}"))
        parts.append(_card("Time spread", f"{ok.time_s.max() - ok.time_s.min():.0f} s",
                           "across feasible designs"))
    parts.append("</div>")

    if not len(ok):
        parts.append('<div class="note warn"><b>Nothing passed.</b> Either the '
                     'search space misses the feasible region, or a limit in '
                     '<code>scoring.Limits</code> is stricter than it needs to '
                     'be. The chart below says which constraint to look at '
                     'first.</div>')
        parts.append(f'<img src="{chart_violations(df)}">')
        return "\n".join(parts)

    parts.append("<h2>The trade-off</h2>")
    parts.append(f'<img src="{chart_tradeoff(df)}" alt="trade-off scatter">')
    parts.append(
        '<div class="note"><b>How to read this.</b> Grey dots fail at least '
        'one hard constraint and are drawn anyway, because where the feasible '
        'region ENDS is most of what a sweep tells you. On the right, the red '
        'line is the Pareto front: for any point below it there exists a '
        'design that is both faster and more likely to finish, so only the '
        'front is worth arguing about. Everything else is dominated.</div>')

    parts.append("<h2>What stops a design</h2>")
    parts.append(f'<img src="{chart_violations(df)}" alt="violations">')

    parts.append("<h2>Which knobs matter</h2>")
    img = chart_sensitivity(df)
    if img:
        parts.append(f'<img src="{img}" alt="parameter sensitivity">')
        parts.append(
            '<div class="note">Correlations here are marginal, not causal — '
            'the other parameters are varying at the same time. A flat trend '
            'means the parameter does not matter much ON ITS OWN, which is '
            'not the same as not mattering. Use this to pick two or three '
            'axes worth a proper grid, then read the grid.</div>')

    parts.append("<h2>Budgets and weight</h2>")
    img = chart_frontier(df)
    if img:
        parts.append(f'<img src="{img}" alt="budget frontier">')

    parts.append("<h2>The leaders</h2>")
    img = chart_top_designs(df)
    if img:
        parts.append(f'<img src="{img}" alt="top designs">')

    cols = [("length", "L"), ("bottom_width", "B.bot"), ("beam", "Beam"),
            ("side_height", "Depth"), ("bow_rocker", "Rocker"),
            ("chine_frac", "Chine"), ("n_frames", "Frames"),
            ("free_water_lanes", "Lanes"), ("chine_tape_layers", "ChTape"),
            ("time_s", "Time s"), ("gm_in", "GM"), ("freeboard_in", "FB"),
            ("downflood_deg", "Flood°"), ("swamp_reserve_gal", "Res gal"),
            ("panel_warp", "Warp"), ("p_finish", "P(fin)"),
            ("expected_score", "Score")]
    cols = [(c, h) for c, h in cols if c in ok.columns]
    rows = []
    for _, r in ok.head(n_top).iterrows():
        rows.append([f"{r[c]:.3g}" if isinstance(r[c], float) else str(r[c])
                     for c, _ in cols])
    parts.append(_table(rows, [h for _, h in cols], numeric_from=0))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Monte Carlo charts
# ---------------------------------------------------------------------------

def _pct_lines(ax, s, fmt="{:.0f}", vertical=True):
    """Mark the 10th, 50th and 90th percentiles on a distribution."""
    for p, style, lw in ((10, ":", 1.1), (50, "-", 1.6), (90, ":", 1.1)):
        v = float(np.percentile(s, p))
        (ax.axvline if vertical else ax.axhline)(v, color=INK, ls=style, lw=lw)
        ax.annotate(f"p{p} {fmt.format(v)}", (v, ax.get_ylim()[1]),
                    rotation=90, va="top", ha="right", fontsize=7,
                    color=INK, xytext=(-2, -3), textcoords="offset points")


def chart_mc_distributions(df: pd.DataFrame) -> str:
    """What the same design actually does across many builds and many days."""
    specs = [("time_s", "crossing time (s)", "{:.0f}", ACCENT),
             ("expected_score", "expected score", "{:.0f}", GOOD),
             ("gm_in", "metacentric height GM (in)", "{:.1f}", ACCENT),
             ("freeboard_in", "freeboard (in)", "{:.1f}", GOOD),
             ("gz_margin_ratio", "righting arm / crew demand", "{:.2f}", WARN),
             ("swamp_reserve_gal", "swamping reserve (gal)", "{:.0f}", MUTED)]
    specs = [s for s in specs if s[0] in df.columns and df[s[0]].notna().any()]

    ncol = 3
    nrow = int(math.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, (col, label, fmt, colour) in enumerate(specs):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        s = df[col].dropna()
        s = s[np.isfinite(s)]
        if not len(s):
            continue
        lo, hi = np.percentile(s, 0.5), np.percentile(s, 99.5)
        ax.hist(np.clip(s, lo, hi), bins=44, color=colour, alpha=0.75)
        _pct_lines(ax, s, fmt)
        ax.set_xlabel(label, fontsize=8.5)
        ax.set_yticks([])
    fig.suptitle("Same design, many builds and many days", fontsize=10.5,
                 y=1.005)
    fig.tight_layout()
    return _png(fig)


def chart_time_cdf(df: pd.DataFrame) -> str:
    """Probability of beating any given time. Reads directly as a race plan."""
    s = df.time_s.dropna()
    s = np.sort(s[np.isfinite(s)].to_numpy())
    if not len(s):
        return ""
    y = np.arange(1, len(s) + 1) / len(s)

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(s, 100 * y, color=ACCENT, lw=2.0)
    for p in (10, 50, 90):
        v = float(np.percentile(s, p))
        ax.plot([v, v], [0, p], color=MUTED, ls=":", lw=1.0)
        ax.plot([s[0], v], [p, p], color=MUTED, ls=":", lw=1.0)
        ax.annotate(f"{p}% under {v:.0f}s", (v, p), fontsize=8,
                    xytext=(4, -10), textcoords="offset points", color=INK)
    ax.set_xlabel("crossing time (s)")
    ax.set_ylabel("chance of being this fast or faster (%)")
    ax.set_title("How likely is any given time")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return _png(fig)


def chart_tornado(sens: pd.DataFrame, target: str = "expected score",
                  top: int = 12) -> str:
    """Which unknown is driving the spread. Read the top three.

    This is the chart that turns uncertainty into a to-do list: a long bar
    against a measurable quantity means measuring it collapses that much of
    the spread.
    """
    if sens is None or not len(sens):
        return ""
    s = sens.head(top)[::-1]
    labels = [_pretty_input(n) for n in s.input]
    colours = [WARN if v < 0 else ACCENT for v in s.spearman]

    fig, ax = plt.subplots(figsize=(7.8, 0.33 * len(s) + 1.2))
    ax.barh(range(len(s)), s.spearman, color=colours, alpha=0.85)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_xlabel(f"rank correlation with {target}   "
                  f"(blue raises it, red lowers it)")
    ax.set_title(f"What is driving the spread in {target}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _png(fig)


_INPUT_NAMES = {
    "u_residuary_scale": "wave-making (residuary scale)  ← time a paddle",
    "u_g_board_psi": "board shear modulus  ← twist a test tube",
    "u_board_quality": "overall board stiffness/quality",
    "u_paperboard_thickness_mm": "board thickness (unknown, 1-2.4mm)  ← measure it",
    "u_paperboard_density": "board density (unknown)",
    "u_paperboard_stiffness_mult": "board stiffness vs. estimate",
    "u_board_ect": "board edge crush strength",
    "u_board_stiffness": "board bending stiffness",
    "u_board_areal": "board weight per area  ← weigh a square",
    "u_tape_tensile": "tape tensile strength",
    "u_blade_efficiency": "paddle blade efficiency",
    "u_power_total_w": "crew power on the day",
    "u_pacing_slip": "went out harder than planned",
    "u_anaerobic_fraction": "how much of their effort is burnable reserve",
    "u_worst_stroke_lean_in": "worst paddle stroke lean",
    "u_crew_kg_mean": "crew centre of gravity height",
    "u_chop_in": "chop on the day",
    "u_headwind_mph": "headwind on the day",
    "u_extra_soak_min": "extra minutes afloat before the gun",
    "u_splash_gal": "water shipped while boarding",
    "u_boarding_offset": "how clumsily she is boarded",
    "u_build_list_in": "build asymmetry (permanent list)",
    "u_extra_weight_lb": "unplanned extra weight",
    "u_dropped_frame": "fitted one fewer frame",
    "u_skimped_chine_tape": "skimped a chine tape layer",
    "u_tape_density": "diagonal tape coverage actually applied",
    "u_wet_factor": "how well the seams were sealed",
    "u_length": "length cut error", "u_beam": "beam cut error",
    "u_bottom_width": "bottom width cut error",
    "u_side_height": "depth cut error",
    "u_bow_rocker": "bow rocker error", "u_stern_rocker": "stern rocker error",
    "u_bow_sheer": "bow sheer error", "u_stern_sheer": "stern sheer error",
    "u_bow_taper": "bow taper error", "u_stern_taper": "stern taper error",
    "u_chine_frac": "chine height error",
    "b_length": "as-built length", "b_beam": "as-built beam",
    "b_bottom_width": "as-built bottom width",
    "b_side_height": "as-built depth",
}


def _pretty_input(name: str) -> str:
    return _INPUT_NAMES.get(name, name.replace("u_", "").replace("b_", "as-built ")
                            .replace("_", " "))


def chart_variance_sources(by_source: dict) -> str:
    """Where the spread comes from: building, racing, or not knowing.

    Three separate Monte Carlo runs with one source switched on at a time,
    plotted against the run with all three. The comparison is the useful
    part, because the three have different remedies -- better jigs, more
    practice, or a measurement.
    """
    order = [k for k in ("build", "race", "model", "all") if k in by_source]
    if len(order) < 2:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))

    labels = {"build": "how you build it", "race": "the day you race",
              "model": "what we do not know", "all": "everything together"}

    for ax, col, name in ((axes[0], "time_s", "crossing time (s)"),
                          (axes[1], "expected_score", "expected score")):
        data = [by_source[k][col].dropna().to_numpy() for k in order]
        data = [d[np.isfinite(d)] for d in data]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for b in parts["bodies"]:
            b.set_facecolor(ACCENT)
            b.set_alpha(0.55)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color(INK)
                parts[key].set_linewidth(1.1)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([labels[k] for k in order], fontsize=8.5)
        ax.set_ylabel(name)
        sds = [f"{np.std(d):.1f}" for d in data]
        ax.set_title(f"{name}   (sd: {', '.join(sds)})", fontsize=9.5)
    fig.suptitle("Where the uncertainty comes from", fontsize=10.5, y=1.02)
    fig.tight_layout()
    return _png(fig)


def chart_robust_vs_nominal(summary: pd.DataFrame) -> str:
    """Designs that look good on paper against designs that hold up.

    Points well below the diagonal are fragile: they score well at their
    drawn dimensions and lose it as soon as the build or the day moves. That
    gap is invisible to a deterministic sweep, and it is the whole reason to
    run this.
    """
    if summary is None or len(summary) < 3:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))

    ax = axes[0]
    x = summary["score_mean"] if "score_mean" in summary else summary["robust_score"]
    sc = ax.scatter(x, summary.robust_score, s=34, c=summary.p_feasible,
                    cmap="viridis", vmin=0, vmax=1, edgecolor="white",
                    linewidth=0.4)
    fig.colorbar(sc, ax=ax, label="fraction of builds that pass")
    lo = float(min(x.min(), summary.robust_score.min())) - 2
    hi = float(max(x.max(), summary.robust_score.max())) + 2
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls=":", lw=1.0)
    ax.set_xlabel("average score across trials")
    ax.set_ylabel("bad-day score (10th percentile)")
    ax.set_title("Fragile designs sit far below the dotted line")

    ax = axes[1]
    ax.scatter(summary.time_p50, summary.p_feasible * 100, s=34,
               c=summary.robust_score, cmap="plasma", edgecolor="white",
               linewidth=0.4)
    ax.set_xlabel("median crossing time (s)")
    ax.set_ylabel("builds that pass every constraint (%)")
    ax.set_title("Speed against robustness")
    fig.tight_layout()
    return _png(fig)


def chart_mc_failures(df: pd.DataFrame) -> str:
    """Which constraint actually bites, and in what fraction of trials."""
    from .montecarlo import failure_rates
    s = failure_rates(df)
    if not len(s):
        return ""
    s = s.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 0.33 * len(s) + 1.1))
    ax.barh(range(len(s)), 100 * s.values, color=WARN, alpha=0.85)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlabel("percent of trials in which this constraint fails")
    ax.set_title("What goes wrong, and how often")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _png(fig)


def montecarlo_html(df: pd.DataFrame, design, summary: dict,
                    by_source: dict | None = None,
                    comparison: pd.DataFrame | None = None) -> str:
    """Report body for a Monte Carlo run on one design."""
    from .montecarlo import sensitivity

    parts = []
    pf = summary["p_feasible"]

    parts.append('<div class="cards">')
    parts.append(_card("Trials", f"{summary['n_trials']:,}",
                       "each a different build and a different day"))
    parts.append(_card("Builds that pass", f"{pf:.0%}",
                       "every hard constraint held",
                       "ok" if pf >= 0.85 else ("bad" if pf < 0.6 else "")))
    parts.append(_card("Crossing time", f"{summary['time_p50']:.0f} s",
                       f"p10 {summary['time_p10']:.0f} · "
                       f"p90 {summary['time_p90']:.0f}"))
    parts.append(_card("Time spread", f"±{summary['time_sd']:.1f} s",
                       "one standard deviation"))
    parts.append(_card("Score, typical", f"{summary['score_p50']:.1f}",
                       f"mean {summary['score_mean']:.1f}"))
    parts.append(_card("Score, bad day", f"{summary['score_p10']:.1f}",
                       "10th percentile — what you rank on"))
    parts.append(_card("GM, bad day", f"{summary['gm_p10']:.1f} in",
                       "10th percentile"))
    parts.append(_card("Righting margin, bad day",
                       f"{summary['gz_margin_p10']:.2f}×",
                       "peak arm over the crew's demand",
                       "bad" if summary["gz_margin_p10"] < 1.0 else "ok"))
    parts.append(_card("Tape, bad day", f"{summary['tape_spare_p10']:+.0%}",
                       "spare on one roll",
                       "bad" if summary["tape_spare_p10"] < 0 else "ok"))
    parts.append("</div>")

    parts.append(
        '<div class="note"><b>What these numbers are.</b> Every trial draws '
        'one plausible built boat, one plausible race day and one plausible '
        'set of true material properties, then runs the ordinary '
        'deterministic model on it. Nothing here randomises a result — the '
        'physics is unchanged and only the inputs move, which is why the '
        'spread means something. <b>What they are not:</b> '
        f'{pf:.0%} is not a {pf:.0%} chance of winning. It is the fraction '
        'of boats-you-might-build on days-you-might-race that pass every '
        'hard constraint, given the spreads written down in '
        '<code>uncertainty.py</code>. Those spreads are considered '
        'judgements, not measured distributions. Compare designs with these '
        'numbers; do not quote them as predictions.</div>')

    parts.append("<h2>The spread</h2>")
    parts.append(f'<img src="{chart_mc_distributions(df)}" alt="distributions">')
    img = chart_time_cdf(df)
    if img:
        parts.append(f'<img src="{img}" alt="time cdf">')

    parts.append("<h2>What goes wrong</h2>")
    img = chart_mc_failures(df)
    if img:
        parts.append(f'<img src="{img}" alt="failure modes">')
        parts.append(
            '<div class="note">A constraint that never appears here is not '
            'binding for this hull — stop spending design effort on it. The '
            'one at the top is the only thing standing between this design '
            'and a higher pass rate.</div>')
    else:
        parts.append("<p>No trial failed any constraint.</p>")

    parts.append("<h2>What is driving the spread</h2>")
    for target, label in (("time_s", "crossing time"),
                          ("expected_score", "expected score")):
        sens = sensitivity(df, target)
        img = chart_tornado(sens, label)
        if img:
            parts.append(f'<img src="{img}" alt="tornado {target}">')
    parts.append(
        '<div class="note"><b>This is a shopping list, not a warning.</b> '
        'A long bar against something measurable means measuring it collapses '
        'that much of the spread. In practice two measurements dominate: '
        'timing one paddle over a known distance, which pins the wave-making '
        'term, and twisting a taped test tube, which pins the board shear '
        'modulus. Everything below those is noise you can live with.</div>')

    if by_source:
        parts.append("<h2>Where the uncertainty comes from</h2>")
        img = chart_variance_sources(by_source)
        if img:
            parts.append(f'<img src="{img}" alt="variance sources">')
            rows = []
            for k in ("build", "race", "model", "all"):
                if k in by_source:
                    d2 = by_source[k]
                    rows.append([
                        {"build": "How you build it",
                         "race": "The day you race",
                         "model": "What we do not know",
                         "all": "Everything together"}[k],
                        f"{d2.time_s.std():.1f} s",
                        f"{d2.expected_score.std():.1f}",
                        f"{(d2.feasible == True).mean():.0%}",  # noqa: E712
                    ])
            parts.append(_table(rows, ["Source", "Time spread (sd)",
                                       "Score spread (sd)", "Builds that pass"]))
            parts.append(
                '<div class="note">Three separate runs with one source '
                'switched on at a time. They matter because the remedies '
                'differ: build spread is fixed with jigs and care, race-day '
                'spread with practice, and model spread only with '
                'measurements. Whichever violin is widest is where your next '
                'hour should go.</div>')

            # If each source alone is survivable but the combination is not,
            # say so plainly -- it is the least intuitive thing on the page.
            singles = [k for k in ("build", "race", "model") if k in by_source]
            if "all" in by_source and len(singles) == 3:
                indiv = [float((by_source[k].feasible == True).mean())  # noqa: E712
                         for k in singles]
                combined = float((by_source["all"].feasible == True).mean())  # noqa: E712
                product = indiv[0] * indiv[1] * indiv[2]
                if combined < product - 0.04:
                    parts.append(
                        f'<div class="note warn"><b>The failures come from '
                        f'coincidences, not from any one cause.</b> Taken one '
                        f'at a time these sources pass '
                        f'{indiv[0]:.0%}, {indiv[1]:.0%} and {indiv[2]:.0%} of '
                        f'builds — if they were independent you would expect '
                        f'{product:.0%} together. The real figure is '
                        f'{combined:.0%}. What actually sinks a trial is a '
                        f'slightly crooked hull AND a bad stroke AND softer '
                        f'board than assumed, all at once, each of which '
                        f'would have been survivable alone. That is why '
                        f'margins cannot be reasoned about one at a time, and '
                        f'it is the argument for keeping a little more of each '
                        f'than any single calculation says you need.</div>')

    if comparison is not None and len(comparison) >= 3:
        parts.append("<h2>Robust ranking</h2>")
        img = chart_robust_vs_nominal(comparison)
        if img:
            parts.append(f'<img src="{img}" alt="robust vs nominal">')
        cols = [("name", "Design"), ("length", "L"), ("bottom_width", "B.bot"),
                ("beam", "Beam"), ("side_height", "Depth"),
                ("n_frames", "Frames"), ("free_water_lanes", "Lanes"),
                ("p_feasible", "Pass"), ("time_p50", "Time p50"),
                ("time_p90", "Time p90"), ("score_mean", "Score mean"),
                ("robust_score", "Bad-day score")]
        cols = [(c, h) for c, h in cols if c in comparison.columns]
        rows = []
        for _, r in comparison.head(25).iterrows():
            row = []
            for c, _h in cols:
                v = r[c]
                if c == "p_feasible":
                    row.append(f"{v:.0%}")
                elif isinstance(v, float):
                    row.append(f"{v:.3g}")
                else:
                    row.append(str(v))
            rows.append(row)
        parts.append(_table(rows, [h for _, h in cols], numeric_from=1))
        parts.append(
            '<div class="note"><b>Ranked on the bad-day score</b>, the 10th '
            'percentile across trials — a tenth of the boats you might build '
            'and days you might race come out at or below it. Ranking on the '
            'average instead would quietly favour designs that are '
            'excellent when everything goes right, which is not the brief.'
            '</div>')

    return "\n".join(parts)
