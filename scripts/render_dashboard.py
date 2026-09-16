#!/usr/bin/env python3
"""
Turn a merge_runs.py output into the public leaderboard page.

    python scripts/merge_runs.py --run pi=... --run mac=... --out out/merged
    python scripts/render_dashboard.py --merged out/merged --out public/leaderboard

Reads combined_ranking.csv and winner_design.csv from --merged (exactly what
merge_runs.py writes) and renders a polished static page plus copies of
those two CSVs, so "everything" stays one click away from the highlights.
This is the one piece of the site that is not live -- it is a snapshot from
whenever this was last run, on purpose: rebuilding it needs every machine's
raw upload downloaded and merged first, which only makes sense to do at a
handful of checkpoints, not on every page load.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FIELD_GROUPS = [
    ("Hull dimensions", [
        ("length", "Length", "in"), ("bottom_width", "Bottom width", "in"),
        ("beam", "Beam", "in"), ("side_height", "Side height", "in"),
        ("bow_rocker", "Bow rocker", "in"), ("chine_frac", "Chine fraction", ""),
        ("n_frames", "Frames", ""), ("skin_layers", "Skin layers", ""),
        ("chine_tape_layers", "Chine tape layers", ""), ("decked_ends", "Decked ends", ""),
    ]),
    ("Race performance", [
        ("time_p10", "Time, fast day (p10)", "s"), ("time_p50", "Time, typical day (p50)", "s"),
        ("time_p90", "Time, slow day (p90)", "s"), ("p_feasible", "Builds that pass every limit", "%"),
        ("score_p10", "Bad-day score (p10)", ""), ("score_mean", "Average score", ""),
        ("score_p90", "Good-day score (p90)", ""), ("p_finish_mean", "Finishes the course", "%"),
    ]),
    ("Stability margins", [
        ("gm_p10", "Metacentric height, bad day (p10)", "in"),
        ("gm_p50", "Metacentric height, typical (p50)", "in"),
        ("freeboard_p10", "Freeboard, bad day (p10)", "in"),
        ("gz_margin_p10", "Righting-arm margin, bad day (p10)", ""),
        ("swamp_reserve_p10", "Swamping reserve, bad day (p10)", "gal"),
    ]),
    ("Structural margins", [
        ("warp_p90", "Panel warp, bad day (p90)", ""),
        ("twist_p90", "Hull twist, bad day (p90)", "deg"),
        ("tape_spare_p10", "Tape spare, bad day (p10)", "%"),
        ("board_spare_p10", "Board spare, bad day (p10)", "%"),
    ]),
]

FMT_1 = {"in", "deg", "s", "gal"}


def fmt(value, unit) -> str:
    if pd.isna(value):
        return "--"
    if unit == "%":
        return f"{value:.0%}"
    if isinstance(value, (int,)) or (isinstance(value, float) and value == int(value) and unit == ""):
        return f"{value:.0f}"
    if unit in FMT_1:
        return f"{value:.1f} {unit}"
    if unit == "":
        return f"{value:.3g}"
    return f"{value:.2f} {unit}"


def render(merged: Path, out: Path) -> None:
    ranking = pd.read_csv(merged / "combined_ranking.csv")
    winner_design = None
    winner_path = merged / "winner_design.csv"
    if winner_path.exists():
        winner_design = pd.read_csv(winner_path).iloc[0]

    ranking = ranking.sort_values("robust_score", ascending=False).reset_index(drop=True)
    top = ranking.iloc[0]

    per_machine = (ranking.sort_values("robust_score", ascending=False)
                   .groupby("source").first().sort_values("robust_score", ascending=False))
    designs_per_machine = ranking.groupby("source").size()
    spread = float(per_machine["robust_score"].max() - per_machine["robust_score"].min())
    agree = spread < 3.0

    def stat_rows(row, group):
        cells = []
        for key, label, unit in group:
            if key not in row or pd.isna(row[key]):
                continue
            value = row[key]
            cells.append(f'<div class="stat"><span class="k">{label}</span>'
                        f'<span class="v">{fmt(value, unit)}</span></div>')
        return "\n".join(cells)

    winner_groups_html = "\n".join(
        f'<div class="group"><h3>{title}</h3><div class="stats">{stat_rows(top, fields)}</div></div>'
        for title, fields in FIELD_GROUPS
    )

    machine_rows = "\n".join(
        f'<tr class="{"is-winner" if src == top["source"] else ""}">'
        f'<td><span class="tag">{src}</span></td>'
        f'<td class="num">{designs_per_machine.get(src, 0):,}</td>'
        f'<td class="num">{row["robust_score"]:.1f}</td>'
        f'<td>{row["orig_name"]}</td>'
        f'<td class="num">{row["length"]:.0f}&times;{row["bottom_width"]:.0f}&times;'
        f'{row["beam"]:.0f}&times;{row["side_height"]:.0f}</td></tr>'
        for src, row in per_machine.iterrows()
    )

    top40 = ranking.head(40)
    ranking_rows = "\n".join(
        f'<tr class="{"is-winner" if i == 0 else ""}"><td>{i + 1}</td>'
        f'<td><span class="tag">{r.source}</span></td><td>{r.orig_name}</td>'
        f'<td class="num">{r.robust_score:.1f}</td><td class="num">{r.p_feasible:.0%}</td>'
        f'<td class="num">{r.time_p50:.1f}s</td>'
        f'<td class="num">{r.length:.0f}&times;{r.bottom_width:.0f}&times;'
        f'{r.beam:.0f}&times;{r.side_height:.0f}</td></tr>'
        for i, r in enumerate(top40.itertuples())
    )

    agree_note = (
        f'<div class="callout good"><b>The {len(per_machine)} machines agree.</b> '
        f'Their best picks land within {spread:.1f} points of each other -- a good sign '
        f'this ranking is not an artefact of which random designs one machine happened to sample.</div>'
        if agree else
        f'<div class="callout warn"><b>The {len(per_machine)} machines disagree by {spread:.1f} points.</b> '
        f'That is a real signal, not noise -- the ranking is still sensitive to exactly which designs '
        f'got sampled. Trust the combined ranking below over any single machine\'s answer, and consider '
        f'a large confirmatory run on the winner before building it.</div>'
    )

    confirm_cmd = (
        f"python scripts/run_montecarlo.py --from-sweep out/merged/winner_design.csv "
        f"--top 1 -n 20000 --material paperboard_unknown"
    )

    generated = datetime.now(timezone.utc).strftime("%B %-d, %Y at %-I:%M %p UTC")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hull Leaderboard</title>
<meta name="description" content="The best cardboard canoe hull found so far, combined from every machine's overnight search.">
<link rel="icon" type="image/svg+xml" href="../guest/favicon.svg">
<style>
:root{{
  --ink:#1a2332; --ink-soft:#4a5568; --ground:#f6f3ec; --panel:#ffffff;
  --line:#ddd4bf; --line-soft:#eae3d3;
  --accent:#b8501f; --accent-ink:#8a3c15; --accent-soft:#f1e2d3;
  --mono-bg:#1a2332; --mono-text:#e9e4d6; --mono-dim:#8b93a3; --mono-accent:#e8a56a;
  --good:#3f7d5c; --good-soft:#e3efe8; --warn:#a6461f; --warn-soft:#f6e3db;
  --font-display:"Archivo",system-ui,sans-serif;
  --font-body:"Public Sans",system-ui,-apple-system,sans-serif;
  --font-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  color-scheme:light dark;
}}
@media (prefers-color-scheme: dark){{
  :root:not([data-theme="light"]){{
    --ink:#eae4d6; --ink-soft:#a8a396; --ground:#181d27; --panel:#1f2530;
    --line:#333c4a; --line-soft:#2a3140;
    --accent:#e8a56a; --accent-ink:#f2c193; --accent-soft:#332216;
    --mono-bg:#0f131b; --mono-text:#e9e4d6; --mono-dim:#6b7280; --mono-accent:#e8a56a;
    --good:#6bbf94; --good-soft:#1c2b23; --warn:#e08a5c; --warn-soft:#332019;
  }}
}}
:root[data-theme="dark"]{{
  --ink:#eae4d6; --ink-soft:#a8a396; --ground:#181d27; --panel:#1f2530;
  --line:#333c4a; --line-soft:#2a3140;
  --accent:#e8a56a; --accent-ink:#f2c193; --accent-soft:#332216;
  --mono-bg:#0f131b; --mono-text:#e9e4d6; --mono-dim:#6b7280; --mono-accent:#e8a56a;
  --good:#6bbf94; --good-soft:#1c2b23; --warn:#e08a5c; --warn-soft:#332019;
}}
*{{box-sizing:border-box}}
body{{background:var(--ground); color:var(--ink); font-family:var(--font-body); font-size:16px; line-height:1.6; margin:0}}
.wrap{{max-width:960px; margin:0 auto; padding:56px 24px 100px}}
h1{{font-family:var(--font-display); font-weight:800; font-size:clamp(28px,5vw,40px); letter-spacing:-0.01em; margin:0 0 8px}}
.dek{{color:var(--ink-soft); font-size:16px; margin:0 0 4px; max-width:66ch}}
.meta-row{{display:flex; gap:16px; flex-wrap:wrap; margin-top:20px; font-size:13px; color:var(--ink-soft)}}
.meta-row a{{color:var(--accent-ink); text-decoration:none; font-weight:600}}
.meta-row a:hover{{text-decoration:underline}}

.overview{{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:28px 0}}
.ov-card{{background:var(--panel); border:1px solid var(--line-soft); border-radius:12px; padding:16px 18px}}
.ov-card .k{{font-family:var(--font-display); font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:var(--accent-ink); font-weight:700; margin-bottom:4px}}
.ov-card .v{{font-family:var(--font-mono); font-size:22px; font-weight:600; font-variant-numeric:tabular-nums}}

.callout{{background:var(--accent-soft); border-radius:10px; padding:16px 18px; margin:20px 0; font-size:14.5px}}
.callout b{{color:var(--accent-ink)}}
.callout.good{{background:var(--good-soft)}} .callout.good b{{color:var(--good)}}
.callout.warn{{background:var(--warn-soft)}} .callout.warn b{{color:var(--warn)}}

.winner{{background:var(--panel); border:2px solid var(--accent); border-radius:16px; padding:28px 26px; margin:36px 0}}
.winner-head{{display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:4px}}
.winner-badge{{font-family:var(--font-display); font-weight:700; font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#fff9f2; background:var(--accent); padding:4px 10px; border-radius:999px}}
.winner-name{{font-family:var(--font-mono); font-size:15px; color:var(--ink-soft)}}
.winner h2{{font-family:var(--font-display); font-weight:800; font-size:26px; margin:6px 0 18px; letter-spacing:-0.005em}}
.group{{margin-top:22px}}
.group:first-of-type{{margin-top:8px}}
.group h3{{font-family:var(--font-display); font-weight:700; font-size:12px; text-transform:uppercase; letter-spacing:0.07em; color:var(--ink-soft); margin:0 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line-soft)}}
.stats{{display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px 18px}}
.stat{{display:flex; flex-direction:column; gap:2px}}
.stat .k{{font-size:12.5px; color:var(--ink-soft)}}
.stat .v{{font-family:var(--font-mono); font-size:15.5px; font-weight:600; font-variant-numeric:tabular-nums}}

.confirm{{margin-top:22px}}
.cmd{{position:relative; background:var(--mono-bg); color:var(--mono-text); font-family:var(--font-mono); font-size:13px; line-height:1.6; border-radius:10px; padding:14px 18px; white-space:pre-wrap; word-break:break-word; margin:10px 0 0}}

.phase{{margin-top:52px}}
.phase h2{{font-family:var(--font-display); font-weight:700; font-size:20px; margin:0 0 4px; letter-spacing:-0.005em}}
.phase-note{{color:var(--ink-soft); font-size:14px; margin:0 0 16px}}

table{{width:100%; border-collapse:collapse; font-size:13.5px; margin:0}}
th,td{{text-align:left; padding:9px 10px; border-bottom:1px solid var(--line-soft)}}
th{{font-family:var(--font-display); font-weight:700; font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:var(--ink-soft)}}
td.num, th.num{{text-align:right; font-variant-numeric:tabular-nums; font-family:var(--font-mono)}}
tr:last-child td{{border-bottom:none}}
tr.is-winner td{{background:var(--accent-soft); font-weight:600}}
.tablewrap{{overflow-x:auto; background:var(--panel); border:1px solid var(--line-soft); border-radius:12px; padding:4px 16px}}
.tag{{display:inline-block; padding:2px 9px; border-radius:999px; background:var(--line-soft); font-family:var(--font-mono); font-size:12px; font-weight:600}}

.footer{{margin-top:60px; padding-top:20px; border-top:1px solid var(--line); color:var(--ink-soft); font-size:13px}}
.footer a{{color:var(--accent-ink)}}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
</head>
<body>
<div class="wrap">

  <h1>Hull Leaderboard</h1>
  <p class="dek">Every overnight search, combined into one ranking -- the best hull found across all of them, with its full stats.</p>
  <div class="meta-row">
    <span>Generated {generated}</span>
    <a href="../">&larr; Raw uploads &amp; all machines</a>
    <a href="combined_ranking.csv">&darr; Full ranking (CSV)</a>
    {"<a href=\"winner_design.csv\">&darr; Winner's full design (CSV)</a>" if winner_design is not None else ""}
  </div>

  <div class="overview">
    <div class="ov-card"><div class="k">Machines</div><div class="v">{len(per_machine)}</div></div>
    <div class="ov-card"><div class="k">Designs tested</div><div class="v">{len(ranking):,}</div></div>
    <div class="ov-card"><div class="k">Spread across machines</div><div class="v">{spread:.1f} pts</div></div>
    <div class="ov-card"><div class="k">Agreement</div><div class="v" style="color:{'var(--good)' if agree else 'var(--warn)'}">{'Agree' if agree else 'Disagree'}</div></div>
  </div>

  {agree_note}

  <div class="winner">
    <div class="winner-head">
      <span class="winner-badge">Overall winner</span>
      <span class="winner-name">{top['source']}_{top['orig_name']}</span>
    </div>
    <h2>{top['length']:.0f}&Prime; &times; {top['bottom_width']:.0f}&Prime; bottom &times; {top['beam']:.0f}&Prime; beam &times; {top['side_height']:.0f}&Prime; deep &mdash; bad-day score {top['robust_score']:.1f}</h2>
    {winner_groups_html}
    <div class="confirm">
      <div class="callout"><b>Before building this one for real:</b> run a large confirmatory pass to shrink the uncertainty on this specific hull.</div>
      <div class="cmd">{confirm_cmd}</div>
    </div>
  </div>

  <div class="phase">
    <h2>Each machine's best pick</h2>
    <p class="phase-note">Sorted by bad-day score -- the winner above is the top row here.</p>
    <div class="tablewrap">
      <table>
        <thead><tr><th>Machine</th><th class="num">Designs tested</th><th class="num">Bad-day score</th><th>Design</th><th class="num">L&times;B.bot&times;Beam&times;Depth</th></tr></thead>
        <tbody>{machine_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="phase">
    <h2>Top {min(40, len(ranking))} designs, combined</h2>
    <p class="phase-note">Every machine's designs in one ranking, tagged by source so nothing gets mixed up. Full list: <a href="combined_ranking.csv" style="color:var(--accent-ink)">combined_ranking.csv</a>.</p>
    <div class="tablewrap">
      <table>
        <thead><tr><th>#</th><th>Machine</th><th>Design</th><th class="num">Bad-day score</th><th class="num">Builds pass</th><th class="num">Time (p50)</th><th class="num">L&times;B.bot&times;Beam&times;Depth</th></tr></thead>
        <tbody>{ranking_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="footer">
    A snapshot from the last time results were consolidated -- not live. Every machine's raw upload is always available in full at <a href="../">the uploads page</a>. Part of a cardboard canoe hull simulator for a college engineering class's lake race.
  </div>

</div>
</body>
</html>
"""

    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html)
    shutil.copy(merged / "combined_ranking.csv", out / "combined_ranking.csv")
    if winner_path.exists():
        shutil.copy(winner_path, out / "winner_design.csv")
    print(f"  wrote {out / 'index.html'}")
    print(f"  wrote {out / 'combined_ranking.csv'}")
    if winner_path.exists():
        print(f"  wrote {out / 'winner_design.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", default="out/merged",
                    help="a merge_runs.py --out directory")
    ap.add_argument("--out", default="public/leaderboard",
                    help="where to write the site page (published as /leaderboard/)")
    args = ap.parse_args()

    merged = Path(args.merged)
    if not (merged / "combined_ranking.csv").exists():
        print(f"{merged}/combined_ranking.csv not found -- run merge_runs.py first.")
        return 1
    render(merged, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
