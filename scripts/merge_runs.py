#!/usr/bin/env python3
"""
Merge overnight runs from several machines into one combined ranking.

    python scripts/merge_runs.py \\
        --run laptop=path/to/laptop_out/mc \\
        --run pi=path/to/pi_out/mc \\
        --run friend=path/to/friend_out/mc \\
        --out out/merged

WHY THIS EXISTS RATHER THAN JUST DUMPING FOUR FOLDERS TOGETHER

Every machine names its own designs r000000, r000001, ... starting from zero
-- the index has no machine identity baked into it. Two machines running
different seeds (which they should, to actually explore different hulls
rather than duplicate each other's work) will each produce a design called
"r000042" that is a COMPLETELY DIFFERENT HULL. Comparing or averaging rows
by that name alone would silently mix up unrelated designs. Every design
here gets tagged with the machine it came from before anything is combined,
which is the whole point of this script rather than a plain `cat` of the
CSVs.

WHAT COMES OUT

  combined_ranking.csv   every design any machine robustness-tested, one
                         table, sorted by the same bad-day score each
                         machine already ranks on -- so the four runs are
                         genuinely comparable, not just concatenated.
  combined.html          the readable version, plus a note on whether the
                         four machines actually agree with each other.
  winner_design.csv      the single best design across all runs, with every
                         HullDesign column intact, ready to feed straight
                         back into run_montecarlo.py for one large
                         confirmatory pass:

    python scripts/run_montecarlo.py --from-sweep out/merged/winner_design.csv \\
        --top 1 -n 20000 --material paperboard_unknown

WHAT "AGREEMENT" MEANS HERE, AND WHY IT MATTERS MORE THAN THE WINNER ITSELF

Four independent runs landing on four different answers is not a failure of
the tool -- it is the most useful thing four machines can tell you that one
machine cannot. It means the ranking is sensitive to exactly which random
designs happened to get sampled, and the "winner" from any single run should
be trusted less than if all four agreed. This script says which situation
you are in rather than just handing back a number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from hullsim import report
from hullsim.design import HullDesign


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--run expects tag=path, got {spec!r}")
    tag, path = spec.split("=", 1)
    return tag.strip(), Path(path.strip())


def load_one(tag: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """A machine's robustness summary, and its full sweep table if present."""
    summary_path = path / "robust_summary.csv"
    if not summary_path.exists():
        # A single-design overnight run in progress, or one that hasn't
        # reached the robustness stage yet, still has this file -- it is
        # written incrementally, one row per design tested so far.
        raise SystemExit(f"[{tag}] no robust_summary.csv in {path} -- "
                         f"has this machine's run produced any results yet?")
    summary = pd.read_csv(summary_path)
    summary["source"] = tag
    summary["orig_name"] = summary["name"].astype(str)
    summary["name"] = tag + "_" + summary["orig_name"]

    sweep_path = path / "stage1_sweep.csv"
    full = None
    if sweep_path.exists():
        full = pd.read_csv(sweep_path)
        full["source"] = tag
        full["orig_name"] = full["name"].astype(str)
    return summary, full


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="TAG=PATH",
                    help="one machine's out/mc directory, tagged so its "
                         "designs never collide with another machine's. "
                         "Repeat for each machine.")
    ap.add_argument("--out", type=str, default="out/merged")
    args = ap.parse_args()

    runs = [parse_run(s) for s in args.run]
    if len(runs) < 2:
        print("only one --run given -- nothing to merge, just read that "
              "machine's own out/mc/overnight.html directly.")
        return 1

    summaries, fulls = [], {}
    for tag, path in runs:
        s, f = load_one(tag, path)
        summaries.append(s)
        if f is not None:
            fulls[tag] = f
        print(f"  {tag:12} {len(s):4} designs robustness-tested"
              + ("" if f is not None else "  (no stage1_sweep.csv -- "
                                          "winner_design.csv may be incomplete "
                                          "if the overall winner is from here)"))

    combined = pd.concat(summaries, ignore_index=True)
    combined = combined.sort_values("robust_score", ascending=False).reset_index(drop=True)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(outdir / "combined_ranking.csv", index=False)

    # -- how much do the machines agree? ------------------------------------
    per_machine_best = (combined.sort_values("robust_score", ascending=False)
                        .groupby("source").first())
    spread = float(per_machine_best["robust_score"].max()
                   - per_machine_best["robust_score"].min())
    agree = spread < 3.0            # a few points is noise; a lot is real disagreement

    print(f"\n  each machine's own best design:")
    for src, row in per_machine_best.sort_values("robust_score", ascending=False).iterrows():
        print(f"    {src:12} {row['orig_name']:10} bad-day score {row['robust_score']:6.1f}"
              f"   {row['length']:.0f}x{row['bottom_width']:.0f}x{row['beam']:.0f}x{row['side_height']:.0f}")
    print(f"\n  spread across machines' best picks: {spread:.1f} points"
          f"  ->  {'the machines AGREE' if agree else 'the machines DISAGREE -- see the note below'}")

    # -- the overall winner, with a full-column row if we can find one ------
    top = combined.iloc[0]
    winner_tag, winner_name = top["source"], top["orig_name"]
    print(f"\n  overall winner: [{winner_tag}] {winner_name}"
          f"  bad-day score {top['robust_score']:.1f}"
          f"  ({top['p_feasible']:.0%} of builds pass, median {top['time_p50']:.0f}s)")

    winner_row = None
    if winner_tag in fulls:
        match = fulls[winner_tag][fulls[winner_tag]["orig_name"] == winner_name]
        if len(match):
            winner_row = match.iloc[0]
    if winner_row is not None:
        design = HullDesign.from_row(winner_row.to_dict())
        pd.DataFrame([design.to_row()]).to_csv(outdir / "winner_design.csv", index=False)
        print(f"\n  winner_design.csv written -- feed it back in for a big "
              f"confirmatory run:")
        print(f"    python scripts/run_montecarlo.py --from-sweep "
              f"{outdir / 'winner_design.csv'} --top 1 -n 20000 "
              f"--material paperboard_unknown")
    else:
        print(f"\n  could not recover the full design for the winner -- "
              f"[{winner_tag}]'s stage1_sweep.csv was not found alongside "
              f"its robust_summary.csv. Copy that file over from that "
              f"machine and re-run this script to get winner_design.csv.")

    # -- a short HTML version -------------------------------------------------
    cols = [("source", "Machine"), ("orig_name", "Design"),
            ("robust_score", "Bad-day score"), ("p_feasible", "Builds pass"),
            ("time_p50", "Time p50"), ("length", "L"), ("bottom_width", "B.bot"),
            ("beam", "Beam"), ("side_height", "Depth")]
    rows = []
    for _, r in combined.head(40).iterrows():
        rows.append([r["source"], r["orig_name"], f"{r['robust_score']:.1f}",
                    f"{r['p_feasible']:.0%}", f"{r['time_p50']:.1f}",
                    f"{r['length']:.0f}", f"{r['bottom_width']:.0f}",
                    f"{r['beam']:.0f}", f"{r['side_height']:.0f}"])
    body = [
        '<div class="note"><b>What this combines.</b> Each machine ran its '
        'own wide search and its own robustness stress-test independently -- '
        'different random seeds mean different hulls got tried, so this is '
        'not four copies of the same work, it is four separate samples of '
        'the design space. Every design keeps the machine it came from so '
        'nothing gets mixed up.</div>',
    ]
    if not agree:
        body.append(
            f'<div class="note warn"><b>The machines disagree by {spread:.1f} '
            f'points.</b> That is a real signal, not noise: it means which '
            f'hull looks best is still sensitive to exactly which random '
            f'designs got sampled. Trust the combined ranking over any one '
            f'machine\'s answer, and consider running the winner through a '
            f'large confirmatory pass (see below) before committing to it.'
            f'</div>')
    else:
        body.append(
            f'<div class="note"><b>The machines agree</b> (best picks within '
            f'{spread:.1f} points of each other) -- a good sign the ranking '
            f'is not an artefact of one machine\'s random draws.</div>')
    body.append(report._table(rows, [h for _, h in cols], numeric_from=2))

    path = report.write_report(
        outdir / "combined.html",
        title="Merged robustness ranking",
        subtitle=f"{len(runs)} machines, {len(combined)} designs tested total",
        sections=[("merged", "\n".join(body))],
    )
    print(f"\n  combined ranking  {outdir / 'combined_ranking.csv'}")
    print(f"  dashboard         {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
