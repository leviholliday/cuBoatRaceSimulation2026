#!/usr/bin/env python3
"""
Run a design sweep and write the dashboard.

    python scripts/run_sweep.py                      # explore, 5000 designs
    python scripts/run_sweep.py -n 20000             # explore harder
    python scripts/run_sweep.py --grid beam,length   # grid two axes
    python scripts/run_sweep.py --refine             # hill-climb the winner
    python scripts/run_sweep.py --chop 2             # rougher water

Writes out/sweep.csv (every design, every column) and out/sweep.html
(the readable version).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from hullsim import report, sweep
from hullsim.design import NAMED, HullDesign
from hullsim.params import NOMINAL, Params
from hullsim.scoring import Limits, Rubric


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--n-designs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=str, default=None,
                    help="comma-separated parameter names to grid instead of sample")
    ap.add_argument("--grid-points", type=int, default=9)
    ap.add_argument("--refine", action="store_true",
                    help="hill-climb from the best design found")
    ap.add_argument("--refine-steps", type=int, default=30)
    ap.add_argument("--chop", type=float, default=None,
                    help="significant chop height, inches")
    ap.add_argument("--crew", type=str, default=None,
                    help="paddler weights, e.g. 117,180")
    ap.add_argument("--power", type=str, default=None,
                    help="paddler shaft power in watts, e.g. 170,150")
    ap.add_argument("--depth", type=float, default=0.0,
                    help="water depth in feet; 0 means deep water")
    ap.add_argument("--residuary-scale", type=float, default=1.0,
                    help="calibration factor from scripts/calibrate.py")
    ap.add_argument("--material", type=str, default=None,
                    help="board preset; see scripts/identify_material.py. "
                         "This changes the physics, not just a number: "
                         "corrugated and solid board fail in different ways.")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--base", type=str, default="baseline",
                    choices=sorted(NAMED),
                    help="design whose unswept fields are held fixed. Grids "
                         "pin everything they do not vary, so centring a grid "
                         "on a hull that already passes is usually what you "
                         "want.")
    ap.add_argument("--base-from", type=str, default=None,
                    metavar="CSV", help="instead of a named base, use the "
                                        "top feasible design from a sweep CSV")
    ap.add_argument("--out", type=str, default="out")
    ap.add_argument("--from-csv", type=str, default=None,
                    help="rebuild the dashboard from an existing sweep.csv "
                         "without re-evaluating anything")
    args = ap.parse_args()

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        outdir = Path(args.out)
        html_path = report.write_report(
            outdir / "sweep.html",
            title="Cardboard canoe design sweep",
            subtitle=(f"{len(df):,} designs · {int(df.feasible.sum()):,} "
                      f"feasible · rebuilt from {args.from_csv}"),
            sections=[("sweep", report.sweep_html(df))],
        )
        print(f"  dashboard {html_path}")
        return 0

    base = NAMED[args.base]
    if args.base_from:
        prev = pd.read_csv(args.base_from)
        pick = prev[prev.feasible] if prev.feasible.any() else prev
        if not len(pick):
            print(f"no designs in {args.base_from}")
            return 2
        base = HullDesign.from_row(pick.iloc[0].to_dict()).with_(name="base")
        print(f"base: top design from {args.base_from} "
              f"({base.length:.0f} x {base.bottom_width:.0f} x "
              f"{base.beam:.0f} x {base.side_height:.0f})")
    if args.chop is not None:
        base = base.with_(chop_height=args.chop)
    if args.crew:
        w = tuple(float(v) for v in args.crew.split(","))
        base = base.with_(crew_weights=w, crew_kg=(17.0,) * len(w),
                          crew_x_frac=tuple(np.linspace(0.30, 0.72, len(w))),
                          crew_power_w=(160.0,) * len(w))
    if args.power:
        base = base.with_(crew_power_w=tuple(float(v) for v in args.power.split(",")))

    ranges = sweep.default_ranges()
    limits, rubric = Limits(), Rubric()
    # The team confirmed the stock is paperboard, not corrugated, so that
    # is the default now rather than NOMINAL (which stays corrugated only
    # so old tests and --material corrugated_c_flute keep working).
    params = Params.of(args.material) if args.material else Params.of("paperboard_unknown")
    print(f"board: {params.material_name} "
          f"({params.board_caliper_in * 25.4:.2f} mm, "
          f"{params.board_areal_lb_ft2:.3f} lb/ft2, "
          f"{'corrugated' if params.is_corrugated else 'solid'})")
    run_kw = dict(limits=limits, rubric=rubric, water_depth_ft=args.depth,
                  residuary_scale=args.residuary_scale, params=params,
                  processes=args.processes)

    if args.grid:
        names = [s.strip() for s in args.grid.split(",") if s.strip()]
        axes = {}
        for nm in names:
            if nm not in ranges:
                print(f"unknown parameter {nm!r}; known: {', '.join(sorted(ranges))}")
                return 2
            spec = ranges[nm]
            if isinstance(spec, tuple) and len(spec) == 2 and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in spec):
                vals = np.linspace(spec[0], spec[1], args.grid_points)
                if isinstance(spec[0], int) and isinstance(spec[1], int):
                    vals = sorted(set(int(round(v)) for v in vals))
                axes[nm] = list(vals)
            else:
                axes[nm] = list(spec)
        designs = sweep.grid(base, axes, filt=sweep.sane)
        label = f"grid over {', '.join(names)}"
    else:
        designs = sweep.latin_hypercube(base, ranges, args.n_designs,
                                        seed=args.seed, filt=sweep.sane)
        label = f"Latin-hypercube sample, {args.n_designs:,} requested"

    print(f"{label}: {len(designs):,} designs pass the sanity filter")
    df = sweep.run(designs, **run_kw)

    if args.refine and df.feasible.any():
        print("refining the leader by hill climbing...")
        top_name = df[df.feasible].iloc[0]["name"]
        start = next(d for d in designs if d.name == top_name)
        best, hist = sweep.hill_climb(start, ranges, steps=args.refine_steps,
                                      **run_kw)
        df2 = sweep.run([best.with_(name="refined")], progress=False, **run_kw)
        df = pd.concat([df2, df], ignore_index=True)
        df = df.sort_values("ranking_score", ascending=False).reset_index(drop=True)
        print(f"  refined score {df2.iloc[0]['ranking_score']:.1f}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    csv = outdir / "sweep.csv"
    df.drop(columns=[c for c in ("detail",) if c in df], errors="ignore").to_csv(
        csv, index=False)

    sub = (f"{len(df):,} designs · {int(df.feasible.sum()):,} feasible · "
           f"{label} · chop {base.chop_height:.1f} in · crew "
           f"{'+'.join(f'{w:.0f}' for w in base.crew_weights)} lb")
    html_path = report.write_report(
        outdir / "sweep.html",
        title="Cardboard canoe design sweep",
        subtitle=sub,
        sections=[("sweep", report.sweep_html(df))],
    )

    print(f"\n  table     {csv}")
    print(f"  dashboard {html_path}")

    if df.feasible.any():
        best = df[df.feasible].iloc[0]
        print(f"\n  best: {best['length']:.0f} x {best['bottom_width']:.0f} bottom "
              f"x {best['beam']:.0f} beam x {best['side_height']:.0f} deep, "
              f"{best['n_frames']:.0f} frames, {best['free_water_lanes']:.0f} lanes")
        print(f"        {best['time_s']:.1f} s, GM {best['gm_in']:+.1f} in, "
              f"P(finish) {best['p_finish']:.0%}, score {best['expected_score']:.1f}")
        print("\n  Now look at one design in detail:")
        print(f"    python scripts/report_design.py --from-sweep {csv} --rank 1")
    else:
        print("\n  Nothing feasible. Open the dashboard and read the "
              "'what stops a design' chart before widening the ranges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
