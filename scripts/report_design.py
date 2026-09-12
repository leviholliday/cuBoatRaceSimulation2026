#!/usr/bin/env python3
"""
Full report on one hull: drawings, stability curve, drag breakdown, cut list.

    python scripts/report_design.py                              # the baseline
    python scripts/report_design.py --from-sweep out/sweep.csv --rank 1
    python scripts/report_design.py --set length=96 beam=36 n_frames=6
    python scripts/report_design.py --compare out/sweep.csv --rank 1,2,3

Writes out/design_<name>.html and prints the headline numbers.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from hullsim import report
from hullsim.design import BASELINE, NAMED, HullDesign
from hullsim.evaluate import evaluate
from hullsim.params import NOMINAL, Params

def design_from_row(row: pd.Series) -> HullDesign:
    """Rebuild the exact HullDesign a sweep row came from.

    The sweep writes every design field, so this is a true round trip rather
    than a partial one padded out with baseline defaults. That matters: a
    report that quietly re-evaluated a slightly different boat from the one
    that scored would be worse than no report at all. There is a test for it.
    """
    return HullDesign.from_row(row.to_dict())


def parse_set(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        k = k.strip()
        if k not in HullDesign.__dataclass_fields__:
            raise SystemExit(f"unknown design field {k!r}")
        cur = getattr(BASELINE, k)
        try:
            val = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            val = v
        if isinstance(cur, bool):
            val = bool(val)
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(val)
        elif isinstance(cur, float):
            val = float(val)
        elif isinstance(cur, tuple) and not isinstance(val, tuple):
            val = tuple(val) if isinstance(val, (list, tuple)) else (val,)
        out[k] = val
    return out


def headline(r: dict) -> None:
    v = "FEASIBLE" if r["feasible"] else "FAILS"
    print(f"\n  {r['name'] or 'design'}: {v}")
    print(f"    {r['length']:.0f} x {r['bottom_width']:.0f} bottom x "
          f"{r['beam']:.0f} beam x {r['side_height']:.0f} deep")
    print(f"    draft {r['draft_in']:.2f} in, freeboard {r['freeboard_in']:.2f} in, "
          f"GM {r['gm_in']:+.2f} in, gunwale under at {r['downflood_deg']:.0f} deg")
    print(f"    {r['time_s']:.1f} s crossing at {r['terminal_v_ms']:.2f} m/s "
          f"(Fr {r['froude']:.2f})")
    print(f"    hull {r['hull_weight_lb']:.1f} lb, cardboard "
          f"{r['board_spare_frac']:+.0%} spare, tape {r['tape_spare_frac']:+.0%} spare")
    print(f"    twist {r['twist_deg']:.1f} deg, girder {r['girder_utilisation']:.0%}, "
          f"chine tape {r['chine_tape_util']:.0%}, warp {r['panel_warp']:.3f}")
    print(f"    P(finish) {r['p_finish']:.0%}, expected score "
          f"{r['expected_score']:.1f}")
    if not r["feasible"]:
        for x in r["violations"].split("; "):
            print(f"      ! {x}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-sweep", type=str, default=None)
    ap.add_argument("--rank", type=str, default="1",
                    help="1-based rank among FEASIBLE designs; comma-separated to compare")
    ap.add_argument("--name", type=str, default=None,
                    help="pick a design by its name column instead of rank")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--chop", type=float, default=None)
    ap.add_argument("--residuary-scale", type=float, default=1.0)
    ap.add_argument("--depth", type=float, default=0.0)
    ap.add_argument("--named", type=str, default=None, choices=sorted(NAMED),
                    help="report on a named reference design")
    ap.add_argument("--material", type=str, default=None,
                    help="board preset; see scripts/identify_material.py. "
                         "This changes the physics, not just a number: "
                         "corrugated and solid board fail in different ways.")
    ap.add_argument("--out", type=str, default="out")
    args = ap.parse_args()

    designs: list[HullDesign] = []
    if args.named:
        designs = [NAMED[args.named]]
    elif args.from_sweep:
        df = pd.read_csv(args.from_sweep)
        if args.name:
            rows = df[df.name.astype(str) == args.name]
            if not len(rows):
                raise SystemExit(f"no design named {args.name!r}")
            designs = [design_from_row(rows.iloc[0])]
        else:
            ok = df[df.feasible] if "feasible" in df and df.feasible.any() else df
            for rk in (int(s) for s in args.rank.split(",")):
                if rk < 1 or rk > len(ok):
                    raise SystemExit(f"rank {rk} outside 1..{len(ok)}")
                designs.append(design_from_row(ok.iloc[rk - 1]))
    else:
        designs = [BASELINE]

    if args.set:
        kw = parse_set(args.set)
        designs = [d.with_(**kw, name=d.name or "custom") for d in designs]
    if args.chop is not None:
        designs = [d.with_(chop_height=args.chop) for d in designs]

    # The team confirmed the stock is paperboard, not corrugated, so that
    # is the default now rather than NOMINAL (which stays corrugated only
    # so old tests and --material corrugated_c_flute keep working).
    params = Params.of(args.material) if args.material else Params.of("paperboard_unknown")
    print(f"  board: {params.material_name} "
          f"({params.board_caliper_in * 25.4:.2f} mm, "
          f"{'corrugated' if params.is_corrugated else 'solid'})")

    outdir = Path(args.out)
    sections, results = [], []
    for i, d in enumerate(designs):
        r = evaluate(d, detail=True, residuary_scale=args.residuary_scale,
                     water_depth_ft=args.depth, p=params)
        results.append(r)
        headline(r)
        head = (f"<h2 style='border:none;padding-top:0'>"
                f"{d.name or 'design'}</h2>" if len(designs) > 1 else "")
        sections.append((d.name or f"d{i}", head + report.single_design_html(r)))

    nm = designs[0].name or "design"
    path = report.write_report(
        outdir / f"design_{nm}.html",
        title=("Hull comparison" if len(designs) > 1
               else f"Hull report · {nm}"),
        subtitle=(f"{len(designs)} designs side by side" if len(designs) > 1
                  else f"{designs[0].length:.0f} × {designs[0].bottom_width:.0f} "
                       f"bottom × {designs[0].beam:.0f} beam × "
                       f"{designs[0].side_height:.0f} deep, chop "
                       f"{designs[0].chop_height:.1f} in"),
        sections=sections,
    )
    print(f"\n  report {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
