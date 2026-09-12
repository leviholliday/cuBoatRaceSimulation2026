#!/usr/bin/env python3
"""
Work out what your board actually is, from two measurements.

    python scripts/identify_material.py --area 12x12 --weight-g 46 --caliper-mm 1.1

WHY THIS MATTERS MORE THAN ANY OTHER MEASUREMENT

The model was built assuming single-wall corrugated cardboard. If the stock
is actually thin solid paperboard -- which "coated one side", supplied on a
roll, is strong evidence for, because corrugated cannot be rolled without
crushing its flutes -- then several conclusions invert:

    corrugated                      thin solid board
    -----------------------------   -----------------------------
    very stiff in bending           roughly a tenth as stiff
    weak edgewise                   several times stronger edgewise
    strongly direction-dependent    mildly so, and you cannot choose
    crushes under double curvature  bends happily
    panel strength is the worry     TORSION is the worry

With corrugated the hull girder runs near a fifth of its allowable and the
panels are fine. With thin solid board the girder drops to a few percent and
the boat twists two to four times as much. Those call for different boats.

WHAT TO MEASURE, WHICH TAKES TWO MINUTES

  1. Cut a rectangle you can measure exactly. A 12 by 12 inch square is
     ideal; anything over about 6 by 6 is fine.
  2. Weigh it. A kitchen scale in grams is plenty.
  3. Measure the thickness. If you have calipers, use them. If not, stack
     ten sheets, measure the stack with a ruler, and divide by ten.
  4. Look at the cut edge. Do you see a wavy corrugated core sandwiched
     between two flat liners, or is it solid all the way through?

Then run this. It tells you which preset to use, or builds a custom one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hullsim.params import MATERIALS, Params, _solid_board


def parse_area(text: str) -> float:
    """Accept '12x12', '12 x 12', or a plain number of square inches."""
    t = text.lower().replace(" ", "")
    if "x" in t:
        a, b = t.split("x", 1)
        return float(a) * float(b)
    return float(t)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", type=str, required=True,
                    help="the piece you weighed, e.g. 12x12 (inches) or 144")
    ap.add_argument("--weight-g", type=float, required=True,
                    help="its mass in grams")
    ap.add_argument("--caliper-mm", type=float, default=None,
                    help="thickness in mm (stack ten sheets and divide)")
    ap.add_argument("--corrugated", action="store_true",
                    help="set this if the cut edge shows a wavy core")
    args = ap.parse_args()

    area_in2 = parse_area(args.area)
    area_ft2 = area_in2 / 144.0
    area_m2 = area_in2 * 0.00064516
    lb_ft2 = (args.weight_g / 453.592) / area_ft2
    gsm = args.weight_g / area_m2

    print(f"\n  Measured: {area_in2:.0f} in2 weighing {args.weight_g:.1f} g")
    print(f"    areal weight  {gsm:.0f} g/m2   =  {lb_ft2:.3f} lb/ft2")

    if args.caliper_mm:
        t_mm = args.caliper_mm
        density = gsm / (t_mm * 1000.0)
        print(f"    caliper       {t_mm:.2f} mm  =  {t_mm / 25.4:.3f} in")
        print(f"    density       {density:.2f} g/cm3")
        if density > 0.45:
            verdict = ("SOLID board. Paper fibre is about 1.5 g/cm3 and solid "
                       "paperboard packs to 0.6-0.9.")
            corrugated = False
        elif density < 0.25:
            verdict = ("CORRUGATED. Mostly air between two liners, which is "
                       "what gives a density this low.")
            corrugated = True
        else:
            verdict = ("AMBIGUOUS at this density. Look at the cut edge: a "
                       "wavy core means corrugated.")
            corrugated = args.corrugated
        print(f"\n  Verdict: {verdict}")
    else:
        t_mm = None
        corrugated = args.corrugated
        print("\n  No caliper given, so density cannot be checked. Measure the "
              "thickness\n  if you can -- it is what separates the two families.")

    print("\n  Closest presets by areal weight:")
    scored = []
    for name in MATERIALS:
        p = Params.of(name)
        err = abs(p.board_areal_lb_ft2 - lb_ft2) / max(lb_ft2, 1e-9)
        if t_mm:
            err += abs(p.board_caliper_in * 25.4 - t_mm) / max(t_mm, 1e-9)
        if p.is_corrugated != corrugated:
            err += 1.0                      # wrong family is a big penalty
        scored.append((err, name, p))
    scored.sort()
    for err, name, p in scored[:3]:
        print(f"    {name:22} {p.board_caliper_in * 25.4:5.2f} mm  "
              f"{p.board_areal_lb_ft2:.3f} lb/ft2  "
              f"{'corrugated' if p.is_corrugated else 'solid':>10}   "
              f"mismatch {err:.0%}")

    best = scored[0]
    print(f"\n  Use:  --material {best[1]}")

    if t_mm and not corrugated:
        custom = _solid_board(t_mm / 25.4, density_g_cm3=gsm / (t_mm * 1000.0))
        print("\n  Or an exact custom preset for YOUR board. Add this to "
              "MATERIALS in hullsim/params.py:")
        print(f'\n    "ours": {{"material_name": "ours", '
              f'**_solid_board({t_mm / 25.4:.4f}, '
              f'density_g_cm3={gsm / (t_mm * 1000.0):.2f})}},')
        print(f"\n    caliper {custom['board_caliper_in']:.4f} in, "
              f"areal {custom['board_areal_lb_ft2']:.3f} lb/ft2, "
              f"bending {custom['board_d_stiff_lbin']:.2f} lb.in, "
              f"edgewise {custom['board_ect_lb_in']:.0f} lb/in")

    print("\n  Then re-run anything with --material, e.g.")
    print(f"    python scripts/run_sweep.py --material {best[1]} -n 20000")
    print("\n  Two numbers still worth measuring separately: how much strength")
    print("  the board keeps after ten minutes in water, and the bending")
    print("  stiffness of a strip. Both are in data/joint_tests.json.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
