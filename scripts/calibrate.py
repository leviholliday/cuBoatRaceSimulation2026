#!/usr/bin/env python3
"""
Turn a stopwatch into a calibrated resistance model.

    python scripts/calibrate.py --time 96 --distance 140
    python scripts/calibrate.py --time 96 --power 320 --set length=96 beam=36

WHY THIS IS THE HIGHEST-VALUE HOUR YOU CAN SPEND ON THE MODEL

Everything in the speed prediction except one thing is computed rather than
assumed. Wetted surface comes from the real geometry. Friction comes from the
ITTC correlation line, which is what every towing tank in the world uses.
Form factor comes from a published correlation. All defensible.

Residuary resistance -- the wave the boat makes -- is the exception. It comes
from a standard-series curve fitted to displacement hulls, and a cardboard
canoe is far bluffer than anything in that series. At the speed this boat
actually reaches, residuary is most of the total drag. So the single most
uncertain term is also the dominant one, and that is the whole reason the
predicted time should be read as a comparison between hulls rather than as a
number of seconds.

One measurement fixes it. Paddle a measured distance flat out from a standing
start, time it, and feed the time in here. What comes back is the scale
factor that makes the model reproduce what actually happened. Pass it to
run_sweep.py with --residuary-scale and every time it prints afterwards is
anchored to your crew in your boat.

HOW TO TAKE THE MEASUREMENT

  1. Any boat you can measure will do -- the race hull, a test hull, a
     borrowed canoe. Measure its length, bottom width, beam and depth, and
     weigh what goes in it.
  2. Mark a distance you know. 140 m matches the course; 50 m is enough.
  3. Standing start, flat out, three runs, take the middle time.
  4. Run this with that time and the same crew power you tell the model to
     assume elsewhere.

A CAVEAT WORTH UNDERSTANDING: this fits residuary scale GIVEN an assumed crew
power. If the power is wrong the scale silently absorbs the error, and you
end up with a model that is right for one speed and wrong everywhere else.
If you can only measure one thing, measure power first -- paddle at a steady
speed you can hold, time it, and use --solve-power to get the power the model
needs to match. Then come back and fit the residuary scale on a sprint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hullsim import constants as C
from hullsim.design import BASELINE
from hullsim.geometry import build_mesh, wetted_surface
from hullsim.hydro import mass_properties, solve_equilibrium
from hullsim.materials import take_off
from hullsim.resistance import drag_table, resistance, simulate_race
from report_design import parse_set


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time", type=float, required=True,
                    help="measured time for the run, seconds")
    ap.add_argument("--distance", type=float, default=C.COURSE_M,
                    help="measured distance, metres (default: the race course)")
    ap.add_argument("--power", type=float, default=None,
                    help="total crew shaft power in watts (default: the design's)")
    ap.add_argument("--solve-power", action="store_true",
                    help="solve for crew power at fixed residuary scale instead")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="hull dimensions of the boat you actually timed")
    ap.add_argument("--residuary-scale", type=float, default=1.0,
                    help="fixed scale, when solving for power instead")
    args = ap.parse_args()

    d = BASELINE
    if args.set:
        d = d.with_(**parse_set(args.set))

    mesh = build_mesh(d)
    mat = take_off(mesh)
    hull_w = d.hull_weight_override if d.hull_weight_override is not None \
        else mat.hull_weight_lb
    mp = mass_properties(mesh, hull_w)
    eq = solve_equilibrium(mesh, mp)
    s_wet = wetted_surface(mesh, eq.waterline_z)

    shaft = args.power if args.power is not None else float(sum(d.crew_power_w))
    max_thrust = 11.0 * d.n_crew

    def drag_fn(scale):
        def f(v):
            return resistance(max(v, 1e-3), wetted_in2=s_wet, lwl_in=eq.lwl,
                              beam_wl_in=eq.beam_wl, draft_in=eq.draft,
                              volume_in3=eq.volume, displacement_lb=mp.weight,
                              cb=eq.cb, cp=eq.cp, chop_in=d.chop_height,
                              residuary_scale=scale)
        return f

    def race_time(scale, power):
        return simulate_race(drag_table(drag_fn(scale)),
                             displacement_lb=mp.weight, shaft_power_w=power,
                             max_thrust_lb=max_thrust,
                             course_m=args.distance).time_s

    print(f"\n  Hull:        {d.length:.0f} x {d.bottom_width:.0f} bottom x "
          f"{d.beam:.0f} beam x {d.side_height:.0f} deep")
    print(f"  All-up:      {mp.weight:.0f} lb  (hull {hull_w:.1f} lb)")
    print(f"  Waterline:   {eq.lwl:.0f} in long, {eq.beam_wl:.1f} in wide, "
          f"draft {eq.draft:.2f} in")
    print(f"  Wetted area: {s_wet / 144:.1f} ft2")
    print(f"  Measured:    {args.distance:.0f} m in {args.time:.1f} s "
          f"({args.distance / args.time:.2f} m/s average)")

    if args.solve_power:
        lo, hi = 5.0, 3000.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if race_time(args.residuary_scale, mid) > args.time:
                lo = mid
            else:
                hi = mid
        power = 0.5 * (lo + hi)
        print(f"\n  Crew shaft power needed to match: {power:.0f} W total, "
              f"{power / d.n_crew:.0f} W each")
        print(f"  (at residuary scale {args.residuary_scale:.2f})")
        print("\n  Sanity check: elite flatwater kayakers put out about 400 W "
              "each at\n  sprint pace. If this comes back above ~350 W each, "
              "the residuary\n  model is overestimating drag, not your crew "
              "being superhuman.")
        print(f"\n  Put it in a design with:  crew_power_w="
              f"{tuple(round(power / d.n_crew) for _ in range(d.n_crew))}")
        return 0

    lo, hi = 0.02, 40.0
    if race_time(hi, shaft) < args.time:
        print("\n  Even the maximum drag the model allows is not enough to be "
              "this slow.\n  Either the crew power assumed is far too high, or "
              "something outside\n  the model is dominating -- wind, a shallow "
              "start, or a hull full of water.")
        return 1
    if race_time(lo, shaft) > args.time:
        print("\n  Even with essentially no wave-making the model cannot be "
              "this fast.\n  The assumed crew power is almost certainly too "
              "low. Re-run with\n  --solve-power to see what power this time "
              "implies.")
        return 1

    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if race_time(mid, shaft) < args.time:
            lo = mid                 # too fast: needs more drag
        else:
            hi = mid
    scale = 0.5 * (lo + hi)

    t_before = race_time(1.0, shaft)
    print(f"\n  Model before calibration: {t_before:.1f} s")
    print(f"  Model after calibration:  {race_time(scale, shaft):.1f} s")
    print(f"\n  RESIDUARY SCALE = {scale:.3f}")
    if scale > 1.3:
        print("  Your hull makes MORE wave than the standard series predicts, "
              "which is\n  what you would expect from something this bluff. "
              "Unsurprising.")
    elif scale < 0.7:
        print("  Your hull makes LESS wave than predicted. A wide flat bottom "
              "starts to\n  generate dynamic lift above roughly Froude 0.5, "
              "which this model does\n  not know about, so that is a plausible "
              "reason rather than an error.")
    else:
        print("  Close to the standard-series prediction. The uncalibrated "
              "model was\n  already about right for this hull.")

    print(f"\n  Use it:\n    python scripts/run_sweep.py "
          f"--residuary-scale {scale:.3f}")
    print(f"    python scripts/report_design.py --residuary-scale {scale:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
