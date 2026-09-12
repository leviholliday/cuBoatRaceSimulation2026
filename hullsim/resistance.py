"""
Resistance and the race time.

Three pieces, in descending order of how much you should trust them:

  1. Frictional resistance -- ITTC-1957 correlation line on the wetted surface
     the geometry module actually computed. This is standard, it is the same
     line every towing tank in the world reduces data with, and there is
     nothing to argue about.

  2. Form factor -- Watanabe's estimate from block coefficient, length/beam and
     beam/draft. A published correlation, applied to a hull bluffer than the
     ships it was fitted to. Treat it as 'about right', not exact.

  3. Residuary (wave-making) resistance -- a documented correlation, not a
     measurement, and the weakest link in this file. See the long note over
     `residuary_over_displacement` for what it is and how to replace it with a
     number of your own.

WHY PIECE 3 MATTERS MORE THAN THE EARLIER HAND ANALYSIS ASSUMED

If you assume the boat travels at 1.2-1.5 m/s then Froude number is ~0.21-0.26,
wave-making is small, and it is fair to say the race is not power-limited. But
that assumption is doing all the work. Two people putting ~200 W of thrust into
a 325 lb boat do not settle at 1.3 m/s -- they accelerate until drag balances
thrust, and for a hull this short that balance lands near Froude 0.4-0.5, right
in the wave-making hump. So the shape of the residuary curve is what actually
sets the predicted time, and its uncertainty is the dominant uncertainty in
every time this model reports.

Which is why `calibrate_residuary` exists. Paddle a measured distance, time it,
and the one free coefficient stops being a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import constants as C
from .params import NOMINAL, Params


# ---------------------------------------------------------------------------
# Viscous resistance
# ---------------------------------------------------------------------------

def friction_coefficient(reynolds: float) -> float:
    """ITTC-1957 model-ship correlation line."""
    re = max(reynolds, 1.0e4)
    return 0.075 / (math.log10(re) - 2.0) ** 2


def form_factor(cb: float, lwl_ft: float, beam_ft: float, draft_ft: float) -> float:
    """Watanabe's form-factor estimate, (1 + k).

        k = -0.095 + 25.6 * Cb / ((L/B)^2 * sqrt(B/T))

    Published for ship hulls. A cardboard canoe is bluffer and shorter than
    anything in that dataset, so the result is clamped to a range that stays
    physically sensible rather than trusted blindly.
    """
    if min(lwl_ft, beam_ft, draft_ft) <= 1e-6:
        return 1.3
    lb = lwl_ft / beam_ft
    bt = beam_ft / draft_ft
    k = -0.095 + 25.6 * cb / (lb ** 2 * math.sqrt(bt))
    return float(np.clip(1.0 + k, 1.10, 2.50))


# ---------------------------------------------------------------------------
# Residuary resistance
# ---------------------------------------------------------------------------

# Reference residuary resistance per pound of displacement, for a displacement
# hull of slenderness L/vol^(1/3) = 6 and prismatic coefficient 0.60. This is
# the classic Taylor-series curve shape: negligible below Froude 0.2, climbing
# steeply through the hump around 0.4-0.5, flattening after.
_FR_REF = np.array([0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                    0.45, 0.50, 0.55, 0.60, 0.70, 0.80])
_RR_REF = np.array([0.0000, 0.0001, 0.0003, 0.0010, 0.0025, 0.0060, 0.0140,
                    0.0300, 0.0520, 0.0750, 0.0950, 0.1100, 0.1300, 0.1450])

SLENDERNESS_REF = 6.0
SLENDERNESS_EXP = 3.0
CP_PENALTY = 4.0


def optimum_cp(froude: float) -> float:
    """Prismatic coefficient that minimises residuary resistance at a given
    Froude number. Standard small-craft design guidance: fine hulls want a low
    Cp at low speed, and the optimum climbs as the bow wave grows."""
    return float(np.clip(0.50 + 0.36 * (froude - 0.20), 0.50, 0.68))


def residuary_over_displacement(froude: float, slenderness: float, cp: float,
                                scale: float = 1.0) -> float:
    """Residuary resistance as a fraction of displacement weight.

    R_r / displacement = g(Fr) * (6 / (L/vol^(1/3)))^3 * (1 + 4*(Cp - Cp_opt)^2)

    THE HONEST DESCRIPTION OF WHAT THIS IS

    It is an engineering correlation with the correct physical structure and
    approximately correct magnitude, not a measurement of this hull:

      - g(Fr) is the shape of the residuary curve for a displacement hull,
        from the Taylor standard series family. The location and steepness of
        the hump are right; the absolute level for a hull as bluff as this one
        is optimistic.
      - The slenderness term is thin-ship scaling. Wave-making falls off
        steeply as a hull gets longer for its volume, and the cube is the
        conventional exponent. This part is well founded.
      - The Cp term penalises being far from the optimum fullness for the
        speed. It is a small correction and a soft one.

    WHAT IT DOES NOT CAPTURE

      - A wide flat bottom starts to generate dynamic lift above roughly
        Froude 0.5, which this model does not know about. Above that the true
        resistance is LOWER than predicted here, so predicted times above
        Froude 0.5 are pessimistic.
      - Flow separation behind a blunt transom. Real, and not in here.
      - Anything about a hull with the fullness of a shoebox, because the
        parent series contained nothing like one.

    `scale` is the single calibration knob. Paddle a measured distance, time
    it, and `calibrate_residuary` will solve for the value that makes the
    model reproduce what actually happened.
    """
    g = float(np.interp(froude, _FR_REF, _RR_REF))
    slend = max(slenderness, 1e-3)
    slender_term = (SLENDERNESS_REF / slend) ** SLENDERNESS_EXP
    cp_term = 1.0 + CP_PENALTY * (cp - optimum_cp(froude)) ** 2
    return scale * g * slender_term * cp_term


def shallow_water_factor(v_ms: float, depth_ft: float, lwl_ft: float) -> float:
    """Resistance multiplier for running in shallow water.

    As the depth Froude number Fr_h = v / sqrt(g*h) approaches 1 the water
    cannot get out of the boat's way, the wave system bunches up, and
    resistance climbs steeply. Below Fr_h of about 0.4 the effect is
    negligible; above 0.7 it is large.

    Returns 1.0 for deep water. Pass depth_ft = 0 or None to disable, which is
    the default -- most of a 140 m crossing is deep enough not to care, but
    the first and last few boat-lengths may not be, and a shallow start is a
    real and commonly missed time loss.
    """
    if not depth_ft or depth_ft <= 0:
        return 1.0
    v_fts = v_ms * 3.28084
    frh = v_fts / math.sqrt(C.G_FT_S2 * depth_ft)
    if frh < 0.4:
        return 1.0
    # Smooth rise through the subcritical range; capped so the solver cannot
    # chase a singularity at Fr_h = 1.
    return float(min(1.0 + 2.2 * (frh - 0.4) ** 2 / max(1.0 - frh, 0.12), 3.0))


def wind_drag(v_ms: float, headwind_mph: float, frontal_ft2: float,
              cd: float = 1.1) -> float:
    """Air drag on the hull and crew, lbf.

    Quantified before being included, which changed how much weight it got:
    at 10 mph of headwind this is about 3% of hull drag and at 15 mph about
    7%. Real, worth a second or two, and nowhere near the dominant term --
    so it is modelled simply and not fussed over.

    The beam-on case was checked too. A crosswind of 15 mph produces a
    heeling arm of under two tenths of an inch against a crew lean that
    already demands more than an inch, so wind heeling is neglected
    deliberately rather than forgotten.
    """
    v_rel = v_ms * 2.23694 + headwind_mph          # mph
    if abs(v_rel) < 1e-9 or frontal_ft2 <= 0:
        return 0.0
    v_fts = v_rel * 1.46667
    rho_air = 0.00238                               # slug/ft^3, sea level
    return math.copysign(0.5 * rho_air * v_fts ** 2 * frontal_ft2 * cd, v_rel)


def added_resistance_waves(chop_in: float, beam_ft: float, lwl_ft: float,
                           displacement_lb: float) -> float:
    """Extra drag from pushing through chop, lbf.

    Non-dimensionalised the standard way for added resistance in waves:

        R_aw = C_aw * rho * g * zeta_a^2 * B^2 / L

    with zeta_a the wave amplitude. On a pond with an inch or two of ripple
    this is a small number, and it is included for completeness rather than
    because it will change a decision. The chop's real cost is water over the
    gunwale, which lives in seakeeping.py, not here.
    """
    if chop_in <= 0:
        return 0.0
    zeta_ft = 0.5 * chop_in / 12.0          # amplitude, half the significant height
    c_aw = 0.6                              # [TYPICAL] blunt small craft, short waves
    return c_aw * C.RHO_LB_FT3 * zeta_ft ** 2 * beam_ft ** 2 / max(lwl_ft, 1e-3)


# ---------------------------------------------------------------------------
# Total resistance
# ---------------------------------------------------------------------------

@dataclass
class DragPoint:
    v_ms: float
    froude: float
    reynolds: float
    cf: float
    form: float
    r_friction: float      # lbf
    r_residuary: float     # lbf
    r_waves: float         # lbf
    r_air: float           # lbf
    r_total: float         # lbf
    power_w: float         # hydrodynamic power = R * v


def resistance(v_ms: float, *, wetted_in2: float, lwl_in: float, beam_wl_in: float,
               draft_in: float, volume_in3: float, displacement_lb: float,
               cb: float, cp: float, chop_in: float = 0.0,
               depth_ft: float = 0.0, residuary_scale: float | None = None,
               headwind_mph: float = 0.0, freeboard_in: float = 0.0,
               p: Params = NOMINAL) -> DragPoint:
    """Total calm-or-choppy water resistance at one speed."""
    if residuary_scale is None:
        residuary_scale = p.residuary_scale
    lwl_ft = max(lwl_in, 1e-6) / 12.0
    beam_ft = max(beam_wl_in, 1e-6) / 12.0
    draft_ft = max(draft_in, 1e-6) / 12.0
    s_ft2 = wetted_in2 / 144.0
    vol_ft3 = max(volume_in3, 1e-9) / 1728.0

    v_fts = v_ms * 3.28084
    re = v_fts * lwl_ft / p.nu_ft2_s
    fr = v_fts / math.sqrt(C.G_FT_S2 * lwl_ft)

    cf = friction_coefficient(re)
    k = form_factor(cb, lwl_ft, beam_ft, draft_ft) * p.form_factor_scale
    q = 0.5 * C.RHO_SLUG_FT3 * v_fts ** 2       # dynamic pressure, lbf/ft^2

    r_f = q * s_ft2 * cf * k

    slenderness = lwl_ft / vol_ft3 ** (1.0 / 3.0)
    r_r = displacement_lb * residuary_over_displacement(fr, slenderness, cp,
                                                        residuary_scale)

    shallow = shallow_water_factor(v_ms, depth_ft, lwl_ft)
    r_f *= shallow
    r_r *= shallow

    r_w = added_resistance_waves(chop_in, beam_ft, lwl_ft, displacement_lb)

    # Frontal area: the hull above water plus a rough allowance for two
    # kneeling bodies, which are most of it.
    frontal = beam_ft * max(freeboard_in, 0.0) / 12.0 + 4.0
    r_air = wind_drag(v_ms, headwind_mph, frontal)

    r_tot = r_f + r_r + r_w + r_air
    return DragPoint(
        v_ms=v_ms, froude=fr, reynolds=re, cf=cf, form=k,
        r_friction=r_f, r_residuary=r_r, r_waves=r_w, r_air=r_air,
        r_total=r_tot,
        power_w=r_tot * v_fts * 1.35582,       # ft.lbf/s -> W
    )


# ---------------------------------------------------------------------------
# Propulsion and the actual race
# ---------------------------------------------------------------------------

def thrust_available(v_ms: float, shaft_power_w: float,
                     max_thrust_lb: float, p: Params = NOMINAL) -> float:
    """Thrust from a given shaft power at a given speed, lbf.

    Power over speed goes to infinity as speed goes to zero, which is not what
    a paddle does. A blade can only push so hard against water before it
    stalls and cavitates the stroke, so thrust is capped. That cap is what
    sets how quickly a heavy boat gets off the line.
    """
    thrust_w = shaft_power_w * p.blade_efficiency
    if v_ms < 1e-3:
        return max_thrust_lb
    thrust_n = thrust_w / v_ms                  # W / (m/s) = N
    return min(thrust_n * 0.224809, max_thrust_lb)


def drag_table(drag_fn, v_max: float = 4.0, n: int = 64):
    """Precompute drag over a speed grid and return a fast interpolator.

    The race integration evaluates drag thousands of times per design, and
    the full resistance calculation is a few dozen floating-point operations
    wrapped in a dataclass. Resistance is smooth and monotonic in speed, so
    a 64-point table interpolated linearly is indistinguishable from the real
    thing and roughly fifty times cheaper. That is the difference between a
    sweep taking an hour and taking a minute.
    """
    grid = np.linspace(0.0, v_max, n)
    vals = np.array([0.0] + [drag_fn(v).r_total for v in grid[1:]])

    def lookup(v: float) -> float:
        return float(np.interp(v, grid, vals))

    return lookup


def terminal_speed(drag_lb_fn, shaft_power_w: float, max_thrust_lb: float,
                   v_hi: float = 4.0, p: Params = NOMINAL) -> float:
    """Steady speed where thrust equals drag. Bisection on (T - R)."""
    def excess(v):
        return thrust_available(v, shaft_power_w, max_thrust_lb, p) - drag_lb_fn(v)

    lo, hi = 0.05, v_hi
    if excess(hi) > 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Pacing and fatigue
#
# Nobody paddles at one constant power for 75 seconds. They go out hard, and
# then they fade, and how much of each decides the time as surely as the hull
# does. So power within the race is modelled rather than assumed constant.
#
# THE MODEL: CRITICAL POWER AND W'
#
# The standard two-parameter description of what a human can produce over a
# few minutes (Monod & Scherrer; Jones and co-workers since):
#
#   CP  critical power, watts. The level you can hold more or less
#       indefinitely. Aerobic.
#   W'  "W prime", joules. A fixed reserve you can spend ABOVE CP, and when
#       it is gone you are pinned at CP whether you like it or not.
#
# Spending above CP drains W' at (P - CP) watts. That is what makes a hard
# start cost something instead of being free, and it is why the sprint-and-die
# strategy dies. Recovery of W' below CP is real but takes minutes, so over a
# 75-second crossing it is neglected here.
#
# WHY THIS MATTERS MORE FOR THIS BOAT THAN FOR MOST
#
# Residuary resistance climbs very steeply past Froude 0.4, which for a
# hundred-inch hull is right where it settles. Overspeeding early therefore
# does not cost a little extra drag, it costs a lot, and that energy comes
# straight out of W'. A boat with a gentler drag curve would forgive a hard
# start; this one does not.
# ---------------------------------------------------------------------------

PACING_SHAPES = ("even", "fast_start", "hard_start", "negative_split")


def pacing_target(strategy: str, progress: float) -> float:
    """Target power as a multiple of the crew's average, versus race progress.

    `progress` runs 0 at the start line to 1 at the finish. Each shape is
    written to average close to 1.0 over the race, so switching strategies
    redistributes the same effort rather than quietly making the crew fitter.
    What separates them is W': shapes that spend early run out early.
    """
    x = min(max(progress, 0.0), 1.0)
    if strategy == "even":
        return 1.0
    if strategy == "fast_start":
        # A brief lift off the line, then settle slightly under.
        return 1.25 if x < 0.15 else 0.956
    if strategy == "hard_start":
        # Sprint it and hang on. W' decides how far "hang on" gets you.
        return 1.60 if x < 0.25 else 0.80
    if strategy == "negative_split":
        return 0.88 if x < 0.5 else 1.12
    return 1.0


@dataclass
class RaceResult:
    time_s: float
    terminal_v: float
    v_at_finish: float
    v_peak: float
    mean_v: float
    froude_at_finish: float
    distance_to_90pct: float    # m to reach 90% of terminal speed
    accel_penalty_s: float      # time lost to not starting at terminal speed
    power_start_w: float        # average over the first quarter
    power_finish_w: float       # average over the last quarter
    fade_frac: float            # 1 - finish/start
    wprime_left_j: float        # anaerobic reserve unspent at the line
    wprime_spent_frac: float
    pinned_at_cp_s: float       # seconds spent with nothing left to give


def simulate_race(drag_lb_fn, *, displacement_lb: float, shaft_power_w: float,
                  max_thrust_lb: float, course_m: float = C.COURSE_M,
                  dt: float = 0.02, turnaround_s: float = 0.0,
                  p: Params = NOMINAL,
                  pacing: str = "even", anaerobic_fraction: float = 0.35,
                  nominal_race_s: float = 78.0,
                  max_power_multiple: float = 2.2) -> RaceResult:
    """Integrate the crossing from a standing start, with a tiring crew.

    Not just distance over terminal speed. The boat has to be accelerated,
    and a heavier hull spends longer doing it -- on a 140 m course that is
    worth real seconds, and it is one of the few places where saving
    cardboard weight buys measurable time rather than comfort. Added mass for
    surge is 10% of displacement, water the hull drags along with it.

    `shaft_power_w` is the crew's AVERAGE power over a race of about
    `nominal_race_s`. It is split into a sustainable part and a spendable
    reserve:

        CP = P_avg * (1 - anaerobic_fraction)
        W' = P_avg * anaerobic_fraction * nominal_race_s

    so that even pacing spends the reserve at exactly the rate that empties
    it on the line. That is deliberate: with the default strategy this
    reproduces the constant-power answer almost exactly, and the fatigue
    machinery only bites when the crew goes out harder than even pace -- which
    is the case worth studying and the one people actually do.
    """
    mass_kg = displacement_lb * 0.453592 * 1.10
    f = float(np.clip(anaerobic_fraction, 0.0, 0.9))
    cp = shaft_power_w * (1.0 - f)
    w_prime = shaft_power_w * f * max(nominal_race_s, 1.0)
    w_left = w_prime

    v = s = t = 0.0
    v_peak = 0.0
    pinned_s = 0.0
    p_sum_start = p_n_start = 0.0
    p_sum_finish = p_n_finish = 0.0

    v_term = terminal_speed(drag_lb_fn, shaft_power_w, max_thrust_lb, p=p)
    d_90 = float("nan")

    max_t = 600.0
    while s < course_m and t < max_t:
        progress = s / course_m
        target = shaft_power_w * pacing_target(pacing, progress)
        target = min(target, max_power_multiple * cp)

        # You cannot spend reserve you do not have.
        affordable = cp + (w_left / dt if dt > 0 else 0.0)
        power = min(target, affordable)
        if power > cp:
            w_left = max(w_left - (power - cp) * dt, 0.0)
        else:
            pinned_s += dt if w_left <= 1e-9 else 0.0

        if progress < 0.25:
            p_sum_start += power
            p_n_start += 1
        elif progress > 0.75:
            p_sum_finish += power
            p_n_finish += 1

        thrust_lb = thrust_available(v, power, max_thrust_lb, p)
        drag_lb = drag_lb_fn(v) if v > 1e-4 else 0.0
        accel = (thrust_lb - drag_lb) * 4.44822 / mass_kg      # lbf -> N -> m/s^2
        v = max(v + accel * dt, 0.0)
        v_peak = max(v_peak, v)
        s += v * dt
        t += dt
        if math.isnan(d_90) and v >= 0.90 * v_term:
            d_90 = s

    p_start = p_sum_start / p_n_start if p_n_start else shaft_power_w
    p_finish = p_sum_finish / p_n_finish if p_n_finish else shaft_power_w
    ideal_t = course_m / v_term if v_term > 1e-6 else float("inf")

    return RaceResult(
        time_s=t + turnaround_s,
        terminal_v=v_term,
        v_at_finish=v,
        v_peak=v_peak,
        mean_v=course_m / t if t > 0 else 0.0,
        froude_at_finish=0.0,
        distance_to_90pct=d_90,
        accel_penalty_s=t - ideal_t,
        power_start_w=p_start,
        power_finish_w=p_finish,
        fade_frac=1.0 - p_finish / max(p_start, 1e-9),
        wprime_left_j=w_left,
        wprime_spent_frac=1.0 - w_left / max(w_prime, 1e-9),
        pinned_at_cp_s=pinned_s,
    )


def calibrate_residuary(measured_time_s: float, drag_fn_factory, *,
                        displacement_lb: float, shaft_power_w: float,
                        max_thrust_lb: float, course_m: float) -> float:
    """Solve for the residuary scale that reproduces a time you actually got.

    HOW TO USE THIS, and it is worth an afternoon:

      1. Build the boat, or a scaled test hull, or borrow any boat whose
         dimensions you can measure.
      2. Paddle a measured distance from a standing start, flat out, and time
         it. Do it three times and take the middle one.
      3. Feed that time in here along with the same crew power you told the
         model to assume.

    What comes back is the number that makes the residuary correlation match
    reality for YOUR hull and YOUR crew. Past that point the model is no
    longer quoting a textbook at you -- it is extrapolating from your own
    measurement, which is a completely different quality of answer.

    Note the coupling: this fits residuary scale GIVEN an assumed crew power.
    Get the power badly wrong and the scale absorbs the error. If you can
    measure only one of the two, measure power at a known steady speed first.
    """
    lo, hi = 0.05, 20.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        res = simulate_race(drag_table(drag_fn_factory(mid)),
                            displacement_lb=displacement_lb,
                            shaft_power_w=shaft_power_w,
                            max_thrust_lb=max_thrust_lb, course_m=course_m)
        if res.time_s < measured_time_s:
            lo = mid            # too fast: needs more drag
        else:
            hi = mid
    return 0.5 * (lo + hi)
