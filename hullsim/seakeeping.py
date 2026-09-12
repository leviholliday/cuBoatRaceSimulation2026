"""
Seakeeping: chop, crew-induced roll, and how she actually gets swamped.

The useful finding in this module, which falls out of the numbers rather than
being assumed: on a pond, the crew is a bigger source of roll than the water
is. An inch or two of ripple moves the waterline by a quarter inch. A paddler
shifting two inches off the centreline heels a marginally stable hull five to
ten degrees, which drops the low gunwale by an inch or more. The chop then
arrives on top of that.

So the chop model here is deliberately modest, and the crew-motion model is
coupled properly to the righting-arm curve -- a stiff boat barely notices the
same lean that rolls a tender one onto its ear.

The swamping chain, which is what actually ends these races:

    ship a little water  ->  heavier, so less freeboard
                         ->  free surface, so less effective GM
                         ->  rolls further for the same lean
                         ->  ships more water

That is a positive feedback loop, and `swamping_cascade` walks it to find how
much water she can take before it runs away. That number -- gallons of reserve
-- is more useful than any single stability figure, because it is the one the
crew can feel while it is happening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import constants as C
from .geometry import HullMesh
from .hydro import (Equilibrium, MassProps, StabilityCurve, free_surface_loss,
                    mass_properties, solve_equilibrium)


# ---------------------------------------------------------------------------
# The wave field
# ---------------------------------------------------------------------------

def chop_wavelength(hs_in: float, fetch_limited: bool = True) -> float:
    """Typical wavelength for chop of a given significant height, inches.

    Short-fetch wind waves sit near their steepness limit. Observed steepness
    for wind ripple on a small lake runs about 1/15 to 1/20, so wavelength is
    roughly seventeen times the significant height. On a pond this keeps the
    waves much shorter than the boat, which is the regime where the hull does
    not contour them and the full wave height shows up as relative motion.
    """
    if hs_in <= 0:
        return 0.0
    return 17.0 * hs_in if fetch_limited else 40.0 * hs_in


def relative_motion_rao(wavelength_in: float, lwl_in: float) -> float:
    """How much of the wave the hull fails to follow, 0 to 1.

    Long waves relative to the hull: she rides over them, the deck stays the
    same distance above the surface, relative motion tends to zero.
    Short waves: she bridges them and barely moves, so the surface comes up
    and down against a nearly stationary hull and relative motion tends to
    the full wave amplitude.

    The transition sits near wavelength equal to waterline length. Pond chop
    is far on the short side of it, which is why the model treats nearly the
    whole wave height as relative motion.
    """
    if wavelength_in <= 1e-6 or lwl_in <= 1e-6:
        return 0.0
    ratio = wavelength_in / (1.2 * lwl_in)
    return float(1.0 / (1.0 + ratio ** 2))


# ---------------------------------------------------------------------------
# Crew-induced roll, coupled to the righting-arm curve
# ---------------------------------------------------------------------------

def heel_from_lean(sc: StabilityCurve, total_weight: float,
                   moving_weight: float, lean_in: float) -> float:
    """Heel angle produced by a paddler shifting their weight sideways.

    Solves W * GZ(phi) = w * lean on the real righting-arm curve rather than
    using the small-angle GM, because a tender hull rolls far enough that the
    small-angle answer is badly optimistic.

    This is the coupling that makes stability feel like something rather than
    being a number in a table: the same two-inch lean that a stiff hull
    shrugs off will put a marginal one's gunwale within an inch of the water
    on every single stroke.
    """
    if lean_in <= 0 or moving_weight <= 0 or total_weight <= 0:
        return 0.0
    demand = moving_weight * lean_in / total_weight     # required GZ, inches

    gz = sc.gz
    ang = sc.heel_deg
    for i in range(1, len(ang)):
        if gz[i] >= demand > gz[i - 1]:
            f = (demand - gz[i - 1]) / max(gz[i] - gz[i - 1], 1e-12)
            return float(ang[i - 1] + f * (ang[i] - ang[i - 1]))
    # Never generates enough righting arm anywhere on the curve: she goes over.
    if demand > sc.gz_max:
        return float(sc.usable_deg)
    return float(ang[-1])


@dataclass
class BoardingResult:
    heel_deg: float
    low_gunwale_drop_in: float
    freeboard_left_in: float
    gz_demand_in: float
    gz_available_in: float
    survives: bool
    verdict: str


def boarding_check(mesh: HullMesh, eq: Equilibrium, sc: StabilityCurve,
                   mp: MassProps, *, offset_frac: float | None = None
                   ) -> BoardingResult:
    """Can the second paddler get in without putting the rail under?

    A commonly fatal moment that hydrostatics at the design condition never
    sees. One paddler is already aboard and settled. The second steps in, and
    for a second or two their entire weight is off the centreline -- not the
    two inches of a paddle stroke but a good fraction of the half-beam,
    because that is where a foot lands when someone is climbing in over a
    gunwale.

    Solved on the real righting-arm curve, so a tender hull is penalised the
    way it deserves rather than being flattered by a small-angle GM.

    `offset_frac` is a quarter of the half-beam: a careful board, stepping
    near the centreline with hands on both gunwales. Raise it toward 0.4 to
    see what a clumsy one costs. Unlike a wave, this is technique the crew
    controls, so it is treated as a floor -- the rail must not actually go
    under -- rather than as a margin to design around.
    """
    d = mesh.design
    if offset_frac is None:
        offset_frac = d.boarding_offset_frac
    if d.n_crew < 2:
        return BoardingResult(0.0, 0.0, eq.freeboard_min, 0.0, sc.gz_max,
                              True, "single-handed: nothing to check")

    boarding_w = max(d.crew_weights)
    lever = offset_frac * 0.5 * eq.beam_wl

    demand = boarding_w * lever / max(mp.weight, 1e-9)
    heel = heel_from_lean(sc, mp.weight, boarding_w, lever)
    drop = 0.5 * eq.beam_wl * math.sin(math.radians(heel))
    left = eq.freeboard_min - drop

    survives = demand < sc.gz_max and left > 0.5
    if demand >= sc.gz_max:
        verdict = ("she goes over during boarding: no righting arm anywhere "
                   "on the curve matches the load")
    elif left <= 0.0:
        verdict = "rail goes under while boarding; she fills before the start"
    elif left < 1.5:
        verdict = f"only {left:.1f} in of rail left while boarding: board slowly, hands on both gunwales"
    else:
        verdict = "boards comfortably"

    return BoardingResult(
        heel_deg=heel, low_gunwale_drop_in=drop, freeboard_left_in=left,
        gz_demand_in=demand, gz_available_in=sc.gz_max,
        survives=survives, verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Water over the gunwale
# ---------------------------------------------------------------------------

@dataclass
class SeakeepingResult:
    chop_in: float
    wavelength_in: float
    rao: float
    sigma_relative_in: float
    paddle_heel_deg: float
    gz_demand_in: float
    heel_freeboard_loss_in: float
    static_freeboard_in: float
    effective_freeboard_in: float
    p_ship_per_wave: float
    n_encounters: float
    p_ship_crossing: float
    expected_shipping_events: float
    verdict: str


def shipping_risk(mesh: HullMesh, eq: Equilibrium, sc: StabilityCurve,
                  mp: MassProps, *, chop_in: float, lean_in: float,
                  speed_ms: float, crossing_s: float) -> SeakeepingResult:
    """Probability of taking water over the gunwale during the crossing.

    Uses Ochi's deck-wetness criterion. In an irregular seaway the relative
    motion between hull and water surface is narrow-banded and Rayleigh
    distributed, so the chance that any one wave exceeds a freeboard f is

        P = exp( -f^2 / (2 * sigma_r^2) )

    with sigma_r the RMS relative motion. This is the standard criterion, and
    the important thing about it is how brutally non-linear it is: freeboard
    appears squared inside an exponential, so halving your freeboard does not
    double the risk, it raises it by orders of magnitude. Freeboard is not a
    linear safety margin. It is a cliff edge, and the model's job is to tell
    you how far from the edge you are standing.
    """
    d = mesh.design
    lam = chop_wavelength(chop_in)
    rao = relative_motion_rao(lam, eq.lwl)
    sigma_wave = chop_in / 4.0           # Hs = 4 * RMS surface elevation
    sigma_r = rao * sigma_wave

    moving = max(d.crew_weights) if d.crew_weights else 0.0
    heel = heel_from_lean(sc, mp.weight, moving, lean_in)
    # The righting arm the crew's own lean demands. If the hull's PEAK
    # righting arm is below this, she cannot resist a normal paddle stroke
    # and goes over on the first one -- regardless of how healthy GM looks.
    gz_demand = moving * lean_in / max(mp.weight, 1e-9)
    half_beam = 0.5 * eq.beam_wl
    heel_loss = half_beam * math.sin(math.radians(heel))

    f_static = eq.freeboard_min
    f_eff = max(f_static - heel_loss, 0.0)

    # Pitch lifts and drops the ends more than the middle, so relative motion
    # is larger there. Scaled linearly with distance from the centre of
    # flotation -- crude, but it is why sheer at the ends is worth having.
    x_rel = abs(eq.freeboard_at - 0.5 * d.length) / max(0.5 * d.length, 1e-9)
    sigma_eff = sigma_r * (1.0 + 0.5 * x_rel)

    if sigma_eff <= 1e-6:
        p_wave = 0.0
    elif f_eff <= 0.0:
        p_wave = 1.0
    else:
        p_wave = math.exp(-f_eff ** 2 / (2.0 * sigma_eff ** 2))

    # Wave encounters over the crossing. Head seas, so celerity adds to speed.
    if lam > 1e-6:
        lam_m = lam * 0.0254
        celerity = math.sqrt(C.G_M_S2 * lam_m / (2.0 * math.pi))
        t_enc = lam_m / max(celerity + speed_ms, 1e-6)
        n_enc = crossing_s / max(t_enc, 1e-6)
    else:
        n_enc = 0.0

    expected = p_wave * n_enc
    p_cross = 1.0 - math.exp(-expected) if expected < 50 else 1.0

    if chop_in <= 0:
        verdict = "flat calm: freeboard is not the binding constraint"
    elif expected < 0.01:
        verdict = "dry crossing"
    elif expected < 0.5:
        verdict = "occasional splash aboard"
    elif expected < 5:
        verdict = "shipping water repeatedly; bail or lose the race"
    else:
        verdict = "green water aboard continuously: she swamps"

    return SeakeepingResult(
        chop_in=chop_in, wavelength_in=lam, rao=rao, sigma_relative_in=sigma_eff,
        paddle_heel_deg=heel, gz_demand_in=gz_demand,
        heel_freeboard_loss_in=heel_loss,
        static_freeboard_in=f_static, effective_freeboard_in=f_eff,
        p_ship_per_wave=p_wave, n_encounters=n_enc,
        p_ship_crossing=p_cross, expected_shipping_events=expected,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# The flooding cascade
# ---------------------------------------------------------------------------

@dataclass
class SwampResult:
    gallons_to_negative_gm: float
    gallons_to_gunwale: float
    gallons_reserve: float
    lanes: int
    steps: list


def _flood_state(mesh: HullMesh, hull_weight: float, gallons: float,
                 lanes: int) -> dict:
    """Float her with a given amount of water aboard."""
    w_water = gallons * 8.345                      # lb per US gallon
    vol_water = w_water / C.RHO_LB_IN3
    floor_area = float(np.trapezoid(2.0 * mesh.half_bottom, mesh.x))
    depth = vol_water / max(floor_area, 1e-9)

    mp = mass_properties(mesh, hull_weight, water_weight=w_water,
                         water_depth=depth)
    # Trim is frozen here. Loose water sits where the boat is deepest and
    # barely moves the longitudinal balance, and solving for trim at every
    # step of the cascade triples its cost for a change in the fourth
    # decimal place.
    eq = solve_equilibrium(mesh, mp, free_to_trim=False)
    fs = free_surface_loss(mesh, eq, depth, lanes, weight_lb=mp.weight)
    return {
        "gallons": gallons, "water_lb": w_water, "water_depth_in": depth,
        "draft_in": eq.draft, "freeboard_in": eq.freeboard_min,
        "gm_solid_in": eq.gm_t, "free_surface_loss_in": fs,
        "gm_effective_in": eq.gm_t - fs, "solved": eq.solved,
    }


def swamping_reserve_fast(mesh: HullMesh, hull_weight: float, *,
                          lanes: int = 1, max_gallons: float = 40.0
                          ) -> float:
    """Gallons of reserve, by bisection. Same answer as the full cascade at a
    fraction of the cost, for use inside a sweep."""
    def ok(g):
        s = _flood_state(mesh, hull_weight, g, lanes)
        return s["gm_effective_in"] > 0.0 and s["freeboard_in"] > 0.0 and s["solved"]

    if not ok(0.0):
        return 0.0
    if ok(max_gallons):
        return max_gallons
    lo, hi = 0.0, max_gallons
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def swamping_cascade(mesh: HullMesh, hull_weight: float, *, lanes: int = 1,
                     max_gallons: float = 60.0, step_gal: float = 1.0
                     ) -> SwampResult:
    """How much water she can take aboard before it runs away.

    Adds water a gallon at a time and re-floats the boat each step, so the
    weight of the water and its free surface both act. Reports the point where
    effective GM goes negative -- she will not come back upright on her own --
    and the point where the gunwale reaches the water.

    Splitting the floor into sealed lanes is the cheapest fix available and
    the model shows why: free-surface loss goes as the cube of the free
    water's width, so n lanes cut it by n squared. Two longitudinal dividers
    are worth more than any amount of extra beam, and cost a few square feet
    of cardboard.
    """
    steps = []
    gal_neg = float("nan")
    gal_gun = float("nan")

    n = int(max_gallons / step_gal) + 1
    for i in range(n):
        gal = i * step_gal
        s = _flood_state(mesh, hull_weight, gal, lanes)
        steps.append(s)

        if math.isnan(gal_neg) and s["gm_effective_in"] <= 0.0:
            gal_neg = gal
        if math.isnan(gal_gun) and (s["freeboard_in"] <= 0.0 or not s["solved"]):
            gal_gun = gal
        if not math.isnan(gal_neg) and not math.isnan(gal_gun):
            break

    found = [g for g in (gal_neg, gal_gun) if not math.isnan(g)]
    reserve = min(found) if found else max_gallons

    return SwampResult(
        gallons_to_negative_gm=gal_neg,
        gallons_to_gunwale=gal_gun,
        gallons_reserve=float(reserve),
        lanes=lanes, steps=steps,
    )


# ---------------------------------------------------------------------------
# Dynamic roll
# ---------------------------------------------------------------------------

@dataclass
class RollResponse:
    roll_period_s: float
    encounter_period_s: float
    tuning_ratio: float           # roll period / encounter period; 1 = resonant
    amplification: float
    wave_slope_deg: float
    beam_filter: float
    roll_amplitude_deg: float
    resonant: bool
    note: str


def roll_response(eq: Equilibrium, chop_in: float, speed_ms: float,
                  gm_in: float, damping: float = 0.18) -> RollResponse:
    """How far the chop rolls her, including resonance.

    Everything else in this model treats heel quasi-statically: apply a
    moment, read the angle off the righting-arm curve. That is only valid if
    the boat can follow the forcing, and whether it can depends on how its
    natural roll period compares with how fast the waves arrive.

        T_roll = 2*pi*k / sqrt(g * GM),   k ~ 0.37 * beam

    Then a damped single-degree-of-freedom amplification against the
    encounter period, and a beam filter: when the boat is wide compared with
    the wavelength it spans several waves at once and the slopes cancel, so
    the forcing largely disappears.

    TWO RESULTS FROM THIS THAT ARE NOT OBVIOUS

    At race speed the chop is short and arrives fast, and the amplification
    comes out near zero -- the hull simply cannot respond, so the quasi-static
    treatment used everywhere else is justified rather than merely assumed.

    STOPPED, it is a different boat. Sitting at the line or being boarded,
    the encounter period stretches to the wave period, and for a hull this
    size in two or three inches of chop that lands close to resonance. The
    dangerous moment for wave-induced roll is before the start, not during
    the race -- which is also exactly when somebody is climbing in over the
    gunwale.

    A stiffer hull is not automatically safer here, either. High GM shortens
    the roll period and can tune it INTO the chop. Nothing else in the model
    pushes back on ever-increasing GM; this does.
    """
    if chop_in <= 0 or gm_in <= 0 or eq.beam_wl <= 0:
        return RollResponse(float("nan"), float("nan"), 0.0, 0.0, 0.0, 0.0,
                            0.0, False, "flat calm")

    k_roll = 0.37 * eq.beam_wl          # roll radius of gyration, small craft
    t_roll = 2.0 * math.pi * k_roll / math.sqrt(C.G_IN_S2 * gm_in)

    lam = chop_wavelength(chop_in)                       # inches
    celerity = math.sqrt(C.G_IN_S2 * lam / (2.0 * math.pi))
    v_in_s = speed_ms * 39.3701
    t_enc = lam / max(celerity + v_in_s, 1e-6)

    r = t_roll / max(t_enc, 1e-9)
    amp = 1.0 / math.sqrt((1.0 - r * r) ** 2 + (2.0 * damping * r) ** 2)

    # Wave slope amplitude, then averaged across the beam.
    slope = math.pi * (0.5 * chop_in) / lam * 2.0        # rad, crest-to-trough
    ratio = math.pi * eq.beam_wl / lam
    beam_filter = abs(math.sin(ratio) / ratio) if ratio > 1e-9 else 1.0

    roll = math.degrees(slope * beam_filter * amp)
    resonant = 0.75 < r < 1.35

    if roll < 1.0:
        note = "she cannot respond to chop this short: quasi-static heel is fine"
    elif resonant:
        note = ("roll period is tuned to the chop: she will roll several times "
                "further than the waves are steep")
    else:
        note = "noticeable roll, off resonance"

    return RollResponse(
        roll_period_s=t_roll, encounter_period_s=t_enc, tuning_ratio=r,
        amplification=amp, wave_slope_deg=math.degrees(slope),
        beam_filter=beam_filter, roll_amplitude_deg=roll,
        resonant=resonant, note=note,
    )
