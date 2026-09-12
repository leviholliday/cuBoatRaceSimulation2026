"""
The crew: where their mass sits, where their shoulders are, and how much room
they need.

WHY THIS IS ITS OWN MODULE NOW

Crew centre-of-gravity height used to be a number you typed in, picked off a
short table of postures. That hid two things that turn out to matter:

  1. It scales with how tall you are. A five-foot-three paddler kneeling
     the same way as a five-foot-ten one has a centre of gravity about two
     inches lower, which is free stability that the old table could not see.

  2. Posture is a TRADE, not a free win. Sitting back on your heels drops
     your centre of gravity, which the stability model rewards. It also drops
     your shoulders toward the gunwale, which makes the stroke worse -- and
     nothing in the model knew that, so it happily recommended crouching as
     low as physically possible. The fifty-year-old Great Cardboard Boat
     Regatta building notes say it in one line: tall sides make it harder to
     paddle. This module is where that finally costs something.

EVERYTHING HERE IS FOR KNEELING CANOE PADDLING

Not kayak. A kayaker sits on the bottom with legs forward and a much lower
centre of gravity, uses a double blade, and has entirely different clearance
geometry. Numbers borrowed from kayak ergonomics would be wrong here in both
directions at once. The postures below are all knees-down, which is what the
team is actually doing.

THE ANTHROPOMETRY

Standard segment fractions of stature H (Winter, Biomechanics and Motor
Control of Human Movement, and the general anthropometric literature):

    whole-body CG, standing        0.55 H
    hip joint, standing            0.53 H
    shoulder, standing             0.82 H
    knee, standing                 0.285 H
    legs                           32% of body mass
    trunk + head + arms            68% of body mass

From those, trunk-plus-head-plus-arms has its CG 0.147 H above the hip joint,
which is the number that does the work below. Checked against the posture
table the project used before: for a 70 in person this reproduces 21.1 in for
kneeling tall against the old table's 21.0, and 15.5 in for a mid kneel
against its 15.5. So this is a generalisation of the old numbers rather than
a replacement for them, and it now moves correctly with stature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Fraction of stature at which the hip joint sits, per posture. Everything
# else follows from this.
POSTURE_HIP_FRAC = {
    "kneeling_tall": 0.26,
    "kneeling_upright": 0.15,
    "kneeling_low": 0.11,
    "back_on_heels": 0.06,
    "sitting_flat": 0.05,
}
POSTURE_NOTES = {
    "kneeling_tall": "thighs vertical, torso upright — most reach, highest CG",
    "kneeling_upright": "knees down, partly sat back — the usual racing kneel",
    "kneeling_low": "sat further back, knees wide for bracing",
    "back_on_heels": "fully sat on the heels — lowest practical CG, least reach",
    "sitting_flat": "sitting on the floor — lowest CG, barely able to paddle",
}

LEG_MASS_FRAC = 0.32
TRUNK_CG_ABOVE_HIP_FRAC = 0.147     # of stature
SHOULDER_ABOVE_HIP_FRAC = 0.29      # of stature


@dataclass
class PaddlerGeometry:
    posture: str
    height_in: float
    hip_z: float           # hip joint above the hull floor
    cg_z: float            # whole-body centre of gravity above the floor
    shoulder_z: float      # shoulder above the floor
    footprint_in: float    # fore-and-aft room this paddler occupies


def paddler_geometry(height_in: float, posture: str = "kneeling_upright",
                     shin_in: float = 20.0) -> PaddlerGeometry:
    """Where one kneeling paddler's mass and shoulders end up.

    `shin_in` is knee to heel, which sets how much of the boat's length the
    paddler occupies fore and aft. Add a little for the knees themselves and
    for leaning into the stroke.
    """
    h = max(float(height_in), 36.0)
    frac = POSTURE_HIP_FRAC.get(posture)
    if frac is None:
        raise KeyError(f"unknown posture {posture!r}; "
                       f"known: {', '.join(sorted(POSTURE_HIP_FRAC))}")

    hip_z = frac * h
    z_trunk = hip_z + TRUNK_CG_ABOVE_HIP_FRAC * h
    # Legs folded under: thighs run from knee (floor) up to the hip, shanks
    # and feet lie flat. Their combined CG sits low and scales with hip height.
    z_legs = 0.30 * hip_z + 0.02 * h
    cg_z = LEG_MASS_FRAC * z_legs + (1.0 - LEG_MASS_FRAC) * z_trunk

    shoulder_z = hip_z + SHOULDER_ABOVE_HIP_FRAC * h

    # Knee to heel, plus the knees, plus room to hinge forward on the catch.
    footprint = shin_in + 0.10 * h

    return PaddlerGeometry(posture=posture, height_in=h, hip_z=hip_z,
                           cg_z=cg_z, shoulder_z=shoulder_z,
                           footprint_in=footprint)


# ---------------------------------------------------------------------------
# Can they actually paddle over the side?
# ---------------------------------------------------------------------------

FULL_CLEARANCE_IN = 17.0
"""Shoulder-above-gunwale height at which the stroke stops being constrained
   at all. Around this the paddler can swing the shaft freely and plant the
   blade where they want it."""

MIN_CLEARANCE_IN = 3.0
"""Below this the gunwale is effectively at shoulder height and there is no
   usable forward stroke left."""

MIN_REACH_EFFICIENCY = 0.55


def paddle_reach_efficiency(shoulder_z: float, gunwale_z: float) -> dict:
    """How much of the stroke survives the height of the side.

    The blade has to clear the gunwale and still reach water that is well
    outboard of it. The less room between the paddler's shoulder and the rail,
    the flatter and shorter the stroke becomes: the shaft has to be held out
    and angled rather than driven down, and less of the blade force ends up
    pointing along the boat.

    Modelled as a smooth loss between a comfortable clearance and none at all.
    The shape is judgement; the EXISTENCE of the penalty is not, and without
    it the model recommends crouching as low as a body can fold, because
    every other term rewards a low centre of gravity and nothing objects.

    Canoe-specific. A kayaker's geometry is different in every respect and
    these numbers do not transfer.
    """
    clearance = float(shoulder_z - gunwale_z)
    span = max(FULL_CLEARANCE_IN - MIN_CLEARANCE_IN, 1e-9)
    f = (clearance - MIN_CLEARANCE_IN) / span
    eff = MIN_REACH_EFFICIENCY + (1.0 - MIN_REACH_EFFICIENCY) * float(
        np.clip(f, 0.0, 1.0))

    if clearance >= FULL_CLEARANCE_IN:
        note = "free swing over the rail"
    elif clearance >= 10.0:
        note = "slightly cramped but a proper stroke is available"
    elif clearance >= MIN_CLEARANCE_IN:
        note = "reaching over a high rail: the stroke goes flat and short"
    else:
        note = "the gunwale is at shoulder height; there is no stroke here"

    return {"clearance_in": clearance, "efficiency": eff, "note": note}


# ---------------------------------------------------------------------------
# Where they sit
# ---------------------------------------------------------------------------

PADDLE_CLASH_GAP_IN = 10.0
"""Clear water needed between two kneeling paddlers on top of their own
   footprints, so the blades and the recovery swing do not meet."""

END_MARGIN_FRAC = 0.06
"""Neither paddler kneels right in the tapered end, where there is no beam to
   kneel on and the rocker has lifted the floor."""


def spacing_check(length_in: float, crew_x_frac, geometries) -> dict:
    """Do two kneeling paddlers fit where they have been put?

    Three ways they do not: they overlap each other, their paddles clash, or
    one of them is kneeling out in a tapered end. The team can slide fore and
    aft freely, so this is a real design variable and worth sweeping -- it
    moves trim, which moves the waterline, which moves everything.
    """
    xs = [f * length_in for f in crew_x_frac]
    order = np.argsort(xs)
    problems = []

    gap = float("inf")
    if len(xs) >= 2:
        for a, b in zip(order[:-1], order[1:]):
            need = 0.5 * (geometries[a].footprint_in
                          + geometries[b].footprint_in) + PADDLE_CLASH_GAP_IN
            gap = min(gap, xs[b] - xs[a])
            if xs[b] - xs[a] < need:
                problems.append(
                    f"paddlers {a + 1} and {b + 1} are {xs[b] - xs[a]:.0f} in "
                    f"apart and need {need:.0f} in: knees overlap or paddles clash")

    lo = END_MARGIN_FRAC * length_in
    hi = (1.0 - END_MARGIN_FRAC) * length_in
    for i, (x, g) in enumerate(zip(xs, geometries)):
        if x - 0.5 * g.footprint_in < lo or x + 0.5 * g.footprint_in > hi:
            problems.append(f"paddler {i + 1} is kneeling in the tapered end")

    return {"gap_in": gap if math.isfinite(gap) else 0.0,
            "problems": problems, "ok": not problems}


def resolve_crew(design) -> dict:
    """Turn heights and postures into the numbers the physics wants.

    Returns per-paddler CG heights (which replace the hand-typed crew_kg when
    a posture is named), shoulder heights, footprints, and the combined reach
    efficiency. One place, so CG and reach can never disagree about what
    posture someone is in.
    """
    n = len(design.crew_weights)
    heights = _pad(design.crew_height_in, n, 68.0)
    postures = _pad(design.crew_posture, n, "kneeling_upright")
    shins = _pad(design.crew_shin_in, n, 20.0)

    geoms, kgs = [], []
    for i in range(n):
        if postures[i] == "custom":
            kgs.append(_pad(design.crew_kg, n, 15.0)[i])
            g = paddler_geometry(heights[i], "kneeling_upright", shins[i])
            geoms.append(g)
        else:
            g = paddler_geometry(heights[i], postures[i], shins[i])
            geoms.append(g)
            kgs.append(g.cg_z)
    return {"geometries": geoms, "crew_kg": tuple(kgs),
            "heights": heights, "postures": postures, "shins": shins}


def _pad(seq, n, default):
    out = list(seq) if seq else []
    while len(out) < n:
        out.append(out[-1] if out else default)
    return out[:n]
