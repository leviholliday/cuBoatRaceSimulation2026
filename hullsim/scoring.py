"""
Constraints and scoring.

Two separate jobs, kept apart on purpose:

  CONSTRAINTS are pass/fail. A hull that cannot displace its own weight is not
  a slightly worse hull, it is not a hull. These never get traded against
  anything.

  THE SCORE is a ranking for everything that passed. It is a preference, not a
  fact, and the honest thing to do with a preference is put it in one editable
  place with the weights visible, rather than bury it in the physics.

WHY THE SCORE IS AN EXPECTED VALUE

The brief was 'we are not going for optimal, we are going for the win, and if
we sink that is not good'. That is exactly the structure of an expected value,
so that is how it is written:

    expected score = P(finish) * points-if-you-finish   +   build points

A hull that is two seconds faster but has a one-in-four chance of going in the
water scores worse than a slightly slower one that always finishes, and it
does so automatically, without anyone having to remember to be cautious. The
failure probabilities are crude -- they are smooth functions of the safety
margins, not validated reliability models -- but the SHAPE is right, and the
shape is what makes the ranking behave sensibly.

THE RUBRIC IS A GUESS AND YOU SHOULD REPLACE IT

The weights in `Rubric` below are a reasonable default for a cardboard boat
race. They are not your course's actual rubric, because that was not in any of
the material available when this was written. Edit the defaults, or pass your
own Rubric, once you have the real scoring sheet. Everything downstream reads
from it, so that is a one-place change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------

@dataclass
class Limits:
    """Pass/fail thresholds. A design violating any of these is discarded."""

    min_freeboard_in: float = 4.0
    """Static freeboard at the lowest point of the gunwale, before any heel."""

    min_gm_in: float = 2.0
    """Small-angle metacentric height, solid (no free surface)."""

    min_effective_gm_in: float = 0.5
    """GM after crew lean and a gallon of shipped water. The real margin."""

    min_downflood_deg: float = 20.0
    """Heel angle at which the gunwale goes under. Below this, one bad stroke
       is the whole race."""

    max_paddle_heel_frac: float = 0.50
    """How much of the way to downflooding a single paddle stroke is allowed
       to take her.

       This is the constraint that catches hulls GM alone waves through. A
       hull can have a healthy GM -- the SLOPE of its righting arm at zero
       heel -- and still have a peak righting arm smaller than what one
       paddler leaning two inches off the centreline demands. Narrow-bottomed
       flared hulls do this routinely: the curve is steep at the origin and
       then flattens early, so she feels stiff for the first few degrees and
       then simply keeps going. GM cannot see it; the peak of the curve can."""

    min_gz_margin_ratio: float = 1.6
    """Peak righting arm divided by the arm the crew's lean demands. Below
       1.0 she capsizes on a normal stroke. 1.6 leaves something for a wake,
       a bad catch, or someone reaching for a dropped paddle."""

    min_gz_area: float = 15.0
    """Area under the righting-arm curve out to downflooding, in.deg. This is
       the energy reserve -- how much of a shove she can absorb, as opposed to
       how stiffly she resists the first degree of it."""

    max_warp_ratio: float = 1.0
    """Flare-panel non-developability, as a fraction of what the MATERIAL
       will take. Corrugated crushes its flutes under forced double
       curvature and tolerates very little; a thin solid sheet bends happily
       and tolerates about three times as much. Judging the raw warp against
       one fixed number would call the same hull unbuildable in one material
       and fine in the other, which is exactly backwards."""

    max_girder_utilisation: float = 0.95
    max_chine_tape_util: float = 0.95
    """Membrane tension in the bottom panel against the chine tape's strength.
       Over 1.0 the seam that is holding the floor on is overloaded."""
    max_twist_deg: float = 12.0
    max_draft_frac: float = 0.55
    """Draft as a fraction of side height. More than this and there is not
       enough hull left above water to be worth calling freeboard."""

    min_boarding_freeboard_in: float = 0.0
    """Rail left above water while the second paddler is climbing in, with
       their weight a third of the half-beam off the centreline. A moment the
       design condition never sees, and a common way to start the race
       already full of water."""

    min_reach_clearance_in: float = 5.0
    """Shoulder above the gunwale, for the shorter paddler in their chosen
       posture. Below this there is no forward stroke worth the name, however
       beautiful the hydrostatics are."""

    require_crew_fit: bool = True
    """Knees must not overlap, paddles must not clash, and nobody kneels out
       in a tapered end."""

    min_swamp_reserve_gal: float = 4.0
    """Gallons of water she can take aboard before losing positive stability."""

    require_material_fit: bool = True
    require_tape_fit: bool = True
    require_floats: bool = True


def check_constraints(r: dict, lim: Limits) -> list[str]:
    """Return the list of violated constraints. Empty means feasible."""
    v = []
    if lim.require_floats and not r["floats"]:
        v.append("does not float")
    if r["freeboard_in"] < lim.min_freeboard_in:
        v.append(f"freeboard {r['freeboard_in']:.1f} < {lim.min_freeboard_in}")
    if r["gm_in"] < lim.min_gm_in:
        v.append(f"GM {r['gm_in']:.1f} < {lim.min_gm_in}")
    if r["gm_effective_in"] < lim.min_effective_gm_in:
        v.append(f"effective GM {r['gm_effective_in']:.1f} < {lim.min_effective_gm_in}")
    if r["downflood_deg"] < lim.min_downflood_deg:
        v.append(f"downflooding at {r['downflood_deg']:.0f} deg")
    heel_frac = r["paddle_heel_deg"] / max(r.get("usable_heel_deg") or 1e-9, 1e-9)
    if heel_frac > lim.max_paddle_heel_frac:
        v.append(f"one paddle stroke heels her {heel_frac:.0%} of the way "
                 f"to downflooding")
    if r.get("gz_margin_ratio", 9e9) < lim.min_gz_margin_ratio:
        v.append(f"peak righting arm only {r['gz_margin_ratio']:.2f}x what the "
                 f"crew's lean demands")
    if r["gz_area"] < lim.min_gz_area:
        v.append(f"righting energy {r['gz_area']:.0f} < {lim.min_gz_area}")
    if r.get("warp_ratio", 0.0) > lim.max_warp_ratio:
        v.append(f"panel warp {r['panel_warp']:.3f} is "
                 f"{r['warp_ratio']:.1f}x what this material will take")
    if r["girder_utilisation"] > lim.max_girder_utilisation:
        v.append(f"hull girder at {r['girder_utilisation']:.0%}")
    if r["chine_tape_util"] > lim.max_chine_tape_util:
        v.append(f"chine tape at {r['chine_tape_util']:.0%} of its strength")
    if r["twist_deg"] > lim.max_twist_deg:
        v.append(f"twists {r['twist_deg']:.0f} deg")
    if r["draft_frac"] > lim.max_draft_frac:
        v.append(f"draft is {r['draft_frac']:.0%} of side height")
    if r.get("boarding_freeboard_in", 9e9) < lim.min_boarding_freeboard_in:
        v.append(f"only {r['boarding_freeboard_in']:.1f} in of rail left "
                 f"while boarding")
    if r.get("reach_clearance_in", 9e9) < lim.min_reach_clearance_in:
        v.append(f"only {r['reach_clearance_in']:.1f} in from shoulder to "
                 f"gunwale: cannot paddle over the side")
    if lim.require_crew_fit and not r.get("crew_spacing_ok", True):
        v.append("crew do not fit where they are placed")
    if r.get("boarding_freeboard_with_chop_in", 9e9) < 0.0:
        v.append("rail goes under while boarding in chop")
    if r["swamp_reserve_gal"] < lim.min_swamp_reserve_gal:
        v.append(f"only {r['swamp_reserve_gal']:.0f} gal of swamping reserve")
    if lim.require_material_fit and r["board_over"]:
        v.append("over the cardboard budget")
    if lim.require_tape_fit and r["tape_over"]:
        v.append("over the tape budget")
    return v


# ---------------------------------------------------------------------------
# Failure probabilities
# ---------------------------------------------------------------------------

def _logistic_survival(margin: float, half: float) -> float:
    """Smooth 0-1 survival probability from a safety margin.

    `margin` is how far past the danger point you are, in the same units as
    `half`, which is the margin at which survival is 50/50. Deliberately a
    gentle curve: the point is to penalise thin margins continuously so the
    optimiser backs away from cliffs, not to claim a calibrated failure rate.

    The exponent is clamped because a sweep will hand this wildly infeasible
    designs -- a hull whose chine tape is at 5000% of strength gives a margin
    of -50 and overflows a float exponential. Such a design is discarded by
    the constraints anyway; it just must not take the sweep down with it.
    """
    z = float(np.clip(margin / max(half, 1e-6), -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-z))


def finish_probability(r: dict) -> dict:
    """Chance of crossing the line upright, broken down by failure mode.

    None of these are validated reliability numbers, and they should not be
    quoted as if they were. What they do is turn 'how much margin is left'
    into something that can be multiplied and traded off, which is what makes
    the expected-score ranking behave the way a cautious crew would.
    """
    # Capsize: how much of the available heel range the crew's own rolling
    # already uses. usable_heel_deg rather than downflood_deg because a hull
    # with enormous freeboard may never immerse a gunwale inside the computed
    # range, leaving downflood_deg undefined.
    limit_deg = r.get("usable_heel_deg")
    if not limit_deg or (isinstance(limit_deg, float) and math.isnan(limit_deg)):
        limit_deg = r.get("downflood_deg") or 60.0
    roll_use = r["paddle_heel_deg"] / max(limit_deg, 1e-6)
    p_upright = _logistic_survival(0.55 - roll_use, 0.12)

    # Swamping: expected shipping events over the crossing against reserve.
    events = r["expected_shipping_events"]
    reserve = max(r["swamp_reserve_gal"], 0.0)
    # Each shipping event is taken as roughly a third of a gallon aboard.
    gal_expected = 0.33 * events
    p_dry = _logistic_survival(reserve - 3.0 * gal_expected, 2.5)

    # Structure: worst of the three utilisations.
    worst = max(r["girder_utilisation"],
                r["twist_deg"] / 12.0,
                r["panel_deflection_in"] / 1.2,
                r["chine_tape_util"])
    p_intact = _logistic_survival(0.85 - worst, 0.18)

    # Joints, from the bench tests. Ranked data gets a wider penalty band
    # than measured data, because a ranking is not a margin.
    p_joints = _logistic_survival(r["joint_multiplier"] - 0.45, 0.18)

    p = p_upright * p_dry * p_intact * p_joints
    return {
        "p_upright": p_upright, "p_dry": p_dry, "p_intact": p_intact,
        "p_joints": p_joints, "p_finish": p,
    }


# ---------------------------------------------------------------------------
# The rubric
# ---------------------------------------------------------------------------

@dataclass
class Rubric:
    """Competition scoring. EDIT THESE to match your actual scoring sheet."""

    points_finish: float = 40.0
    """Awarded for finishing at all, regardless of time."""

    points_speed: float = 40.0
    """Awarded on a sliding scale between `slow_time_s` and `fast_time_s`."""

    fast_time_s: float = 70.0
    slow_time_s: float = 180.0

    points_build: float = 20.0
    """Craftsmanship, design documentation, and the like. Approximated here
       from things the model can actually see: whether the hull is foldable
       from flat board, whether it fits the material budget with margin, and
       whether the structure has reserve rather than scraping through."""

    def speed_points(self, time_s: float) -> float:
        if not math.isfinite(time_s):
            return 0.0
        f = (self.slow_time_s - time_s) / max(self.slow_time_s - self.fast_time_s, 1e-6)
        return self.points_speed * float(np.clip(f, 0.0, 1.0))

    def build_points(self, r: dict) -> float:
        warp = float(np.clip(1.0 - r["panel_warp"] / 0.055, 0.0, 1.0))
        spare = float(np.clip(r["board_spare_frac"] / 0.25, 0.0, 1.0))
        tape = float(np.clip(r["tape_spare_frac"] / 0.20, 0.0, 1.0))
        struct = float(np.clip(r["structure_score"], 0.0, 1.0))
        return self.points_build * (0.35 * warp + 0.2 * spare + 0.15 * tape
                                    + 0.30 * struct)


def score_design(r: dict, rubric: Rubric | None = None) -> dict:
    """Expected competition score, and the pieces it came from."""
    rubric = rubric or Rubric()
    probs = finish_probability(r)
    p = probs["p_finish"]

    if_finish = rubric.points_finish + rubric.speed_points(r["time_s"])
    build = rubric.build_points(r)
    expected = p * if_finish + build

    return {
        **probs,
        "points_if_finish": if_finish,
        "points_build": build,
        "expected_score": expected,
        "best_case_score": if_finish + build,
    }
