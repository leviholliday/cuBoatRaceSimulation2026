"""
HullDesign: every knob the simulator understands, in one dataclass.

A design is a plain value object. It holds no results and does no physics.
Everything downstream takes one of these and returns numbers, which is what
makes the sweep driver trivial and the physics testable.

Coordinate system used everywhere in this project:

    x  longitudinal, 0 at the bow tip, increasing AFT, to `length` at the stern
    y  transverse,   0 on the centreline, positive to starboard
    z  vertical,     0 at the LOWEST point of the keel, positive UP

All lengths in inches, weights in pounds, angles in degrees unless the name
says otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Tuple

from . import constants as C


def _coerce(text: str, like):
    if isinstance(like, bool):
        return text.strip().lower() in ("true", "1")
    if isinstance(like, int):
        return int(round(float(text)))
    if isinstance(like, float):
        return float(text)
    return text


@dataclass(frozen=True)
class HullDesign:
    # ---------------- principal dimensions ----------------
    length: float = 102.0
    """Length overall, stem to stern."""

    bottom_width: float = 29.0
    """Width of the flat bottom panel at midship."""

    beam: float = 35.0
    """Width at the gunwale at midship. Flare = beam - bottom_width."""

    side_height: float = 13.0
    """Depth at midship, from the bottom panel up to the gunwale."""

    # ---------------- plan view (looking down) ----------------
    bow_taper: float = 24.0
    """Length over which the hull narrows toward the stem. 0 = blunt scow bow."""

    stern_taper: float = 12.0
    """Length over which it narrows toward the stern. 0 = full-width transom."""

    bow_entry_exp: float = 1.0
    """Shape of the bow taper in plan.
       0.5 = blunt / full entry, 1.0 = straight wedge, 2.0 = fine hollow entry.
       Lower numbers carry volume further forward; higher numbers cut water
       more cleanly but give away buoyancy exactly where the bow needs it."""

    stern_exit_exp: float = 1.0
    """Same idea aft. A canoe-like run wants ~1.5; a transom is handled by
       stern_taper = 0 instead."""

    stem_width_frac: float = 0.04
    """Bottom half-width at the very stem, as a fraction of the midship value.
       Cardboard cannot fold to a true knife edge; this is the taped stem."""

    transom_width_frac: float = 0.40
    """Bottom width at the very stern as a fraction of midship. Only used when
       stern_taper > 0. With stern_taper = 0 the stern is full width."""

    # ---------------- profile view (from the side) ----------------
    bow_rocker: float = 2.0
    """How far the keel line lifts at the bow above its lowest point.
       This is the 'tipped up in front' the team had heard about."""

    stern_rocker: float = 1.0
    """Same at the stern."""

    rocker_exp: float = 2.0
    """How the rocker develops from midship to the end.
       1.0 = straight ramp, 2.0 = parabolic (flat amidships), 3.0 = flat
       amidships with a sharp upturn right at the ends."""

    bow_sheer: float = 3.0
    """How far the gunwale line rises at the bow above the midship gunwale.
       This is free freeboard exactly where spray arrives."""

    stern_sheer: float = 1.5
    """Same at the stern."""

    sheer_exp: float = 2.0
    """Shape of the sheer rise, same convention as rocker_exp."""

    # ---------------- section shape ----------------
    chine_frac: float = 0.5
    """Fraction of the local depth over which the side flares out from the
       bottom panel to full beam. Above that the topsides run vertical.
       1.0 = a pure trapezoid (flares all the way to the gunwale)
       0.5 = flares over the bottom half, vertical above
       ->0 = a plain box with vertical sides"""

    # ---------------- structure ----------------
    skin_layers: int = 1
    """Layers of board in the hull skin. 2 doubles weight and tape but roughly
       8x the panel bending stiffness if the flutes are crossed."""

    flute_direction: str = "transverse"
    """Orientation of the corrugations in the hull skin.
       'transverse'   = flutes run around the girth (bow-to-stern bending is
                        weak, hoop stiffness is strong, easy to wrap)
       'longitudinal' = flutes run bow to stern (strong hull girder, but the
                        board will not wrap around the chine without crushing)"""

    n_frames: int = 4
    """Transverse ring frames. They set the panel span for buckling and carry
       the torsion box."""

    gunwale_tube_dia: float = 2.0
    gunwale_tube_wraps: int = 3
    chine_log_dia: float = 1.5
    chine_log_wraps: int = 3
    n_thwarts: int = 3

    decked_ends: bool = True
    """Decks over bow and stern. They close the torsion box, which is worth
       several orders of magnitude of torsional stiffness, and they keep
       green water out of the ends."""

    tape_diag_density: float = 1.0
    """Coverage of the primary +45 degree tape layer over the bottom, 0 to 1.

       Running tape at 45 degrees puts it along the direction principal stress
       actually takes under torque, which is why it is worth doing at all. But
       full coverage of the bottom eats over half the roll on its own, so this
       is a real budget decision and it belongs in the sweep rather than being
       assumed."""

    tape_diag_opposing: float = 0.33
    """Coverage of the opposing -45 layer, as a fraction of full. Torque can
       come either way; the opposing layer is what catches the other one."""

    joint_type: str = "full_wrap"
    """How tube-to-tube and tube-to-panel joints are made. Scored from the
       team's own bench tests -- see data/joint_tests.yaml.
       'full_wrap' | 'partial_flap' | 'through_axle' | 'butt_tape'"""

    chine_tape_layers: int = 2
    """Layers of tape over each chine seam, inside plus outside.

       This is not cosmetic. Once the bottom panel deflects past its own
       thickness it carries load as a membrane, and the membrane tension has
       to be reacted at the chine -- by this tape. structure.panels() reports
       how close that tension is to the tape's strength, and on a bare
       single-layer bottom it is usually the first thing to run out."""

    bottom_doubler: bool = True
    """An extra layer under the kneeling stations, flutes crossed."""

    # ---------------- crew ----------------
    crew_weights: Tuple[float, ...] = (114.4, 170.0)
    """Measured. The lighter paddler is the fitter one."""

    crew_height_in: Tuple[float, ...] = (63.0, 65.0)
    """Standing height of each paddler.

       Not cosmetic. Centre-of-gravity height, shoulder height and how much
       of the boat each paddler occupies all scale with stature, and a
       five-foot-three crew kneels about an inch and a half lower than a
       five-foot-ten one for free."""

    crew_shin_in: Tuple[float, ...] = (20.0, 20.0)
    """Knee to heel. Sets the fore-and-aft room each kneeling paddler needs,
       which is what stops them being placed too close together."""

    crew_posture: Tuple[str, ...] = ("kneeling_upright", "kneeling_upright")
    """How each paddler kneels. See crew.POSTURE_HIP_FRAC.

       'kneeling_tall' | 'kneeling_upright' | 'kneeling_low'
       | 'back_on_heels' | 'sitting_flat' | 'custom'

       A named posture DERIVES crew_kg from stature, so the centre of gravity
       and the shoulder height can never disagree about what position someone
       is in. 'custom' falls back to whatever crew_kg says.

       This is a genuine trade rather than a free win: sitting lower drops the
       centre of gravity, which helps stability, and drops the shoulders
       toward the gunwale, which costs stroke."""

    crew_kg: Tuple[float, ...] = (14.0, 14.5)
    """CG height above the hull floor. Ignored unless crew_posture is
       'custom' -- otherwise crew.py derives it from height and posture."""

    crew_x_frac: Tuple[float, ...] = (0.30, 0.72)
    """Where each paddler kneels, as a fraction of length from the BOW.

       Free to slide, and worth sweeping: moving the crew changes trim, which
       changes the waterline, which changes everything. Bounded by knees
       overlapping and paddles clashing -- see crew.spacing_check."""

    crew_power_w: Tuple[float, ...] = (170.0, 150.0)
    """AVERAGE shaft power each paddler holds over the whole crossing. The
       lighter paddler here is the fitter one -- that asymmetry is a real
       input, not a detail.

       Average, not peak: nobody holds their ten-second power for seventy-five
       seconds. How that average is distributed across the race is set by
       `pacing` and `anaerobic_fraction` below."""

    pacing: str = "even"
    """How the crew spends their effort across the course.

       'even'            one steady effort the whole way
       'fast_start'      a brief lift off the line, then settle just under
       'hard_start'      sprint it and hang on
       'negative_split'  hold back early, build over the second half

       Each shape averages to roughly the same total effort, so this is a
       question of distribution, not fitness. What separates them is the
       anaerobic reserve: shapes that spend early run out early, and on a
       hull whose drag climbs steeply past Froude 0.4 the wasted energy of
       overspeeding does not come back."""

    anaerobic_fraction: float = 0.35
    """Fraction of the crew's average power that comes from the spendable
       anaerobic reserve rather than from sustainable aerobic output.

       Sets the split into critical power and W' -- see the long note in
       resistance.py. 0.35 is reasonable for an untrained pair over about
       seventy-five seconds. Higher means more of their effort is burnable
       reserve, which makes a hard start survivable for longer and the fade
       worse when it comes."""

    # ---------------- environment / operation ----------------
    headwind_mph: float = 0.0
    """Wind along the course. Positive is a headwind, negative a tailwind.
       Worth about 3% of hull drag at 10 mph and 7% at 15 -- real, but not a
       term to agonise over."""

    chop_height: float = 1.0
    """Significant wave height on the course, inches. 0 = glass,
       1 = light ripple, 2 = noticeable chop and boat wakes, 3 = windy."""

    paddle_lean_in: float = 2.0
    """How far a paddler's weight moves off the centreline on each stroke.

       Given as a DISTANCE rather than a heel angle on purpose. The angle is
       not an input, it is a result: the same two-inch lean barely tips a
       stiff hull and rolls a tender one onto its ear. seakeeping.heel_from_lean
       solves it against the real righting-arm curve, which is what makes
       stability couple back into how wet the crew gets."""

    free_water_lanes: int = 1
    """How many sealed longitudinal channels the floor is split into. Water
       that gets aboard can only slosh within one lane, and free-surface loss
       goes as the cube of the free water's width, so n lanes cut the loss by
       n squared. 1 = open cockpit, 3 = two dividers."""

    wet_factor: float = C.WET_FACTOR_FLOOR
    """Strength the board retains after LONG immersion -- the floor it decays
       toward, not a fixed knockdown. How far down that curve she actually
       gets is set by how long she has been in the water."""

    pre_race_soak_min: float = 3.0
    """Minutes afloat before the gun: launching, boarding, drifting to the
       line, waiting for the start.

       This is where nearly all the soaking happens. The race itself is barely
       75 seconds; sitting at the line for ten minutes costs far more strength
       than paddling the course does."""

    build_list_in: float = 0.0
    """Permanent transverse centre-of-gravity offset from an asymmetric build.

       One side wider, heavier, or better taped than the other. Invisible
       until you float her, and then permanent: she sits with a list, and
       that list subtracts from the righting arm available for every stroke
       of the race. It enters the model exactly like a lean that never goes
       away, which is why half an inch of it is a large fraction of what one
       paddler's lean already demands."""

    extra_weight_lb: float = 0.0
    """Weight the take-off does not predict: patches, doubled seams, overlap
       you did not plan, a repair. Always positive."""

    water_aboard_gal: float = 0.0
    """Water already in her at the start -- splash from boarding, mostly.
       Carried as weight AND as a free surface, so it costs freeboard and
       righting arm at the same time."""

    boarding_offset_frac: float = 0.25
    """How far off the centreline the second paddler's weight lands while
       climbing in, as a fraction of the half-beam. 0.25 is a careful board
       with hands on both gunwales; 0.4 is a clumsy one."""

    hull_weight_override: float | None = None
    """Set this once the real hull has been on a scale. Until then the hull
       weight is computed from the material take-off, which is the honest way
       but carries the board-density uncertainty with it."""

    # ---------------- bookkeeping ----------------
    name: str = ""

    # -------------------------------------------------------------------
    def with_(self, **kw) -> "HullDesign":
        """Return a copy with some fields changed. Designs are immutable so a
        sweep can hand the same base around without anyone mutating it."""
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_row(self) -> dict:
        """Every field, flattened so a whole design survives a CSV round trip.

        Tuple fields become pipe-separated strings. Without this the sweep
        writes only the columns the charts happen to use, and re-loading a row
        silently substitutes baseline values for everything else -- so the
        detailed report would describe a subtly different boat from the one
        that scored. `from_row` is the exact inverse.
        """
        out = {}
        for k, v in asdict(self).items():
            out[k] = "|".join(str(x) for x in v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_row(cls, row) -> "HullDesign":
        """Rebuild a design from a `to_row` mapping. Missing keys keep their
        defaults, so an older or hand-edited CSV still loads."""
        kw = {}
        for f, spec in cls.__dataclass_fields__.items():
            if f not in row:
                continue
            v = row[f]
            if v is None or (isinstance(v, float) and v != v):   # NaN
                continue
            cur = getattr(cls(), f)
            if isinstance(cur, tuple):
                parts = [p for p in str(v).split("|") if p != ""]
                kw[f] = tuple(_coerce(p, cur[0] if cur else 0.0) for p in parts)
            elif isinstance(cur, bool):
                kw[f] = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1")
            elif isinstance(cur, int):
                kw[f] = int(round(float(v)))
            elif isinstance(cur, float):
                kw[f] = float(v)
            elif isinstance(cur, str):
                kw[f] = str(v)
        return cls(**kw)

    # -------------------------------------------------------------------
    @property
    def crew_weight(self) -> float:
        return float(sum(self.crew_weights))

    @property
    def n_crew(self) -> int:
        return len(self.crew_weights)

    @property
    def flare(self) -> float:
        return self.beam - self.bottom_width

    def validate(self) -> list[str]:
        """Geometric nonsense that should never reach the physics. Returns a
        list of problems; empty list means the design is at least coherent."""
        p = []
        if self.length <= 0 or self.beam <= 0 or self.side_height <= 0:
            p.append("non-positive principal dimension")
        if self.bottom_width > self.beam:
            p.append("bottom_width exceeds beam (tumblehome is not modelled)")
        if self.bow_taper + self.stern_taper > self.length:
            p.append("bow and stern tapers overlap")
        if not (0.0 < self.chine_frac <= 1.0):
            p.append("chine_frac must be in (0, 1]")
        if self.bow_rocker >= self.side_height or self.stern_rocker >= self.side_height:
            p.append("rocker equals or exceeds side height: no depth left at the end")
        if self.skin_layers < 1:
            p.append("skin_layers must be at least 1")
        if len(self.crew_weights) != len(self.crew_kg) != len(self.crew_x_frac):
            p.append("crew arrays are different lengths")
        if any(not (0.0 < f < 1.0) for f in self.crew_x_frac):
            p.append("crew_x_frac must be strictly inside the hull")
        if self.flute_direction not in ("transverse", "longitudinal"):
            p.append(f"unknown flute_direction {self.flute_direction!r}")
        return p


# ---------------------------------------------------------------------------
# Named reference designs
# ---------------------------------------------------------------------------

# The hull the team's earlier hand analysis landed on. Kept exactly as it was
# so the tool can be honest about it: at full diagonal tape coverage it is
# over the tape roll, which is a real finding rather than a bug to tune away.
BASELINE = HullDesign(name="baseline-102x29x35x13")

# What a 20,000-design sweep landed on, rounded to numbers you can actually
# cut to. Round dimensions matter: nobody measures 105.199 inches.
#
# Two things about it are worth noticing, because neither is what you would
# guess from canoe intuition:
#
#   The section is nearly a box. Only three inches of total flare. Flare is
#   what makes a canoe seaworthy in real water and gives a paddler somewhere
#   to reach, but on flat water it costs waterplane inertia for a given beam
#   AND it is the one thing that makes a side panel non-developable. On a
#   pond, in cardboard, a near-box section wins on both counts.
#
#   The rocker is small. An inch and a half at the bow. The "tipped up in
#   front" idea buys a little in chop and costs buoyancy where the bow needs
#   it most, so the sweep keeps only a token amount.
RECOMMENDED = HullDesign(
    length=104.0, bottom_width=33.0, beam=36.0, side_height=14.0,
    bow_taper=15.0, stern_taper=0.0, bow_entry_exp=1.4,
    bow_rocker=1.5, stern_rocker=0.5, bow_sheer=2.0, stern_sheer=0.5,
    chine_frac=0.75, n_frames=6, free_water_lanes=2,
    tape_diag_density=0.35, tape_diag_opposing=0.15, chine_tape_layers=2,
    crew_posture=("kneeling_upright", "kneeling_upright"),
    pre_race_soak_min=1.0,
    name="recommended-104x33x36x14",
)
# THREE CHOICES HERE ARE NOT DIMENSIONS, AND ALL THREE ARE FREE
#
# Posture: an upright kneel, not a crouch. Crouching lowers the centre of
# gravity, which every stability term rewards, and lowers the shoulders
# toward the rail, which costs stroke -- so there is an optimum in the
# middle rather than at the bottom. Kneeling TALL scores best on a single
# deterministic run and passes only about a fifth of simulated builds;
# kneeling upright gives up a few tenths of a point and passes nearly all of
# them. See crew.py.
#
# Side height 14 in rather than 15: deep enough for freeboard, shallow
# enough that a five-foot-three paddler can still get a stroke over it. The
# two pull opposite ways and the optimum is interior.
#
# One minute afloat before the gun, not three. Board strength decays toward
# its wet floor with a time constant of about six minutes, and the race
# itself is barely seventy-five seconds -- so almost all the soaking a hull
# suffers happens while it waits at the line. Launching ten minutes early
# costs more strength than the entire crossing does.

# The same idea stretched for speed. About six seconds quicker with slightly
# less of every margin -- a fair illustration of the trade rather than a
# recommendation.
FAST = HullDesign(
    length=112.0, bottom_width=31.0, beam=34.0, side_height=14.0,
    bow_taper=18.0, stern_taper=6.0, bow_entry_exp=1.4,
    bow_rocker=1.5, stern_rocker=0.5, bow_sheer=2.0, stern_sheer=0.5,
    chine_frac=0.75, n_frames=7, free_water_lanes=2,
    tape_diag_density=0.35, tape_diag_opposing=0.15, chine_tape_layers=2,
    crew_posture=("kneeling_upright", "kneeling_upright"),
    pre_race_soak_min=1.0,
    name="fast-112x31x34x14",
)

NAMED = {"baseline": BASELINE, "recommended": RECOMMENDED, "fast": FAST}
