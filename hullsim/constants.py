"""
Physical constants and material properties.

EVERY number in this file is one of three kinds, and the kind is always labelled:

    [EXACT]     A definition or a physical constant. Not adjustable.
    [TYPICAL]   A published value for this class of material. Defensible in a
                report, but it is not YOUR cardboard. Measure it if it matters.
    [MEASURE]   A placeholder that is currently a guess. Replace it with your
                own bench number before you trust anything downstream of it.

Units: this project works internally in INCHES, POUNDS (force) and SECONDS,
because that is how the cardboard, the tape roll and the crew are measured.
SI conversions are provided where the source literature is metric.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Water and gravity
# ---------------------------------------------------------------------------

G_FT_S2 = 32.174            # [EXACT] standard gravity, ft/s^2
G_IN_S2 = G_FT_S2 * 12.0    # [EXACT] in/s^2
G_M_S2 = 9.80665            # [EXACT] m/s^2

# Fresh water. 62.4 lb/ft^3 is the textbook value at ~60 F.
RHO_LB_FT3 = 62.4                    # [EXACT enough] lb/ft^3
RHO_LB_IN3 = RHO_LB_FT3 / 1728.0     # [EXACT] lb/in^3  = 0.0361
RHO_SLUG_FT3 = RHO_LB_FT3 / G_FT_S2  # [EXACT] 1.940 slug/ft^3

# Kinematic viscosity of fresh water. Strongly temperature dependent:
#   10 C -> 1.31e-6 m^2/s,  15 C -> 1.14e-6,  20 C -> 1.00e-6
# An Ohio lake in September sits around 18-21 C. 1.05e-6 is the fair pick.
NU_M2_S = 1.05e-6                        # [TYPICAL] 19 C fresh water
NU_FT2_S = NU_M2_S * 10.7639             # [EXACT conversion] 1.130e-5 ft^2/s

# ---------------------------------------------------------------------------
# Corrugated board
#
# The team's stock is described as ~4 mm, coated one side. That is single-wall
# C-flute (nominal caliper 3.6-4.0 mm), the most common shipping board.
#
# Sources for the [TYPICAL] values below:
#   - Flute caliper / geometry: standard C-flute is 3.6-4.0 mm with ~39 flutes
#     per foot. Widely published; see e.g. InSite Packaging flute-profile guide.
#   - ECT: single-wall C-flute is commercially specified at 32 lb/in (the
#     standard grade) or 44 lb/in (upgraded). ECT is the CD edgewise crush
#     strength, i.e. load per inch of board edge, flutes vertical.
#   - Bending stiffness: single-wall C-flute is reported in the 8-14 N.m range
#     machine direction and roughly half that cross direction. 1.5-1.7x MD:CD
#     anisotropy is the commonly quoted ratio.
#
# NONE of these were measured on the team's actual board. Anything the model
# concludes about absolute structural margin inherits that uncertainty; the
# RELATIVE comparison between two hulls is far more trustworthy than the
# absolute number.
# ---------------------------------------------------------------------------

BOARD_CALIPER_IN = 0.157        # [TYPICAL] 4.0 mm single-wall C-flute
BOARD_AREAL_LB_FT2 = 0.135      # [TYPICAL] ~660 g/m^2 incl. one-side coating
BOARD_ECT_LB_IN = 32.0          # [TYPICAL] standard single-wall C-flute
BOARD_D_MD_LBIN = 88.5          # [TYPICAL] 10.0 N.m flexural rigidity, MD
BOARD_D_CD_LBIN = 48.7          # [TYPICAL] 5.5 N.m flexural rigidity, CD

# Effective in-plane Young's modulus of the board treated as a solid plate of
# thickness BOARD_CALIPER_IN. Back-calculated from D = E t^3 / (12 (1-v^2)):
#   E_MD = 12 (1 - 0.3^2) D_MD / t^3
# This is a smeared-plate idealisation. The real board is a sandwich and this
# modulus has no meaning outside that idealisation -- it is only used to keep
# the buckling and girder formulas dimensionally honest.
BOARD_POISSON = 0.30            # [TYPICAL] smeared orthotropic board
_T3 = BOARD_CALIPER_IN ** 3
BOARD_E_MD_PSI = 12.0 * (1 - BOARD_POISSON ** 2) * BOARD_D_MD_LBIN / _T3
BOARD_E_CD_PSI = 12.0 * (1 - BOARD_POISSON ** 2) * BOARD_D_CD_LBIN / _T3

# Wet knockdown. Corrugated board loses strength as it takes up moisture:
# roughly 50% of compression strength at 80% RH, and far more once liquid
# water wicks into the flutes. A fully taped hull on a 2-3 minute crossing sees
# splash and seam wicking, not soaking.
#   1.00 = bone dry,  0.70 = taped hull, short exposure (default),
#   0.45 = long exposure or a leaking seam,  0.20 = visibly soggy.
# SOAKING, WITH TAPE AS THE ONLY WATERPROOFING
#
# This team's rules do not allow paint or any sealant. Tape is it. That makes
# tape coverage of the immersed hull the single thing controlling how fast
# the board goes soft, and it means the tape budget is not only a structural
# decision -- it is the waterproofing decision as well, competing for the
# same finite roll.
#
# Anchors from the literature, both for TREATED board, which is better than
# anything achievable here with tape alone:
#   - a coated corrugated board lost 47% of flat crush after 5 min immersed
#   - a board treated with a commercial water-resistant coating lost 51%
# So roughly half the strength goes in five minutes even when the board has
# a proper coating. Bare board is worse and faster; taped board is better
# only where the tape actually is.

SOAK_FLOOR_BARE = 0.22          # [TYPICAL] untaped board, long immersion
SOAK_FLOOR_SEALED = 0.55        # [TYPICAL] fully taped, long immersion
SOAK_TAU_BARE_MIN = 1.2         # [MEASURE] bare board wicks in about a minute
SOAK_TAU_SEALED_MIN = 8.2       # [MEASURE] fully taped, seams only

WET_FACTOR_FLOOR = SOAK_FLOOR_SEALED
WET_FACTOR_DEFAULT = WET_FACTOR_FLOOR


def soak_params(tape_coverage: float) -> tuple[float, float]:
    """Long-immersion floor and time constant, given how much of the immersed
    hull is actually under tape.

    Linear between bare and fully sealed. Crude, but it puts the right
    variable in charge: with no paint allowed, the only thing between the
    board and the lake is how much tape you were willing to spend."""
    c = min(max(float(tape_coverage), 0.0), 1.0)
    floor = SOAK_FLOOR_BARE + (SOAK_FLOOR_SEALED - SOAK_FLOOR_BARE) * c
    tau = SOAK_TAU_BARE_MIN + (SOAK_TAU_SEALED_MIN - SOAK_TAU_BARE_MIN) * c
    return floor, tau

# ---------------------------------------------------------------------------
# Tape
# ---------------------------------------------------------------------------

TAPE_WIDTH_MM = 48.0                          # [EXACT] the roll they have
TAPE_WIDTH_IN = TAPE_WIDTH_MM / 25.4          # 1.89 in
TAPE_ROLL_M = 100.0                           # [EXACT] one roll
TAPE_ROLL_IN = TAPE_ROLL_M * 39.3701          # 3937 in
TAPE_TENSILE_LB_IN = 25.0                     # [TYPICAL] cloth duct tape
TAPE_AREAL_LB_FT2 = 0.040                     # [TYPICAL] ~0.30 mm cloth tape
TAPE_LB_PER_IN = TAPE_AREAL_LB_FT2 * TAPE_WIDTH_IN / 144.0   # weight per linear inch

# ---------------------------------------------------------------------------
# Stock
# ---------------------------------------------------------------------------

SHEET_LEN_FT = 55.0
SHEET_WIDTH_IN = 41.0
SHEET_AREA_IN2 = SHEET_LEN_FT * 12.0 * SHEET_WIDTH_IN   # 27,060 in^2

# ---------------------------------------------------------------------------
# Course
# ---------------------------------------------------------------------------

COURSE_M = 140.0                # [EXACT] Cedar Lake crossing
COURSE_IN = COURSE_M * 39.3701

# ---------------------------------------------------------------------------
# Crew / propulsion
#
# Elite flatwater kayakers put out ~400 W at sprint pace (Gomes et al., power
# balance studies). Two untrained-to-moderately-fit college students paddling a
# barge for ~2 minutes is nowhere near that. Blade propulsive efficiency (the
# fraction of shaft power that becomes useful thrust, the rest being blade slip)
# runs 0.60-0.75 for a paddle.
#
# These are the numbers most worth replacing with a measurement: paddle a known
# distance, time it, and back out the power the model needs to match.
# ---------------------------------------------------------------------------

PADDLER_POWER_W_DEFAULT = 160.0   # [MEASURE] shaft power per paddler, 2 min effort
BLADE_EFFICIENCY = 0.65           # [TYPICAL] paddle propulsive efficiency
PADDLE_WEIGHT_LB = 2.0            # [TYPICAL] each

# Crew centre-of-gravity height above the hull floor, by posture. These come
# from the team's own earlier analysis and are worth confirming with the
# inclining test rather than trusting.
# Superseded by hullsim/crew.py, which derives centre-of-gravity height from
# stature and posture instead of using one table for everybody. Kept only as
# the reference the new model was checked against: for a 70 in paddler it
# reproduces these to within half an inch.
POSTURE_KG_IN_LEGACY_70IN = {
    "kneeling_tall": 21.0,
    "kneeling_low": 15.5,
    "sitting_flat": 11.0,
}


def soak_factor(minutes_in_water: float, floor: float = WET_FACTOR_FLOOR,
                tau_min: float = SOAK_TAU_SEALED_MIN) -> float:
    """Strength retained after a hull has been in the water this long.

    Starts at 1.0 dry and decays exponentially toward `floor`. Used instead of
    one fixed knockdown because the hull is nearly dry at the gun and weakest
    at the finish, and because a crew that launches ten minutes early starts
    the race with a materially weaker boat than one that launches at two.
    """
    t = max(float(minutes_in_water), 0.0)
    return float(floor + (1.0 - floor) * math.exp(-t / max(tau_min, 1e-6)))

# ---------------------------------------------------------------------------
# Numerical settings
# ---------------------------------------------------------------------------

N_STATIONS = 41         # longitudinal stations for hydrostatic integration
HEEL_ANGLES_DEG = (0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 35, 40, 50, 60)
