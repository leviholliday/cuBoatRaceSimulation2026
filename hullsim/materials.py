"""
Material take-off: cardboard area, tape length, and the hull's own weight.

Everything here is derived from the actual mesh rather than from formulas with
the dimensions substituted in, so a hull with more rocker really does need
more skin, and a longer bow taper really does cost more stem seam. That
matters because cardboard and tape are the two hard constraints on the whole
project -- one roll of each, and no second chances -- and a take-off that
quietly assumes a prismatic box will under-count exactly the shapes the sweep
is most likely to recommend.

The hull weight that comes out of here feeds straight back into displacement,
so an over-optimistic take-off shows up as an over-optimistic freeboard. The
single best thing you can do to this module is weigh a measured square of your
actual board and put the number in data/joint_tests.json.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .geometry import HullMesh
from .params import NOMINAL, Params


def _curve_length(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Arc length of a 3-D curve through the station points."""
    return float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)))


@dataclass
class MaterialResult:
    board_items: list
    board_used_in2: float
    board_available_in2: float
    board_spare_frac: float
    board_over: bool
    bottom_tape_coverage: float

    tape_items: list
    tape_used_in: float
    tape_available_in: float
    tape_spare_frac: float
    tape_over: bool
    tape_reserve_in: float

    hull_weight_lb: float
    board_weight_lb: float
    tape_weight_lb: float

    flags: list = field(default_factory=list)


def take_off(mesh: HullMesh, *, tape_reserve_in: float = 250.0,
             p: Params = NOMINAL) -> MaterialResult:
    """Cut list and tape plan for one hull."""
    d = mesh.design
    x = mesh.x
    depth = np.maximum(mesh.gunwale_z - mesh.keel_z, 1e-9)
    hc = np.clip(d.chine_frac, 1e-3, 1.0) * depth

    # --- developed girth at each station -------------------------------
    w_bottom = 2.0 * mesh.half_bottom
    flare_run = mesh.half_beam - mesh.half_bottom
    flare_rise = hc
    w_flare = 2.0 * np.hypot(flare_run, flare_rise)
    w_topside = 2.0 * np.maximum(depth - hc, 0.0)

    # Longitudinal edge curves, which are what the panels are actually cut to.
    len_keel = _curve_length(x, np.zeros_like(x), mesh.keel_z)
    len_chine = _curve_length(x, mesh.half_beam, mesh.keel_z + hc)
    len_sheer = _curve_length(x, mesh.half_beam, mesh.gunwale_z)

    a_bottom = float(np.trapezoid(w_bottom, x)) * (len_keel / max(d.length, 1e-9))
    a_flare = float(np.trapezoid(w_flare, x)) * (len_chine / max(d.length, 1e-9))
    a_topside = float(np.trapezoid(w_topside, x)) * (len_sheer / max(d.length, 1e-9))

    lap = 2.0                                    # in of overlap on every seam
    skin = (a_bottom + a_flare + a_topside) * d.skin_layers
    skin += lap * (len_chine * 2 + len_sheer * 2)

    items = [("Hull skin, %d layer(s), developed" % d.skin_layers, skin)]

    if d.bottom_doubler:
        pads = float(np.trapezoid(w_bottom, x)) * 0.35
        items.append(("Bottom doubler under the kneeling stations, flutes crossed", pads))

    mid = mesh.n_stations // 2
    b_gun_mid = 2.0 * mesh.half_beam[mid]
    h_mid = float(depth[mid])

    if d.decked_ends:
        # Deck area is the plan area over the tapered ends.
        n_b = max(int(d.bow_taper / max(d.length / (mesh.n_stations - 1), 1e-9)), 1)
        n_s = max(int(d.stern_taper / max(d.length / (mesh.n_stations - 1), 1e-9)), 1)
        a_bow = float(np.trapezoid(2.0 * mesh.half_beam[:n_b + 1], x[:n_b + 1]))
        a_stern = float(np.trapezoid(2.0 * mesh.half_beam[-(n_s + 1):], x[-(n_s + 1):]))
        items.append(("Bow + stern decks (these close the torsion box)", a_bow + a_stern))

    if d.stern_taper < 1e-6:
        items.append(("Transom", b_gun_mid * h_mid))

    if d.n_frames:
        # Each ring frame is a 4 in wide strip following the section girth.
        girth_mid = float(w_bottom[mid] + w_flare[mid] + w_topside[mid])
        items.append((f"Ring frames x{d.n_frames}, 4 in strip",
                      d.n_frames * 4.0 * girth_mid))

    if d.free_water_lanes > 1:
        n_div = d.free_water_lanes - 1
        items.append((f"Longitudinal dividers x{n_div} (free-surface control)",
                      n_div * d.length * min(h_mid, 6.0) * 1.2))

    def tube_area(dia, wraps, length):
        return math.pi * dia * wraps * length

    items.append((f"Gunwale tubes x2 ({d.gunwale_tube_dia:g} in, {d.gunwale_tube_wraps} wraps)",
                  2 * tube_area(d.gunwale_tube_dia, d.gunwale_tube_wraps, len_sheer)))
    items.append((f"Chine logs x2 ({d.chine_log_dia:g} in, {d.chine_log_wraps} wraps)",
                  2 * tube_area(d.chine_log_dia, d.chine_log_wraps, len_chine)))
    items.append((f"Thwarts x{d.n_thwarts} (2 in, 3 wraps)",
                  d.n_thwarts * tube_area(2.0, 3, b_gun_mid)))
    items.append(("Kneel pads, 2 stations x 3 layers", 2 * 3 * 18 * 24))

    board_used = sum(a for _, a in items)

    # --- tape ------------------------------------------------------------
    tw = C.TAPE_WIDTH_IN
    ang = math.radians(45.0)
    b_bot_mid = float(w_bottom[mid])

    # +/-45 diagonals over the bottom: the direction principal stress runs
    # under torque, which is the whole reason to bias the tape rather than
    # running it fore and aft.
    diag_len = b_bot_mid / math.sin(ang)
    n_diag = d.length / (tw / math.cos(ang))

    dens = float(np.clip(d.tape_diag_density, 0.0, 1.0))
    opp = float(np.clip(d.tape_diag_opposing, 0.0, 1.0))
    tape = [
        (f"Bottom skin, +45 diagonals, {dens:.0%} coverage",
         n_diag * diag_len * dens),
        (f"Opposing -45 diagonals, {opp:.0%} coverage",
         n_diag * diag_len * opp),
        (f"Chine seams x2, {d.chine_tape_layers} layer(s) each",
         2 * len_chine * d.chine_tape_layers),
        ("Sheer / lap seam x2", 2 * len_sheer),
        ("Stem seams, doubled", 2 * 2 * math.hypot(d.bow_taper, h_mid)),
        ("Gunwale cut-edge capping x2", 2 * len_sheer),
    ]
    if d.stern_taper < 1e-6:
        tape.append(("Transom perimeter", b_gun_mid + 2 * h_mid))
    else:
        tape.append(("Stern seams, doubled", 2 * 2 * math.hypot(d.stern_taper, h_mid)))
    if d.n_frames:
        girth_mid = float(w_bottom[mid] + w_flare[mid] + w_topside[mid])
        tape.append((f"Ring frame tabs x{d.n_frames}", d.n_frames * girth_mid * 0.8))
    if d.decked_ends:
        tape.append(("Deck perimeter seams", 2 * (b_gun_mid + d.bow_taper + d.stern_taper)))
    if d.free_water_lanes > 1:
        tape.append((f"Divider seams x{d.free_water_lanes - 1}",
                     (d.free_water_lanes - 1) * 2 * d.length))
    tape.append(("Tube seams, single line + spots",
                 (2 * len_sheer + 2 * len_chine + d.n_thwarts * b_gun_mid) * 0.6))
    tape.append(("Thwart and tube end joints", 24.0 * (4 + d.n_thwarts * 2)))
    tape.append(("RACE-DAY REPAIR RESERVE", tape_reserve_in))

    tape_used = sum(v for _, v in tape)

    # --- weight ------------------------------------------------------------
    board_w = board_used / 144.0 * p.board_areal_lb_ft2
    tape_w = tape_used * p.tape_lb_per_in
    hull_w = board_w + tape_w

    flags = []
    if board_used > C.SHEET_AREA_IN2:
        flags.append(f"over the cardboard roll by {board_used - C.SHEET_AREA_IN2:,.0f} in2")
    if tape_used > C.TAPE_ROLL_IN:
        flags.append(f"over the tape roll by {tape_used - C.TAPE_ROLL_IN:,.0f} in")
    if tape_used - tape_reserve_in > C.TAPE_ROLL_IN:
        flags.append("over the tape roll even before the repair reserve")

    # How much of the IMMERSED hull is actually under tape. With no paint
    # allowed this is the waterproofing, not just the shear path. The two
    # diagonal layers are laid independently, so their union is
    # a + b - a*b rather than a + b.
    coverage = dens + opp - dens * opp

    return MaterialResult(
        bottom_tape_coverage=float(min(max(coverage, 0.0), 1.0)),
        board_items=items, board_used_in2=board_used,
        board_available_in2=C.SHEET_AREA_IN2,
        board_spare_frac=(C.SHEET_AREA_IN2 - board_used) / C.SHEET_AREA_IN2,
        board_over=board_used > C.SHEET_AREA_IN2,
        tape_items=tape, tape_used_in=tape_used,
        tape_available_in=C.TAPE_ROLL_IN,
        tape_spare_frac=(C.TAPE_ROLL_IN - tape_used) / C.TAPE_ROLL_IN,
        tape_over=tape_used > C.TAPE_ROLL_IN,
        tape_reserve_in=tape_reserve_in,
        hull_weight_lb=hull_w, board_weight_lb=board_w, tape_weight_lb=tape_w,
        flags=flags,
    )
