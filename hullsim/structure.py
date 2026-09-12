"""
Structure: will the hull hold its shape long enough to finish?

Hydrostatics tells you whether a rigid hull of this shape floats. It says
nothing about whether cardboard can BE that shape under load, and for a
cardboard boat that is usually the question that decides the race. This module
covers the four ways these hulls actually fail:

  1. Torsion. The hull is an open trough, and an open thin-walled section has
     essentially no torsional stiffness. Two paddlers stroking on opposite
     sides twist it like a wet towel. This is the number-one observed failure.
  2. Hull girder bending. Two point masses sitting on distributed buoyancy.
     The hull is a beam, and cardboard is a poor one.
  3. Panel deflection. The bottom between frames is a plate under water
     pressure with someone kneeling on it. Left unsupported it does not fail
     so much as give up and hang from the tape.
  4. Joints. Where tubes meet panels. Not derivable from published data, so
     it is driven by the team's own bench tests -- see data/joint_tests.json.

A NOTE ON THE MATERIAL DIRECTION, WHICH IS EASY TO GET BACKWARDS

Corrugated board has a stiff axis and a soft axis, and the MD/CD naming used
in the packaging literature is a reliable source of confusion. This module
avoids it and talks about flutes instead, which you can check by hand in two
seconds:

    Take a scrap. Try to fold it with the crease running ALONG the flutes.
    It folds easily -- you are just crushing the flutes flat.
    Now try to fold it with the crease running ACROSS the flutes.
    It fights you -- you would have to shorten the flutes to do it.

So a panel is STIFF when its span runs across the flutes and SOFT when its
span runs along them. Same for strength: edgewise compression along the flute
axis is strong (that is what the flutes are for, and what ECT measures);
across the flutes the liners buckle between flute contacts at a fraction of
that. Everything below is written in those terms.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import constants as C
from .geometry import HullMesh
from .params import NOMINAL, Params

# Every material property this module needs now comes from a Params instance
# rather than a module constant, so a Monte Carlo trial can hand it a
# different-but-plausible board. Params() reproduces the old numbers exactly.

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Joints -- empirical, from the team's own bench tests
# ---------------------------------------------------------------------------

def load_joint_data(path: Path | None = None) -> dict:
    """Load the measured joint table.

    Deliberately not derived from theory. The published corrugated-board
    literature is built around flat-panel box compression -- the McKee formula
    and its descendants predict how a shipping box crushes under top load.
    That is a genuinely different failure mode from a taped cardboard tube
    loaded in bending at a joint, and dressing one up as the other would give
    a number that looks rigorous and is wrong.

    The team's bench tests are the better source for this specific
    construction, so they are the input.
    """
    p = path or (DATA_DIR / "joint_tests.json")
    if not p.exists():
        return {"joints": {}, "tube_wraps": {}, "calibrated": False}
    with open(p) as fh:
        return json.load(fh)


def joint_strength(joint_type: str, data: dict | None = None) -> tuple[float, str]:
    """Relative strength multiplier for a joint type, and its provenance.

    Returns (multiplier, source) where source is 'measured' if it came from
    a bench test with a real load in it, and 'ranked' if it is only the
    observed ordering with placeholder magnitudes.
    """
    data = data or load_joint_data()
    j = data.get("joints", {}).get(joint_type)
    if not j:
        return 1.0, "unknown"
    return float(j.get("relative_strength", 1.0)), j.get("source", "ranked")


def tube_capacity(dia_in: float, wraps: int, data: dict | None = None,
                  p: Params = NOMINAL) -> dict:
    """Bending capacity of a rolled cardboard tube.

    Two competing effects, both of which the team saw on the bench:
      - more wraps means more material and more capacity, but with diminishing
        returns once the inner wraps stop being able to shed load outward;
      - a smaller diameter is stronger per wrap, but changes the failure mode
        from a gradual bend to a sudden buckle, which gives no warning.

    The second one matters for design: a member that buckles without warning
    is worse than a slightly weaker one that sags first, because the crew can
    react to a sag.
    """
    data = data or load_joint_data()
    wrap_tbl = data.get("tube_wraps", {})

    t_eff = wraps * p.board_caliper_in
    r_mean = max(0.5 * dia_in - 0.5 * t_eff, 0.05)

    # Thin-walled tube section modulus.
    i_sec = math.pi * r_mean ** 3 * t_eff
    z_sec = i_sec / max(r_mean + 0.5 * t_eff, 1e-6)

    # Capacity from material strength, with the wrap efficiency the bench
    # tests showed (the first wraps do almost nothing until the tube closes).
    eff = float(wrap_tbl.get(str(wraps), {}).get("efficiency",
                _default_wrap_efficiency(wraps)))
    m_material = z_sec * p.sigma_along_flutes * eff

    # Local (Brazier / shell) buckling of a thin tube in bending. This is the
    # mode that produces the sudden kink rather than a gradual sag.
    # Classical shell buckling stress: sigma_cr = k * E * t / r
    e_eff = 0.5 * (p.e_stiff_psi + p.e_soft_psi)
    sigma_cr = p.shell_buckle_knockdown * e_eff * t_eff / r_mean
    m_buckle = z_sec * sigma_cr

    governs = "buckling" if m_buckle < m_material else "material"
    return {
        "dia_in": dia_in,
        "wraps": wraps,
        "wall_t_in": t_eff,
        "section_modulus_in3": z_sec,
        "moment_material_lbin": m_material,
        "moment_buckling_lbin": m_buckle,
        "moment_capacity_lbin": min(m_material, m_buckle),
        "governing_mode": governs,
        "warns_before_failing": governs == "material",
    }


def _default_wrap_efficiency(wraps: int) -> float:
    """How much of the theoretical section a rolled tube actually develops.

    The bench tests found a tube needs about three wraps before it holds real
    load -- below that the roll is still loose enough that the layers slide
    past each other instead of acting as one section.
    """
    return {1: 0.15, 2: 0.45, 3: 0.80, 4: 0.90, 5: 0.95}.get(int(wraps), 1.0)


# ---------------------------------------------------------------------------
# Torsion
# ---------------------------------------------------------------------------

@dataclass
class TorsionResult:
    j_open: float
    j_closed: float
    stiffness_ratio: float
    applied_torque_lbin: float
    twist_open_deg: float
    twist_closed_deg: float
    twist_actual_deg: float
    frame_spacing_in: float
    verdict: str


def torsion(mesh: HullMesh, wet_factor: float = 1.0,
            p: Params = NOMINAL) -> TorsionResult:
    """Open trough versus closed box, and how far she actually twists.

    St-Venant torsion constant for a thin-walled OPEN section is

        J = sum(s * t^3) / 3

    which is tiny because it goes as thickness CUBED. Close the same section
    with decks and it becomes a Bredt single-cell box,

        J = 4 * A_enclosed^2 / integral(ds / t)

    which goes as enclosed AREA squared. For a hull this size that is three to
    four orders of magnitude more stiffness from the same material, and it is
    the single highest-leverage structural decision on the boat.
    """
    d = mesh.design
    t = p.board_caliper_in * d.skin_layers

    mid = mesh.n_stations // 2
    b_bot = 2.0 * mesh.half_bottom[mid]
    b_gun = 2.0 * mesh.half_beam[mid]
    h = float(mesh.gunwale_z[mid] - mesh.keel_z[mid])

    depth_frac = np.clip(d.chine_frac, 1e-3, 1.0)
    flare_run = 0.5 * (b_gun - b_bot)
    flare_rise = depth_frac * h
    side_len = math.hypot(flare_rise, flare_run) + max(h - flare_rise, 0.0)

    girth = b_bot + 2.0 * side_len
    j_open = girth * t ** 3 / 3.0

    # Enclosed area of the closed cell: the trapezoid of the section.
    area_enc = 0.5 * (b_bot + b_gun) * h
    perim = girth + b_gun
    j_closed = 4.0 * area_enc ** 2 / (perim / t)

    # Applied torque. Two paddlers stroking on opposite sides make a couple;
    # when they are out of phase the couple is at its worst. Blade force of
    # about 20 lbf at the boat's half-beam is a fair peak for this crew.
    lever = 0.5 * b_gun
    blade_force = 20.0
    torque = 2.0 * blade_force * lever

    # Diagonal tape carries the shear flow the board alone is poor at. Cutting
    # the diagonal coverage to save tape buys the saving back in twist, which
    # is exactly the trade the sweep should be allowed to see.
    cover = float(np.clip((d.tape_diag_density + d.tape_diag_opposing) / 1.33,
                          0.0, 1.0))
    tape_shear = p.tape_shear_floor + (1.0 - p.tape_shear_floor) * cover

    g = p.g_board_psi * wet_factor * tape_shear
    span = d.length

    # WARPING TORSION, WITHOUT WHICH THE OPEN CASE IS NONSENSE
    #
    # Pure St-Venant torsion of an open thin-walled section goes as thickness
    # cubed and is essentially zero -- put the numbers in and an undecked
    # cardboard hull twists through hundreds of radians, which is obviously
    # not what happens. What actually stops it is warping restraint: the two
    # gunwale tubes bend in opposite directions in plan, like the flanges of
    # an I-beam resisting a couple, and for an open channel that mechanism
    # dominates St-Venant completely.
    #
    # Treating each gunwale as a beam that must deflect (beam/2)*theta:
    #     GJ_equivalent = 24 * E * I_gunwale * beam^2 / L^2
    #
    # This is why gunwale tubes are structure and not trim, and why running
    # them full length in one piece matters more than their diameter.
    t_gun = d.gunwale_tube_wraps * p.board_caliper_in
    r_gun = max(0.5 * d.gunwale_tube_dia - 0.5 * t_gun, 0.05)
    i_gun = math.pi * r_gun ** 3 * t_gun
    e_gun = 0.5 * (p.e_stiff_psi + p.e_soft_psi) * wet_factor
    gj_warp = 24.0 * e_gun * i_gun * b_gun ** 2 / max(span ** 2, 1e-9)

    gj_open = g * j_open + gj_warp
    gj_closed = g * j_closed + gj_warp
    twist_open = math.degrees(torque * span / max(gj_open, 1e-9))
    twist_closed = math.degrees(torque * span / max(gj_closed, 1e-9))
    twist_actual = twist_closed if d.decked_ends else twist_open

    spacing = d.length / (d.n_frames + 1)

    if twist_actual < 2.0:
        verdict = "stiff"
    elif twist_actual < 8.0:
        verdict = "noticeable twist, still controllable"
    elif twist_actual < 25.0:
        verdict = "wrings visibly; seams will start working open"
    else:
        verdict = "no torsional integrity: she folds"

    return TorsionResult(
        j_open=j_open, j_closed=j_closed,
        stiffness_ratio=gj_closed / max(gj_open, 1e-12),
        applied_torque_lbin=torque,
        twist_open_deg=twist_open, twist_closed_deg=twist_closed,
        twist_actual_deg=twist_actual,
        frame_spacing_in=spacing, verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Hull girder
# ---------------------------------------------------------------------------

@dataclass
class GirderResult:
    max_moment_lbin: float
    moment_at_in: float
    section_modulus_in3: float
    stress_psi: float
    allowable_psi: float
    utilisation: float
    sag_in: float
    verdict: str


def hull_girder(mesh: HullMesh, section_area: np.ndarray, hull_weight: float,
                wet_factor: float = 1.0, p: Params = NOMINAL) -> GirderResult:
    """The hull as a beam: two paddlers on distributed buoyancy.

    Buoyancy pushes up along the whole hull in proportion to immersed section
    area. Weight comes down mostly at two points where the paddlers kneel.
    The difference is a distributed load, and integrating it twice gives the
    bending moment the cardboard has to carry. For a boat loaded in the middle
    and buoyed at the ends this is a SAG, and the compression lands in the
    bottom panel -- the same panel that is already wet.
    """
    d = mesh.design
    x = mesh.x

    buoy = C.RHO_LB_IN3 * section_area                # lb per inch of length

    # Hull's own weight spread in proportion to section girth.
    girth = 2.0 * (mesh.half_bottom + (mesh.gunwale_z - mesh.keel_z))
    g_tot = float(np.trapezoid(girth, x))
    w_hull = hull_weight * girth / max(g_tot, 1e-9)

    load = buoy - w_hull                              # lb/in, up positive

    # Crew as narrow distributed patches -- a kneeling paddler is ~16 in long,
    # not a point, and pretending otherwise puts a false spike in the moment.
    for w, xf in zip(d.crew_weights, d.crew_x_frac):
        x0 = xf * d.length
        patch = np.exp(-0.5 * ((x - x0) / 8.0) ** 2)
        patch /= max(float(np.trapezoid(patch, x)), 1e-9)
        load -= w * patch

    paddles = d.n_crew * C.PADDLE_WEIGHT_LB
    load -= paddles / max(d.length, 1e-9)

    # Remove any residual net force / moment from discretisation so the beam
    # is in equilibrium and the moment closes to zero at the far end.
    load -= float(np.trapezoid(load, x)) / max(d.length, 1e-9)
    shear = np.concatenate([[0.0], np.cumsum(np.diff(x) * 0.5 * (load[1:] + load[:-1]))])
    shear -= np.linspace(0.0, shear[-1], shear.size)
    moment = np.concatenate([[0.0], np.cumsum(np.diff(x) * 0.5 * (shear[1:] + shear[:-1]))])
    moment -= np.linspace(0.0, moment[-1], moment.size)

    i_peak = int(np.argmax(np.abs(moment)))
    m_max = float(abs(moment[i_peak]))

    # Section modulus of the open U at midship, about the horizontal axis.
    mid = mesh.n_stations // 2
    t = p.board_caliper_in * d.skin_layers
    b_bot = 2.0 * mesh.half_bottom[mid]
    h = float(mesh.gunwale_z[mid] - mesh.keel_z[mid])

    # Thin-walled open channel: a bottom, two side walls, and the two gunwale
    # tubes sitting right at the top.
    #
    # The tubes are not trim. They sit at the maximum distance from the
    # neutral axis, where area counts for its distance SQUARED, so a pair of
    # rolled tubes contributes far more to the girder than the same cardboard
    # spread anywhere else. They are the top flange of the beam, the same way
    # they are the flanges that resist twist in the torsion calculation.
    a_bot = b_bot * t
    a_side = 2.0 * h * t

    t_gun = d.gunwale_tube_wraps * p.board_caliper_in
    r_gun = max(0.5 * d.gunwale_tube_dia - 0.5 * t_gun, 0.05)
    a_gun = 2.0 * (2.0 * math.pi * r_gun * t_gun)
    i_gun_own = 2.0 * (math.pi * r_gun ** 3 * t_gun)
    z_gun = h                                     # tubes ride on the sheer

    a_tot = a_bot + a_side + a_gun
    z_na = (a_side * 0.5 * h + a_gun * z_gun) / max(a_tot, 1e-9)
    i_sec = (a_bot * z_na ** 2
             + 2.0 * (t * h ** 3 / 12.0 + h * t * (0.5 * h - z_na) ** 2)
             + i_gun_own + a_gun * (z_gun - z_na) ** 2)
    c_max = max(z_na, h - z_na)
    z_mod = i_sec / max(c_max, 1e-9)

    # Longitudinal compression runs ACROSS the flutes when the skin is wrapped
    # the easy way (flutes around the girth), which is the weak direction.
    # Flute direction only means anything for a corrugated board. A solid
    # sheet has mild fibre anisotropy instead, and you cannot choose which
    # way it faces on a roll, so the softer direction is assumed.
    if not p.is_corrugated:
        sigma_allow = p.sigma_across_flutes
    elif d.flute_direction == "longitudinal":
        sigma_allow = p.sigma_along_flutes
    else:
        sigma_allow = p.sigma_across_flutes
    sigma_allow *= wet_factor

    stress = m_max / max(z_mod, 1e-9)
    util = stress / max(sigma_allow, 1e-9)

    e_eff = (p.e_stiff_psi if (p.is_corrugated
                               and d.flute_direction == "longitudinal")
             else p.e_soft_psi) * wet_factor
    sag = m_max * d.length ** 2 / (9.6 * max(e_eff * i_sec, 1e-9))

    if util < 0.4:
        verdict = "ample"
    elif util < 0.7:
        verdict = "adequate"
    elif util < 1.0:
        verdict = "marginal: no reserve for a hard landing"
    else:
        verdict = "OVERSTRESSED: the bottom creases amidships"

    return GirderResult(
        max_moment_lbin=m_max, moment_at_in=float(x[i_peak]),
        section_modulus_in3=z_mod, stress_psi=stress,
        allowable_psi=sigma_allow, utilisation=util,
        sag_in=sag, verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _plate_deflection(q_psi: float, span_a: float, span_b: float,
                      d_flex: float, et: float) -> tuple[float, bool, float, float]:
    """Centre deflection of a two-way panel, including membrane action.

    TWO THINGS THAT ARE EASY TO GET WRONG HERE, AND BOTH MATTER

    1. Small-deflection plate theory (w = 5qL^4/384D) only holds while the
       deflection stays under about the panel's own thickness. A bare
       cardboard bottom blows straight past that -- the linear formula
       predicts inches of sag -- at which point it has stopped behaving like
       a plate and started behaving like a membrane, carrying load in tension.
       Membrane stiffening is strongly non-linear, so leaving it out makes a
       cardboard bottom look hopeless when in practice it sags and then holds.

    2. The bottom spans BOTH ways: across the boat between the chines, and
       along it between the ring frames. The load splits between the two, and
       it splits steeply -- the share carried in each direction goes as the
       fourth power of the OTHER span. With frames closer together than the
       boat is wide, most of the load goes to the frames, not the chines.
       Attributing all of it to the shorter span and then reacting it at the
       chine (which is the longer one) overstates the chine load several
       times over.

    So: load shares from the fourth-power rule, then

        q = (384*D/5)*(1/a^4 + 1/b^4)*w  +  (64/3)*E*t*(1/a^4 + 1/b^4)*w^3

    with the membrane coefficient following from the parabolic-cable result
    (strain = (8/3)(w/L)^2), so the tension reported below is consistent with
    the deflection rather than being a second, separate approximation.

    Returns (deflection, membrane_regime, tension_across_a, tension_across_b)
    where the tensions are in lb per inch of the edge that reacts them.
    """
    if q_psi <= 0 or min(span_a, span_b) <= 0:
        return 0.0, False, 0.0, 0.0

    inv4 = 1.0 / span_a ** 4 + 1.0 / span_b ** 4
    k_bend = (384.0 / 5.0) * max(d_flex, 1e-9) * inv4
    k_memb = (64.0 / 3.0) * max(et, 1e-9) * inv4

    lo, hi = 0.0, min(span_a, span_b)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if k_bend * mid + k_memb * mid ** 3 < q_psi:
            lo = mid
        else:
            hi = mid
    w = 0.5 * (lo + hi)

    linear = q_psi / max(k_bend, 1e-12)
    membrane = bool(linear > 2.0 * w)

    # Membrane tension in each direction: N = (8/3) * E*t * (w/L)^2.
    n_a = (8.0 / 3.0) * et * (w / span_a) ** 2
    n_b = (8.0 / 3.0) * et * (w / span_b) ** 2
    return w, membrane, n_a, n_b


@dataclass
class PanelResult:
    water_pressure_psi: float
    span_in: float
    deflection_water_in: float
    deflection_knee_in: float
    deflection_linear_in: float
    membrane_regime: bool
    chine_tape_tension_lb_in: float
    chine_tape_utilisation: float
    knee_pressure_psi: float
    flat_crush_ok: bool
    frames_needed_for_half_inch: int
    verdict: str


def panels(mesh: HullMesh, draft_in: float, wet_factor: float = 1.0,
           p: Params = NOMINAL) -> PanelResult:
    """Bottom panel between frames: water pushing up, a knee pushing down.

    Treated as a one-way strip on the SHORTER of its two spans -- between the
    chines, or between the ring frames -- because that is the direction it
    will actually carry load. Simply supported, which is honest: a taped chine
    is nowhere near a fixed edge.

    What usually comes out of this is that the bare panel is far too floppy in
    bending, deflects well past its own thickness, and therefore stops acting
    as a plate at all. Past that point it is carrying load as a tension
    membrane hanging off the chine tape, which is not a load path anyone
    designed and not one the tape enjoys.
    """
    d = mesh.design
    mid = mesh.n_stations // 2
    b_bot = 2.0 * mesh.half_bottom[mid]            # transverse span, chine to chine
    frame_spacing = d.length / (d.n_frames + 1)    # longitudinal span, frame to frame
    span = min(b_bot, frame_spacing)               # reported as the governing span

    q_water = C.RHO_LB_IN3 * max(draft_in, 0.0)    # psi, uniform

    layers = d.skin_layers + (1 if d.bottom_doubler else 0)
    # Crossed layers stack as a thicker plate, so rigidity grows faster than
    # linearly -- the classic reason to cross the flutes on a doubler.
    # Crossed layers stack better than linearly for corrugated (the classic
    # reason to cross a doubler's flutes); for solid board extra layers just
    # add thickness, so the exponent is the plain cubic of a thicker plate.
    stack_exp = 2.2 if p.is_corrugated else 3.0
    d_eff = p.board_d_soft_lbin * layers ** stack_exp * wet_factor
    e_board = p.e_soft_psi
    if p.is_corrugated and d.flute_direction == "longitudinal":
        d_eff *= p.board_d_stiff_lbin / p.board_d_soft_lbin
        e_board = p.e_stiff_psi
    et = e_board * p.board_caliper_in * layers * wet_factor

    w_water, memb_water, _, _ = _plate_deflection(q_water, b_bot,
                                                  frame_spacing, d_eff, et)

    # A kneeling paddler puts most of their weight on a small patch. Smeared
    # over the panel as an equivalent pressure so the same membrane-corrected
    # solve applies -- a point load on a membrane is a much harder problem
    # and the extra rigour would not change a decision here.
    heaviest = max(d.crew_weights) if d.crew_weights else 0.0
    knee_load = 0.45 * heaviest
    pad_area = 18.0 * 24.0 * 0.5 if d.bottom_doubler else 8.0 * 10.0
    q_knee = knee_load / max(b_bot * frame_spacing, 1e-9) * 0.6
    w_both, memb, n_chine, _n_frame = _plate_deflection(
        q_water + q_knee, b_bot, frame_spacing, d_eff, et)
    w_knee = max(w_both - w_water, 0.0)

    knee_psi = knee_load / max(pad_area, 1e-9)
    flat_crush_allow = p.flat_crush_psi * wet_factor

    total = w_both
    linear = ((q_water + q_knee) * 5.0 / (384.0 * max(d_eff, 1e-9))
              / (1.0 / b_bot ** 4 + 1.0 / frame_spacing ** 4))

    # The membrane tension running across the boat has to be reacted at the
    # chine, by tape. This is the load path nobody draws and the one that
    # actually lets go. Tape along the chine is loaded across its width, the
    # weaker direction for cloth tape, hence the 0.7. Layers share it.
    tape_n = n_chine
    layers_tape = max(int(d.chine_tape_layers), 1)
    tape_allow = p.tape_tensile_lb_in * 0.7 * layers_tape * wet_factor
    tape_util = tape_n / max(tape_allow, 1e-9)

    # How many frames would hold the water-pressure deflection to half an inch?
    need = 0
    if q_water > 1e-9 and d_eff > 0:
        for n_try in range(0, 41):
            w_try, *_ = _plate_deflection(q_water + q_knee, b_bot,
                                          d.length / (n_try + 1), d_eff, et)
            if w_try <= 0.5:
                need = n_try
                break
        else:
            need = 40

    if tape_util >= 1.0:
        verdict = ("bottom is hanging from the chine tape and the tape is "
                   "over its tensile strength: this seam opens")
    elif total < 0.25:
        verdict = "firm underfoot"
    elif total < 0.75:
        verdict = "visible oil-canning, acceptable"
    elif memb:
        verdict = ("panel has left bending behind and is carrying load as a "
                   "membrane: add frames or a doubler")
    else:
        verdict = "soft bottom; add frames"

    return PanelResult(
        water_pressure_psi=q_water, span_in=span,
        deflection_water_in=w_water, deflection_knee_in=w_knee,
        deflection_linear_in=linear, membrane_regime=memb,
        chine_tape_tension_lb_in=tape_n, chine_tape_utilisation=tape_util,
        knee_pressure_psi=knee_psi,
        flat_crush_ok=knee_psi < flat_crush_allow,
        frames_needed_for_half_inch=need, verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------

@dataclass
class StructureResult:
    torsion: TorsionResult
    girder: GirderResult
    panel: PanelResult
    gunwale: dict
    joint_multiplier: float
    joint_source: str
    score: float = 0.0
    flags: list = field(default_factory=list)


def evaluate_structure(mesh: HullMesh, section_area: np.ndarray,
                       hull_weight: float, draft_in: float,
                       joint_data: dict | None = None,
                       p: Params = NOMINAL,
                       wet_factor: float | None = None) -> StructureResult:
    d = mesh.design
    # Strength retained by the time she crosses the line, not a fixed
    # knockdown: the hull is nearly dry at the gun and weakest at the finish.
    wf = d.wet_factor if wet_factor is None else wet_factor

    tor = torsion(mesh, wf, p)
    gir = hull_girder(mesh, section_area, hull_weight, wf, p)
    pan = panels(mesh, draft_in, wf, p)
    gun = tube_capacity(d.gunwale_tube_dia, d.gunwale_tube_wraps, joint_data, p)
    mult, src = joint_strength(d.joint_type, joint_data)

    flags = []
    if gir.utilisation >= 1.0:
        flags.append("hull girder overstressed")
    if tor.twist_actual_deg > 8.0:
        flags.append("excessive torsional twist")
    if pan.membrane_regime:
        flags.append("bottom panel in membrane regime")
    if pan.chine_tape_utilisation >= 1.0:
        flags.append("chine tape over its tensile strength")
    if not pan.flat_crush_ok:
        flags.append("kneeling load crushes the board flat")
    if not gun["warns_before_failing"]:
        flags.append("gunwale tube buckles without warning")
    if p.is_corrugated and d.flute_direction == "longitudinal":
        flags.append("longitudinal flutes will not wrap the chine without crushing")

    # A single 0-1 structural health number, for ranking. Deliberately blunt;
    # the individual numbers above are what you act on.
    s_gir = float(np.clip(1.0 - gir.utilisation, 0.0, 1.0))
    s_tor = float(np.clip(1.0 - tor.twist_actual_deg / 15.0, 0.0, 1.0))
    s_pan = float(np.clip(1.0 - (pan.deflection_water_in + pan.deflection_knee_in) / 1.5,
                          0.0, 1.0))
    s_tape = float(np.clip(1.0 - pan.chine_tape_utilisation, 0.0, 1.0))
    score = ((0.32 * s_tor + 0.28 * s_gir + 0.20 * s_pan + 0.20 * s_tape)
             * float(np.clip(mult, 0.2, 1.2)))

    return StructureResult(
        torsion=tor, girder=gir, panel=pan, gunwale=gun,
        joint_multiplier=mult, joint_source=src,
        score=score, flags=flags,
    )
