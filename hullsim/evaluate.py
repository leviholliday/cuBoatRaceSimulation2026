"""
evaluate(design) -> everything.

One function, one design, one complete answer. This is the layer the sweep
driver calls a few thousand times and the report layer calls once, so the two
can never disagree about what a design is worth.

It returns a flat dict of scalars (for tables, sorting and charts) with the
rich objects tucked under `detail` (for the single-design report). Flat first
because ninety percent of the use is ranking and plotting, and a flat row is
what both of those want.

ORDER OF OPERATIONS, AND WHY THERE IS NO CIRCULARITY

    design -> mesh -> material take-off -> hull weight -> mass properties
           -> equilibrium -> righting arms -> resistance -> race time
           -> seakeeping -> swamping reserve -> structure -> score

Hull weight comes out of the take-off, which depends only on the geometry, so
it is known before the boat is ever floated. That is why none of this needs
iterating, and it is also why weighing your actual cardboard is the highest-
value measurement available: everything downstream of the take-off inherits
its error.
"""

from __future__ import annotations

import math

import numpy as np

from . import constants as C
from .crew import paddle_reach_efficiency, resolve_crew, spacing_check
from .design import HullDesign
from .geometry import (build_mesh, lateral_plane, panel_warp,
                       tracking_efficiency, wetted_surface)
from .hydro import (apply_upsetting, clip_sections, free_surface_loss,
                    inclining_prediction, mass_properties, righting_arms,
                    solve_equilibrium)
from .materials import take_off
from .params import NOMINAL, Params
from .resistance import drag_table, resistance, simulate_race
from .scoring import Limits, Rubric, check_constraints, score_design
from .seakeeping import (boarding_check, roll_response, shipping_risk,
                         swamping_cascade, swamping_reserve_fast)
from .structure import evaluate_structure, load_joint_data

# Coarser heel grid for sweeps: enough to locate the peak and the
# downflooding angle, half the cost of the full grid.
SWEEP_ANGLES = (0, 3, 6, 10, 15, 20, 26, 33, 40, 50)


def evaluate(design: HullDesign, *, limits: Limits | None = None,
             rubric: Rubric | None = None, joint_data: dict | None = None,
             detail: bool = True, water_depth_ft: float = 0.0,
             residuary_scale: float | None = None,
             p: Params = NOMINAL) -> dict:
    """Evaluate one hull completely.

    `detail=False` is the sweep path: coarser heel grid, coarser swamping
    cascade, no per-item cut list. Same physics, about a third of the cost.
    """
    limits = limits or Limits()
    rubric = rubric or Rubric()
    joint_data = joint_data if joint_data is not None else load_joint_data()
    if residuary_scale is not None:
        p = p.with_(residuary_scale=residuary_scale)

    problems = design.validate()
    if problems:
        return _infeasible(design, problems)

    # --- who is aboard, and how they are sitting ---------------------------
    # Centre-of-gravity height and shoulder height both come from stature and
    # posture, so they cannot disagree. This is what makes posture a trade
    # rather than a free win: crouching lowers the CG and lowers the shoulders
    # toward the rail at the same time.
    cr = resolve_crew(design)
    if design.crew_kg != cr["crew_kg"]:
        design = design.with_(crew_kg=cr["crew_kg"])

    mesh = build_mesh(design)

    # --- material and weight ----------------------------------------------
    mat = take_off(mesh, p=p)
    hull_w = (design.hull_weight_override if design.hull_weight_override
              is not None else mat.hull_weight_lb) + design.extra_weight_lb

    # --- float her ---------------------------------------------------------
    # Water already aboard counts twice: as weight, which costs freeboard,
    # and as a free surface, which costs righting arm. Both are applied.
    floor_area = float(np.trapezoid(2.0 * mesh.half_bottom, mesh.x))
    w_aboard = design.water_aboard_gal * 8.345
    depth_aboard = (w_aboard / C.RHO_LB_IN3) / max(floor_area, 1e-9)

    mp = mass_properties(mesh, hull_w, water_weight=w_aboard,
                         water_depth=depth_aboard)
    eq = solve_equilibrium(mesh, mp)

    angles = SWEEP_ANGLES if not detail else C.HEEL_ANGLES_DEG
    sc_hull = righting_arms(mesh, mp, eq, angles)

    fs_now = free_surface_loss(mesh, eq, depth_aboard, design.free_water_lanes,
                               weight_lb=mp.weight) if w_aboard > 0 else 0.0
    sc = apply_upsetting(sc_hull, fs_now, design.build_list_in)

    warp = float(panel_warp(mesh).max())
    # Judged against what THIS material will take. Corrugated darts and
    # crushes its flutes under forced double curvature; a thin solid sheet
    # simply bends, so the same geometry is fine on one and unbuildable on
    # the other.
    warp_ratio = warp / max(p.warp_tolerance, 1e-9)
    s_wet = wetted_surface(mesh, eq.waterline_z)

    # --- resistance and the race -------------------------------------------
    def drag_at(v):
        return resistance(
            max(v, 1e-3), wetted_in2=s_wet, lwl_in=eq.lwl,
            beam_wl_in=eq.beam_wl, draft_in=eq.draft, volume_in3=eq.volume,
            displacement_lb=mp.weight, cb=eq.cb, cp=eq.cp,
            chop_in=design.chop_height, depth_ft=water_depth_ft,
            headwind_mph=design.headwind_mph,
            freeboard_in=eq.freeboard_min, p=p,
        )

    # Directional stability: rocker lifts the lateral plane out of the water,
    # the boat wanders, and every correction stroke is power not going
    # forwards. Nothing in the drag calculation knows about this.
    lp = lateral_plane(mesh, eq.waterline_z)
    track = tracking_efficiency(lp["tracking_index"], design.crew_power_w)

    # Can they get a stroke over the side? The gunwale amidships is what the
    # paddler has to reach across, and a crouched short paddler in a deep hull
    # runs out of room. Without this the model recommends folding as low as a
    # body goes, because every other term rewards a low centre of gravity.
    gun_z = float(mesh.gunwale_z[mesh.n_stations // 2])
    reaches = [paddle_reach_efficiency(g.shoulder_z, gun_z)
               for g in cr["geometries"]]
    reach_eff = float(np.mean([r["efficiency"] for r in reaches]))
    reach_clearance = float(min(r["clearance_in"] for r in reaches))

    spacing = spacing_check(design.length, design.crew_x_frac, cr["geometries"])

    shaft = (float(sum(design.crew_power_w))
             * track["efficiency"] * reach_eff)
    # Two paddlers digging in from a standstill. Peak blade force is higher,
    # but this is the stroke-cycle average, which is what accelerates a boat.
    max_thrust = p.max_thrust_per_paddler_lb * design.n_crew
    drag_lookup = drag_table(drag_at)
    race = simulate_race(drag_lookup, displacement_lb=mp.weight,
                         shaft_power_w=shaft, max_thrust_lb=max_thrust,
                         course_m=C.COURSE_M, p=p, pacing=design.pacing,
                         anaerobic_fraction=design.anaerobic_fraction)
    finish_drag = drag_at(max(race.v_at_finish, 1e-3))

    # --- seakeeping --------------------------------------------------------
    sea = shipping_risk(mesh, eq, sc, mp, chop_in=design.chop_height,
                        lean_in=design.paddle_lean_in,
                        speed_ms=max(race.mean_v, 0.1),
                        crossing_s=race.time_s)

    # Roll from the chop, at race speed and stopped. These are very different
    # numbers: moving, the encounters are too fast for the hull to respond;
    # stopped at the line they slow to the wave period, which for a boat this
    # size can land on resonance -- and stopped at the line is exactly when
    # somebody is climbing in.
    roll_racing = roll_response(eq, design.chop_height,
                                max(race.mean_v, 0.1), eq.gm_t)
    roll_stopped = roll_response(eq, design.chop_height, 0.0, eq.gm_t)

    board = boarding_check(mesh, eq, sc, mp)
    # Boarding happens stationary, so the wave roll that matters there is the
    # stopped one, and it adds to the heel from the person stepping in.
    board_total_heel = board.heel_deg + roll_stopped.roll_amplitude_deg
    board_fb = eq.freeboard_min - 0.5 * eq.beam_wl * math.sin(
        math.radians(min(board_total_heel, 89.0)))

    if detail:
        swamp = swamping_cascade(mesh, hull_w, lanes=design.free_water_lanes,
                                 max_gallons=60.0, step_gal=1.0)
        swamp_gal = swamp.gallons_reserve
    else:
        swamp = None
        swamp_gal = swamping_reserve_fast(mesh, hull_w,
                                          lanes=design.free_water_lanes)

    # Effective GM: a standardised stress test, one gallon sloshing about,
    # which is a far more honest number to design to than the flat-calm
    # empty-boat GM. Uses whichever is worse, the probe or what is actually
    # aboard, and charges for any built-in list as well.
    d1 = (max(design.water_aboard_gal, 1.0) * 8.345 / C.RHO_LB_IN3) \
        / max(floor_area, 1e-9)
    fs_loss = free_surface_loss(mesh, eq, d1, design.free_water_lanes,
                                weight_lb=mp.weight)
    gm_eff = eq.gm_t - fs_loss - design.build_list_in

    # --- structure ---------------------------------------------------------
    # The hull is nearly dry at the gun and weakest at the finish, so the
    # structural checks use the strength left at the END of the race.
    # No paint is allowed under these rules, so tape IS the waterproofing.
    # How fast she goes soft is set by how much of the immersed hull got
    # covered -- which makes the tape budget a structural AND a sealing
    # decision competing for one roll.
    minutes_afloat = design.pre_race_soak_min + race.time_s / 60.0
    soak_floor, soak_tau = C.soak_params(mat.bottom_tape_coverage)
    wet_now = C.soak_factor(minutes_afloat, floor=soak_floor, tau_min=soak_tau)

    area, _, _ = clip_sections(mesh.verts, 0.0, 1.0, eq.waterline_z)
    st = evaluate_structure(mesh, area, hull_w, eq.draft, joint_data, p,
                            wet_factor=wet_now)

    # --- flat row ----------------------------------------------------------
    # Every design field, so a CSV row can rebuild the exact hull that was
    # scored. The charts only use a handful of these, but a report that
    # silently re-evaluated a different boat would be worse than no report.
    row = dict(design.to_row())
    row.update({
        # weight
        "hull_weight_lb": hull_w, "all_up_lb": mp.weight, "kg_in": mp.kg,
        # hydrostatics
        "floats": bool(eq.solved), "draft_in": eq.draft,
        "draft_frac": eq.draft / max(design.side_height, 1e-9),
        "freeboard_in": eq.freeboard_min, "trim_deg": eq.trim_deg,
        "gm_in": eq.gm_t, "gm_effective_in": gm_eff,
        "free_surface_loss_in": fs_loss,
        "bm_in": eq.bm_t, "kb_in": eq.kb,
        "lwl_in": eq.lwl, "beam_wl_in": eq.beam_wl,
        "cb": eq.cb, "cp": eq.cp, "volume_in3": eq.volume,
        "wetted_in2": s_wet,
        # stability
        "gz_max_in": sc.gz_max, "heel_at_gz_max": sc.heel_at_gz_max,
        "downflood_deg": sc.downflood_deg, "vanishing_deg": sc.vanishing_deg,
        "gz_area": sc.area_to_downflood, "usable_heel_deg": sc.usable_deg,
        "heel_at_rest_deg": sc.heel_at_rest,
        "gz_demand_in": sea.gz_demand_in,
        "gz_margin_ratio": sc.gz_max / max(sea.gz_demand_in, 1e-9),
        # speed
        "time_s": race.time_s, "terminal_v_ms": race.terminal_v,
        "mean_v_ms": race.mean_v, "froude": finish_drag.froude,
        "drag_lb": finish_drag.r_total,
        "drag_friction_lb": finish_drag.r_friction,
        "drag_residuary_lb": finish_drag.r_residuary,
        "form_factor": finish_drag.form,
        "accel_penalty_s": race.accel_penalty_s,
        "hydro_power_w": finish_drag.power_w,
        "v_peak_ms": race.v_peak,
        "power_start_w": race.power_start_w,
        "power_finish_w": race.power_finish_w,
        "fade_frac": race.fade_frac,
        "wprime_spent_frac": race.wprime_spent_frac,
        "pinned_at_cp_s": race.pinned_at_cp_s,
        # seakeeping
        "paddle_heel_deg": sea.paddle_heel_deg,
        "heel_freeboard_loss_in": sea.heel_freeboard_loss_in,
        "effective_freeboard_in": sea.effective_freeboard_in,
        "expected_shipping_events": sea.expected_shipping_events,
        "p_ship_crossing": sea.p_ship_crossing,
        "swamp_reserve_gal": swamp_gal,
        "boarding_heel_deg": board.heel_deg,
        "boarding_freeboard_in": board.freeboard_left_in,
        "boarding_ok": board.survives,
        "boarding_heel_with_chop_deg": board_total_heel,
        "boarding_freeboard_with_chop_in": board_fb,
        # dynamic roll
        "roll_period_s": roll_racing.roll_period_s,
        "roll_racing_deg": roll_racing.roll_amplitude_deg,
        "roll_stopped_deg": roll_stopped.roll_amplitude_deg,
        "roll_tuning_stopped": roll_stopped.tuning_ratio,
        "roll_resonant_stopped": roll_stopped.resonant,
        # tracking
        "tracking_index": lp["tracking_index"],
        "tracking_efficiency": track["efficiency"],
        "tracking_rocker_loss": track["rocker_loss"],
        "tracking_asymmetry_loss": track["asymmetry_loss"],
        "effective_shaft_w": shaft,
        # crew
        "reach_efficiency": reach_eff,
        "reach_clearance_in": reach_clearance,
        "crew_spacing_in": spacing["gap_in"],
        "crew_spacing_ok": spacing["ok"],
        "crew_kg_mean": float(np.mean(cr["crew_kg"])),
        "shoulder_z_min": float(min(g.shoulder_z for g in cr["geometries"])),
        # soaking
        "minutes_afloat": minutes_afloat,
        "wet_factor_at_finish": wet_now,
        "bottom_tape_coverage": mat.bottom_tape_coverage,
        "soak_floor": soak_floor, "soak_tau_min": soak_tau,
        # structure
        "panel_warp": warp, "warp_ratio": warp_ratio,
        "material": p.material_name, "is_corrugated": p.is_corrugated,
        "board_caliper_in": p.board_caliper_in,
        "board_areal_lb_ft2": p.board_areal_lb_ft2,
        "chine_tape_util": st.panel.chine_tape_utilisation,
        "girder_utilisation": st.girder.utilisation,
        "girder_stress_psi": st.girder.stress_psi,
        "twist_deg": st.torsion.twist_actual_deg,
        "torsion_ratio": st.torsion.stiffness_ratio,
        "panel_deflection_in": (st.panel.deflection_water_in
                                + st.panel.deflection_knee_in),
        "frames_needed": st.panel.frames_needed_for_half_inch,
        "joint_multiplier": st.joint_multiplier,
        "structure_score": st.score,
        # materials
        "board_used_in2": mat.board_used_in2,
        "board_spare_frac": mat.board_spare_frac, "board_over": mat.board_over,
        "tape_used_in": mat.tape_used_in,
        "tape_spare_frac": mat.tape_spare_frac, "tape_over": mat.tape_over,
    })

    violations = check_constraints(row, limits)
    row["feasible"] = len(violations) == 0
    row["n_violations"] = len(violations)
    row["violations"] = "; ".join(violations)

    row.update(score_design(row, rubric))
    # expected_score stays clean and comparable across all designs, so it can
    # be plotted and read directly. ranking_score is what the sweep sorts on:
    # same number, with an unbridgeable penalty per violated constraint, so an
    # infeasible hull can never outrank a feasible one no matter how fast it is.
    row["ranking_score"] = row["expected_score"] - 1000.0 * len(violations)

    if detail:
        row["detail"] = {
            "design": design, "mesh": mesh, "mass": mp, "equilibrium": eq,
            "stability": sc, "race": race, "seakeeping": sea,
            "swamping": swamp, "structure": st, "materials": mat,
            "boarding": board, "tracking": track, "lateral_plane": lp,
            "crew": cr, "reaches": reaches, "spacing": spacing,
            "roll_racing": roll_racing, "roll_stopped": roll_stopped,
            "violations": violations,
            "drag_curve": [drag_at(v) for v in np.arange(0.2, 3.01, 0.1)],
            "inclining": [inclining_prediction(mp, eq.gm_t, w, lev)
                          for w in (15, 25) for lev in (12, 18)],
        }
    return row


def _infeasible(design: HullDesign, problems: list[str]) -> dict:
    """A row for a design that is not even geometrically coherent."""
    row = {k: float("nan") for k in (
        "hull_weight_lb", "all_up_lb", "kg_in", "draft_in", "draft_frac",
        "freeboard_in", "trim_deg", "gm_in", "gm_effective_in", "gz_max_in",
        "downflood_deg", "gz_area", "time_s", "terminal_v_ms", "froude",
        "drag_lb", "panel_warp", "girder_utilisation", "twist_deg",
        "panel_deflection_in", "swamp_reserve_gal", "structure_score",
        "board_spare_frac", "tape_spare_frac", "expected_score",
        "p_finish", "expected_shipping_events", "paddle_heel_deg",
    )}
    row.update({
        "name": design.name, "length": design.length,
        "bottom_width": design.bottom_width, "beam": design.beam,
        "side_height": design.side_height,
        "floats": False, "feasible": False,
        "board_over": True, "tape_over": True,
        "n_violations": len(problems), "violations": "; ".join(problems),
        "expected_score": -1e6,
    })
    return row
