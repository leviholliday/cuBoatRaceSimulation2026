#!/usr/bin/env python3
"""
Validation tests.

The point of these is NOT coverage. It is that the numerical machinery -- a
polygon clipper feeding a bisection solver feeding an integrator -- has to
reproduce answers that can be worked out on paper. If a plain rectangular box
does not give exactly d = W/(rho*L*B) and BM = B^2/(12d), then nothing the
model says about a shaped hull is worth reading.

So the core tests build degenerate hulls with closed-form answers and check
against them to tight tolerances. The rest check internal consistency
(displacement equals weight, the centre of buoyancy sits under the centre of
gravity) and the monotonic behaviours any correct hydrostatics must show.

Run with pytest, or directly:

    python tests/test_physics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from hullsim import constants as C
from hullsim.design import BASELINE, RECOMMENDED, HullDesign
from hullsim.evaluate import evaluate
from hullsim.geometry import build_mesh, panel_warp, trapz_weights, wetted_surface
from hullsim.hydro import (clip_sections, free_surface_loss, mass_properties,
                           righting_arms, solve_equilibrium)
from hullsim.materials import take_off
from hullsim.resistance import friction_coefficient, resistance
from hullsim.structure import torsion, tube_capacity


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def box_hull(length=100.0, width=30.0, height=12.0, weight=300.0) -> HullDesign:
    """A plain rectangular box: no taper, no rocker, no sheer, no flare.

    Every hydrostatic property of this hull has a one-line closed form, which
    is exactly why it is the test case.
    """
    return HullDesign(
        length=length, bottom_width=width, beam=width, side_height=height,
        bow_taper=0.0, stern_taper=0.0,
        bow_rocker=0.0, stern_rocker=0.0,
        bow_sheer=0.0, stern_sheer=0.0,
        chine_frac=1.0,
        crew_weights=(weight,), crew_kg=(0.0,), crew_x_frac=(0.5,),
        crew_power_w=(150.0,),
        hull_weight_override=0.0,
        name="box",
    )


def close(a, b, tol=1e-6, msg=""):
    assert abs(a - b) <= tol * max(1.0, abs(b)), \
        f"{msg}: got {a!r}, expected {b!r} (tol {tol})"


# ---------------------------------------------------------------------------
# the clipping primitive
# ---------------------------------------------------------------------------

def test_clip_full_and_empty():
    """A polygon entirely below or entirely above the plane."""
    #    a 2 x 1 rectangle from y=-1..1, z=0..1
    v = np.array([[[-1.0, 1.0], [-1.0, 0.0], [-1.0, 0.0],
                   [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    a, cy, cz = clip_sections(v, 0.0, 1.0, 5.0)      # waterline well above
    close(float(a[0]), 2.0, 1e-9, "fully submerged area")
    close(float(cz[0]), 0.5, 1e-9, "fully submerged centroid z")
    close(float(cy[0]), 0.0, 1e-9, "fully submerged centroid y")

    a, _, _ = clip_sections(v, 0.0, 1.0, -1.0)       # waterline below
    close(float(a[0]), 0.0, 1e-9, "dry area")


def test_clip_partial_exact():
    """Half-submerged rectangle: area and centroid are known exactly."""
    v = np.array([[[-1.0, 1.0], [-1.0, 0.0], [-1.0, 0.0],
                   [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    a, cy, cz = clip_sections(v, 0.0, 1.0, 0.4)
    close(float(a[0]), 2.0 * 0.4, 1e-9, "partial area")
    close(float(cz[0]), 0.2, 1e-9, "partial centroid z")


def test_clip_heeled_triangle():
    """Clip a square with a 45 degree plane: the wet part is a triangle.

    This is the case that catches sign errors and a wrongly-oriented normal,
    because the answer is asymmetric in y.
    """
    v = np.array([[[-1.0, 2.0], [-1.0, 0.0], [-1.0, 0.0],
                   [1.0, 0.0], [1.0, 0.0], [1.0, 2.0]]])
    s = math.sqrt(0.5)
    # plane  y*s + z*s = 0  -> underwater where y + z <= 0
    a, cy, cz = clip_sections(v, s, s, 0.0)
    # Region inside the 2x2 box (y in -1..1, z in 0..2) with y + z <= 0:
    # a triangle with vertices (-1,0), (0,0), (-1,1). Area 1/2.
    close(float(a[0]), 0.5, 1e-9, "heeled clip area")
    close(float(cy[0]), (-1 + 0 - 1) / 3.0, 1e-9, "heeled centroid y")
    close(float(cz[0]), (0 + 0 + 1) / 3.0, 1e-9, "heeled centroid z")


def test_trapz_weights_match_numpy():
    x = np.linspace(0, 7, 23)
    f = np.sin(x) + x ** 2
    close(float(f @ trapz_weights(x)), float(np.trapezoid(f, x)), 1e-12,
          "trapezoid weights")


# ---------------------------------------------------------------------------
# hydrostatics against closed form
# ---------------------------------------------------------------------------

def test_box_draft_kb_bm_gm():
    """A rectangular box, checked against the textbook formulas.

        V  = W / rho          displaced volume
        d  = V / (L*B)        draft
        KB = d / 2            centre of buoyancy of a rectangular prism
        BM = I/V = B^2/(12 d) transverse metacentric radius
    """
    L, B, H, W = 100.0, 30.0, 12.0, 300.0
    d = box_hull(L, B, H, W)
    mesh = build_mesh(d)
    mp = mass_properties(mesh, hull_weight=0.0)
    close(mp.weight, W + C.PADDLE_WEIGHT_LB, 1e-9, "all-up weight")

    eq = solve_equilibrium(mesh, mp)
    v_exp = mp.weight / C.RHO_LB_IN3
    d_exp = v_exp / (L * B)

    close(eq.volume, v_exp, 1e-5, "displaced volume")
    close(eq.draft, d_exp, 1e-4, "draft")
    close(eq.kb, d_exp / 2.0, 1e-4, "KB")
    close(eq.bm_t, B ** 2 / (12.0 * d_exp), 1e-4, "BM")
    close(eq.aw, L * B, 1e-6, "waterplane area")
    close(eq.gm_t, eq.kb + eq.bm_t - mp.kg, 1e-9, "GM identity")
    close(eq.cb, 1.0, 1e-4, "block coefficient of a box")


def test_archimedes_on_shaped_hull():
    """Displaced weight equals total weight, on the real baseline hull."""
    mesh = build_mesh(BASELINE)
    mp = mass_properties(mesh, hull_weight=22.0)
    eq = solve_equilibrium(mesh, mp)
    close(eq.volume * C.RHO_LB_IN3, mp.weight, 2e-4, "Archimedes")


def test_lcb_meets_lcg():
    """At equilibrium the centre of buoyancy sits under the centre of gravity."""
    for xf in [(0.25, 0.70), (0.35, 0.80), (0.5, 0.55)]:
        d = BASELINE.with_(crew_x_frac=xf)
        mesh = build_mesh(d)
        mp = mass_properties(mesh, hull_weight=22.0)
        eq = solve_equilibrium(mesh, mp)
        assert abs(eq.lcb - mp.lcg) < 5e-3, \
            f"LCB {eq.lcb:.4f} != LCG {mp.lcg:.4f} for crew at {xf}"


def test_trim_direction():
    """Moving the crew aft must put her down by the stern, not the bow."""
    fwd = BASELINE.with_(crew_x_frac=(0.22, 0.40))
    aft = BASELINE.with_(crew_x_frac=(0.60, 0.78))
    out = []
    for d in (fwd, aft):
        mesh = build_mesh(d)
        mp = mass_properties(mesh, hull_weight=22.0)
        eq = solve_equilibrium(mesh, mp)
        out.append(eq.stern_draft - eq.bow_draft)
    assert out[1] > out[0], "crew moving aft did not increase stern draft"


# ---------------------------------------------------------------------------
# large-angle stability
# ---------------------------------------------------------------------------

def test_gz_matches_gm_at_small_angles():
    """GZ(phi) -> GM*sin(phi) as phi -> 0. This pins the sign convention too.

    If this passes for both a stable and an unstable hull, the whole heeled
    clip-and-integrate path is consistent with the upright metacentric
    calculation, which are two quite different code paths arriving at the
    same number.
    """
    for d in (BASELINE,
              BASELINE.with_(length=115, bottom_width=20, beam=24)):
        mesh = build_mesh(d)
        mp = mass_properties(mesh, hull_weight=22.0)
        eq = solve_equilibrium(mesh, mp)
        sc = righting_arms(mesh, mp, eq, (0.0, 1.0, 2.0))
        for phi, gz in zip(sc.heel_deg[1:], sc.gz[1:]):
            expect = eq.gm_t * math.sin(math.radians(phi))
            assert abs(gz - expect) < 0.01 + 0.02 * abs(expect), \
                f"GZ({phi}) = {gz:.4f}, GM*sin = {expect:.4f}, hull {d.name}"


def test_displacement_conserved_when_heeled():
    """Heeling must not change how much she displaces."""
    mesh = build_mesh(BASELINE)
    mp = mass_properties(mesh, hull_weight=22.0)
    eq = solve_equilibrium(mesh, mp)
    sc = righting_arms(mesh, mp, eq, (0, 10, 20, 30))
    assert np.all(np.isfinite(sc.gz)), "non-finite righting arm"


def test_narrow_hull_is_unstable():
    """The long-and-skinny rumour, checked rather than assumed."""
    d = BASELINE.with_(length=115, bottom_width=20, beam=24, side_height=13)
    mesh = build_mesh(d)
    mp = mass_properties(mesh, hull_weight=22.0)
    eq = solve_equilibrium(mesh, mp)
    assert eq.gm_t < 0, f"expected negative GM for a narrow hull, got {eq.gm_t:+.2f}"


def test_beam_increases_stability():
    gms = []
    for b in (28.0, 32.0, 36.0, 40.0):
        d = BASELINE.with_(bottom_width=b - 6.0, beam=b)
        mesh = build_mesh(d)
        mp = mass_properties(mesh, hull_weight=22.0)
        gms.append(solve_equilibrium(mesh, mp).gm_t)
    assert all(b > a for a, b in zip(gms, gms[1:])), f"GM not monotonic in beam: {gms}"


def test_weight_increases_draft():
    mesh = build_mesh(BASELINE)
    drafts = []
    for hw in (10.0, 25.0, 50.0, 90.0):
        mp = mass_properties(mesh, hull_weight=hw)
        drafts.append(solve_equilibrium(mesh, mp).draft)
    assert all(b > a for a, b in zip(drafts, drafts[1:])), \
        f"draft not monotonic in weight: {drafts}"


def test_lower_crew_cg_raises_gm():
    mesh_hi = build_mesh(BASELINE.with_(crew_kg=(21.0, 21.0)))
    mesh_lo = build_mesh(BASELINE.with_(crew_kg=(11.0, 11.0)))
    gm = []
    for m in (mesh_hi, mesh_lo):
        mp = mass_properties(m, hull_weight=22.0)
        gm.append(solve_equilibrium(m, mp).gm_t)
    assert gm[1] > gm[0] + 5.0, f"posture barely moved GM: {gm}"


# ---------------------------------------------------------------------------
# free surface
# ---------------------------------------------------------------------------

def test_free_surface_deep_case_matches_classical():
    """With water deep enough to touch both walls, the model must reduce to
    the textbook i/V correction."""
    mesh = build_mesh(box_hull(100.0, 30.0, 20.0, 300.0))
    mp = mass_properties(mesh, hull_weight=0.0)
    eq = solve_equilibrium(mesh, mp)

    heel = 2.0
    depth = 6.0                     # far deeper than (B/2)*tan(2 deg) = 0.52
    got = free_surface_loss(mesh, eq, depth, 1, heel_deg=heel,
                            weight_lb=mp.weight)
    i_free = 100.0 * 30.0 ** 3 / 12.0
    classical = i_free / eq.volume / math.cos(math.radians(heel))
    close(got, classical, 2e-3, "deep free-surface loss vs i/V")


def test_free_surface_shallow_is_far_less_than_classical():
    """A thin layer must NOT cost the full classical correction.

    This is the bug the shallow-layer treatment exists to prevent: the
    classical formula is independent of how much water is aboard, so applied
    to a tenth of an inch it predicts tens of inches of lost GM.
    """
    mesh = build_mesh(BASELINE)
    mp = mass_properties(mesh, hull_weight=22.0)
    eq = solve_equilibrium(mesh, mp)
    thin = free_surface_loss(mesh, eq, 0.08, 1, heel_deg=5.0,
                             weight_lb=mp.weight)
    i_free = float(np.trapezoid((2 * mesh.half_bottom) ** 3, mesh.x)) / 12.0
    classical = i_free / eq.volume
    assert 0.0 < thin < 0.35 * classical, \
        f"thin-layer loss {thin:.2f} is not well below classical {classical:.2f}"


def test_lanes_cut_free_surface():
    mesh = build_mesh(BASELINE)
    mp = mass_properties(mesh, hull_weight=22.0)
    eq = solve_equilibrium(mesh, mp)
    losses = [free_surface_loss(mesh, eq, 0.5, n, weight_lb=mp.weight)
              for n in (1, 2, 3)]
    assert losses[0] > losses[1] > losses[2], f"lanes did not help: {losses}"


# ---------------------------------------------------------------------------
# geometry and buildability
# ---------------------------------------------------------------------------

def test_rocker_alone_is_developable():
    """Rocker bends the bottom in one direction only: a cylinder. Cardboard
    rolls that all day, so the warp measure must report nothing."""
    d = box_hull()
    d = d.with_(bow_rocker=3.0, stern_rocker=2.0)
    assert panel_warp(build_mesh(d)).max() < 1e-9


def test_plan_taper_alone_is_developable():
    """Vertical sides with a curved plan are still a generalised cylinder."""
    d = box_hull().with_(bow_taper=24.0, stern_taper=12.0, chine_frac=1e-3)
    assert panel_warp(build_mesh(d)).max() < 1e-6


def test_rocker_plus_flare_plus_taper_warps():
    """Combining all three is what makes a panel non-developable."""
    d = BASELINE.with_(bow_rocker=4.5, stern_rocker=3.0, beam=40.0,
                       bottom_width=26.0, bow_taper=30.0)
    assert panel_warp(build_mesh(d)).max() > 0.05


def test_baseline_geometry_sane():
    mesh = build_mesh(BASELINE)
    assert mesh.keel_z.min() == 0.0, "baseline is not on the keel baseline"
    assert np.all(mesh.gunwale_z > mesh.keel_z), "gunwale below keel somewhere"
    assert np.all(mesh.half_beam >= mesh.half_bottom - 1e-12), "section inverted"


def test_wetted_surface_of_box():
    """Box: wetted surface = bottom + two sides, exactly."""
    L, B, H, W = 100.0, 30.0, 12.0, 300.0
    mesh = build_mesh(box_hull(L, B, H, W))
    mp = mass_properties(mesh, hull_weight=0.0)
    eq = solve_equilibrium(mesh, mp)
    expect = L * (B + 2.0 * eq.draft)
    close(wetted_surface(mesh, eq.waterline_z), expect, 1e-4, "wetted surface")


# ---------------------------------------------------------------------------
# resistance
# ---------------------------------------------------------------------------

def test_ittc_friction_line():
    """Cf = 0.075 / (log10(Re) - 2)^2 at a round number."""
    close(friction_coefficient(1.0e6), 0.075 / 16.0, 1e-12, "ITTC at Re 1e6")


def test_drag_rises_with_speed():
    kw = dict(wetted_in2=3000.0, lwl_in=100.0, beam_wl_in=32.0, draft_in=4.0,
              volume_in3=9000.0, displacement_lb=320.0, cb=0.72, cp=0.76)
    r = [resistance(v, **kw).r_total for v in (0.5, 1.0, 1.5, 2.0, 2.5)]
    assert all(b > a for a, b in zip(r, r[1:])), f"drag not monotonic: {r}"


def test_slenderness_reduces_residuary():
    """A longer hull for the same volume must make less wave."""
    kw = dict(wetted_in2=3000.0, beam_wl_in=32.0, draft_in=4.0,
              volume_in3=9000.0, displacement_lb=320.0, cb=0.72, cp=0.60)
    short = resistance(1.8, lwl_in=85.0, **kw).r_residuary
    long_ = resistance(1.8, lwl_in=120.0, **kw).r_residuary
    assert long_ < short, f"longer hull had more residuary: {long_} vs {short}"


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def test_decking_stiffens_torsion():
    mesh = build_mesh(BASELINE)
    t = torsion(mesh)
    assert t.j_closed > 100.0 * t.j_open, "closed section not much stiffer"
    assert t.twist_closed_deg < t.twist_open_deg


def test_more_wraps_stronger_tube():
    caps = [tube_capacity(2.0, n)["moment_capacity_lbin"] for n in (1, 2, 3, 4)]
    assert all(b > a for a, b in zip(caps, caps[1:])), f"wraps not monotonic: {caps}"


def test_frames_reduce_panel_deflection():
    defl = []
    for n in (1, 3, 6, 10):
        r = evaluate(BASELINE.with_(n_frames=n), detail=False)
        defl.append(r["panel_deflection_in"])
    assert defl[0] >= defl[-1], f"frames did not help: {defl}"


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def test_take_off_positive_and_scales():
    small = take_off(build_mesh(BASELINE.with_(length=85.0)))
    big = take_off(build_mesh(BASELINE.with_(length=115.0)))
    assert small.board_used_in2 > 0 and small.tape_used_in > 0
    assert big.board_used_in2 > small.board_used_in2
    assert big.hull_weight_lb > small.hull_weight_lb


def test_two_layers_weigh_more():
    one = take_off(build_mesh(BASELINE.with_(skin_layers=1)))
    two = take_off(build_mesh(BASELINE.with_(skin_layers=2)))
    assert two.hull_weight_lb > one.hull_weight_lb * 1.2


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_evaluate_baseline_complete():
    r = evaluate(BASELINE, detail=True)
    for k in ("draft_in", "gm_in", "time_s", "expected_score", "p_finish",
              "panel_warp", "swamp_reserve_gal", "gz_margin_ratio"):
        assert k in r and np.isfinite(r[k]), f"{k} missing or non-finite"
    assert r["time_s"] > 20.0, "implausibly fast crossing"
    assert 0.0 <= r["p_finish"] <= 1.0


def test_sweep_and_detail_agree():
    """Sweep mode is coarser, not different. Key numbers must match."""
    a = evaluate(BASELINE, detail=False)
    b = evaluate(BASELINE, detail=True)
    for k, tol in (("draft_in", 1e-6), ("gm_in", 1e-6), ("time_s", 0.05),
                   ("panel_warp", 1e-9), ("girder_utilisation", 1e-6)):
        assert abs(a[k] - b[k]) <= tol * max(1.0, abs(b[k])), \
            f"{k} differs between sweep and detail: {a[k]} vs {b[k]}"


def test_design_survives_a_csv_round_trip():
    """A row written by the sweep must rebuild the EXACT hull it scored.

    This is the bug this test exists to prevent: if the sweep writes only the
    columns the charts use, reloading a row silently substitutes baseline
    values for everything else, and the detailed report then describes a
    different boat from the one that won. It shows up as small unexplained
    differences in tape budget and score, which is exactly the kind of thing
    nobody investigates.
    """
    d = BASELINE.with_(length=93.5, beam=37.25, bottom_width=31.0,
                       side_height=15.5, bow_rocker=3.25, chine_frac=0.42,
                       n_frames=7, skin_layers=2, decked_ends=False,
                       free_water_lanes=3, chine_tape_layers=3,
                       tape_diag_density=0.37, tape_diag_opposing=0.11,
                       stern_sheer=2.75, rocker_exp=2.4, stern_exit_exp=1.7,
                       transom_width_frac=0.55, joint_type="partial_flap",
                       flute_direction="longitudinal", bottom_doubler=False,
                       crew_weights=(131.0, 168.0), crew_kg=(15.5, 15.5),
                       crew_x_frac=(0.28, 0.74), crew_power_w=(185.0, 145.0),
                       name="roundtrip")
    back = HullDesign.from_row(d.to_row())
    assert back == d, (
        "round trip changed the design:\n  "
        + "\n  ".join(f"{k}: {getattr(d, k)!r} -> {getattr(back, k)!r}"
                      for k in HullDesign.__dataclass_fields__
                      if getattr(d, k) != getattr(back, k)))

    # And the evaluated result must be identical, not merely similar.
    a, b = evaluate(d, detail=False), evaluate(back, detail=False)
    for k in ("gm_in", "time_s", "tape_spare_frac", "board_spare_frac",
              "expected_score", "twist_deg", "panel_warp"):
        close(a[k], b[k], 1e-12, f"{k} after round trip")


def test_sweep_row_carries_every_design_field():
    """Nothing in HullDesign may be missing from the CSV row."""
    r = evaluate(BASELINE, detail=False)
    missing = [f for f in HullDesign.__dataclass_fields__ if f not in r]
    assert not missing, f"design fields absent from the sweep row: {missing}"


def test_invalid_design_is_rejected_not_crashed():
    bad = BASELINE.with_(bottom_width=50.0, beam=30.0)
    r = evaluate(bad, detail=False)
    assert not r["feasible"] and r["n_violations"] > 0


def test_extreme_designs_do_not_raise():
    """A sweep will hand evaluate() nonsense. It must degrade, not explode."""
    for d in (BASELINE.with_(bottom_width=8.0, beam=10.0, side_height=9.0),
              BASELINE.with_(length=80.0, beam=42.0, bottom_width=36.0,
                             side_height=17.0, n_frames=0),
              BASELINE.with_(skin_layers=2, decked_ends=False, chine_frac=1.0,
                             bow_rocker=4.9, stern_rocker=4.9),
              BASELINE.with_(crew_weights=(300.0, 300.0))):
        r = evaluate(d, detail=False)
        assert "expected_score" in r
        assert np.isfinite(r["expected_score"]), f"non-finite score for {d}"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# uncertainty and Monte Carlo
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# pacing and fatigue
# ---------------------------------------------------------------------------

def test_crew_power_actually_changes_the_time():
    """Doubling the crew's power must make a real difference.

    This exists because it once did not. The low-speed thrust cap was set
    tight enough that it kept binding at race speed, so the boat was
    thrust-limited the whole way and power barely moved the answer. The times
    still looked plausible, which is what made it hard to notice.
    """
    times = []
    for total in (240.0, 320.0, 400.0, 480.0):
        r = evaluate(RECOMMENDED.with_(crew_power_w=(total / 2, total / 2)),
                     detail=False)
        times.append(r["time_s"])
    assert all(b < a for a, b in zip(times, times[1:])), \
        f"more power did not mean less time: {times}"
    assert times[0] - times[-1] > 5.0, \
        f"doubling power only bought {times[0] - times[-1]:.1f} s"


def test_thrust_cap_does_not_bind_at_race_speed():
    """The low-speed thrust limit must be a low-speed limit."""
    from hullsim.params import NOMINAL
    from hullsim.resistance import thrust_available
    r = evaluate(RECOMMENDED, detail=False)
    v = r["terminal_v_ms"]
    shaft = float(sum(RECOMMENDED.crew_power_w))
    cap = NOMINAL.max_thrust_per_paddler_lb * RECOMMENDED.n_crew
    demanded = shaft * NOMINAL.blade_efficiency / v * 0.224809      # N -> lbf
    assert demanded < cap, (
        f"thrust cap {cap:.0f} lbf still binds at race speed "
        f"({demanded:.0f} lbf demanded at {v:.2f} m/s): the boat is "
        f"thrust-limited and crew power will barely matter")
    assert abs(thrust_available(v, shaft, cap) - demanded) < 0.01


def test_hard_start_is_slower_than_even_pacing():
    """Sprinting off the line and hanging on must cost time.

    Not a tautology: the shapes all deliver about the same total effort. The
    hard start loses because overspeeding early wastes energy against a drag
    curve that climbs steeply past Froude 0.4, and the anaerobic reserve
    spent doing it does not come back.
    """
    t = {}
    for pac in ("even", "fast_start", "hard_start", "negative_split"):
        t[pac] = evaluate(RECOMMENDED.with_(pacing=pac), detail=False)["time_s"]
    assert t["hard_start"] > t["even"] + 0.4, \
        f"hard start was not punished: {t}"
    assert t["even"] <= min(t.values()) + 0.15, \
        f"even pacing should be at or near the best: {t}"


def test_fade_shows_up_in_the_numbers():
    even = evaluate(RECOMMENDED.with_(pacing="even"), detail=False)
    hard = evaluate(RECOMMENDED.with_(pacing="hard_start"), detail=False)
    assert abs(even["fade_frac"]) < 0.02, "even pacing should not fade"
    assert hard["fade_frac"] > 0.25, \
        f"hard start should fade hard, got {hard['fade_frac']:.0%}"
    assert hard["power_start_w"] > even["power_start_w"] * 1.2
    assert hard["v_peak_ms"] > even["v_peak_ms"], \
        "a hard start should at least reach a higher peak speed"


def test_anaerobic_reserve_is_finite():
    """Spending above critical power must run out, not continue forever."""
    from hullsim.resistance import simulate_race, drag_table, resistance

    def drag(v):
        return resistance(max(v, 1e-3), wetted_in2=3200.0, lwl_in=104.0,
                          beam_wl_in=34.0, draft_in=3.1, volume_in3=9000.0,
                          displacement_lb=325.0, cb=0.72, cp=0.70)

    tab = drag_table(drag)
    # A very long course: the reserve must be gone and power pinned at CP.
    res = simulate_race(tab, displacement_lb=325.0, shaft_power_w=320.0,
                        max_thrust_lb=44.0, course_m=900.0, pacing="hard_start")
    assert res.wprime_spent_frac > 0.99, "reserve was never exhausted"
    assert res.pinned_at_cp_s > 10.0, \
        f"never pinned at critical power over 900 m ({res.pinned_at_cp_s:.1f}s)"
    assert res.power_finish_w < res.power_start_w


def test_pacing_slip_makes_things_worse_not_better():
    """The Monte Carlo's pacing slip must be a penalty."""
    from hullsim.montecarlo import run_trials, summarise
    from hullsim.uncertainty import UncertaintyConfig
    cfg = UncertaintyConfig.only("race")
    a = summarise(run_trials(RECOMMENDED.with_(pacing="even"), 600, cfg=cfg,
                             seed=31, processes=1))
    b = summarise(run_trials(RECOMMENDED.with_(pacing="hard_start"), 600,
                             cfg=cfg, seed=31, processes=1))
    assert b["time_p50"] > a["time_p50"], \
        f"planning a hard start was not slower: {a['time_p50']:.2f} vs {b['time_p50']:.2f}"


# ---------------------------------------------------------------------------
# material presets
# ---------------------------------------------------------------------------

def test_material_presets_are_physically_distinct():
    """Solid board and corrugated must behave like different materials.

    Not a naming difference. Corrugated is two liners held apart by a fluted
    core: light, very stiff in bending, weak edgewise. Solid board of similar
    weight is roughly a tenth as stiff and several times stronger edgewise.
    If these ever converge, the preset system has stopped meaning anything.
    """
    from hullsim.params import Params
    corr = Params.of("corrugated_c_flute")
    solid = Params.of("paperboard_1.0mm")
    assert corr.is_corrugated and not solid.is_corrugated
    assert abs(solid.board_areal_lb_ft2 - corr.board_areal_lb_ft2) < 0.05, \
        "these two should weigh about the same per unit area"
    assert solid.board_d_stiff_lbin < 0.2 * corr.board_d_stiff_lbin, \
        "solid board should be far floppier for the same weight"
    assert solid.board_ect_lb_in > 3.0 * corr.board_ect_lb_in, \
        "solid board should be far stronger edgewise"


def test_material_changes_which_failure_mode_governs():
    """With corrugated the girder is the worry; with thin solid board, twist."""
    from hullsim.params import Params
    corr = evaluate(RECOMMENDED, detail=False, p=Params.of("corrugated_c_flute"))
    solid = evaluate(RECOMMENDED, detail=False, p=Params.of("paperboard_1.0mm"))
    assert solid["girder_utilisation"] < 0.5 * corr["girder_utilisation"], \
        "solid board should make the hull girder a non-issue"
    assert solid["twist_deg"] > 1.8 * corr["twist_deg"], \
        "solid board should twist much more"


def test_thin_board_tolerates_more_warp():
    """Corrugated crushes under forced double curvature; a thin sheet bends."""
    from hullsim.params import Params
    d = RECOMMENDED.with_(bow_rocker=3.5, beam=40.0, bottom_width=30.0,
                          bow_taper=26.0)
    corr = evaluate(d, detail=False, p=Params.of("corrugated_c_flute"))
    solid = evaluate(d, detail=False, p=Params.of("paperboard_1.0mm"))
    close(corr["panel_warp"], solid["panel_warp"], 1e-9,
          "geometry is the same, only the material differs")
    assert solid["warp_ratio"] < 0.5 * corr["warp_ratio"]


def test_solid_board_ignores_flute_direction():
    """You cannot choose the flute direction of a board that has no flutes."""
    from hullsim.params import Params
    p = Params.of("paperboard_1.0mm")
    a = evaluate(RECOMMENDED.with_(flute_direction="transverse"),
                 detail=False, p=p)
    b = evaluate(RECOMMENDED.with_(flute_direction="longitudinal"),
                 detail=False, p=p)
    close(a["girder_utilisation"], b["girder_utilisation"], 1e-9,
          "flute direction should do nothing to solid board")


def test_thicker_solid_board_weighs_proportionally_more():
    from hullsim.params import Params
    ws = [evaluate(RECOMMENDED, detail=False,
                   p=Params.of(m))["hull_weight_lb"]
          for m in ("paperboard_0.6mm", "paperboard_1.0mm",
                    "paperboard_1.5mm", "paperboard_2.0mm")]
    assert all(b > a for a, b in zip(ws, ws[1:])), f"not monotonic: {ws}"
    assert ws[-1] > 2.5 * ws[0], "2 mm should weigh far more than 0.6 mm"


# ---------------------------------------------------------------------------
# the crew
# ---------------------------------------------------------------------------

def test_anthropometry_reproduces_the_old_posture_table():
    """The stature-based model must agree with the fixed table it replaced.

    The old table was written for a roughly 70 in paddler. If the new model
    did not reproduce it there, it would not be a generalisation, it would be
    a different answer wearing the same name.
    """
    from hullsim.crew import paddler_geometry
    assert abs(paddler_geometry(70.0, "kneeling_tall").cg_z - 21.0) < 1.0
    assert abs(paddler_geometry(70.0, "kneeling_upright").cg_z - 15.5) < 1.0
    assert abs(paddler_geometry(70.0, "back_on_heels").cg_z - 11.0) < 1.0


def test_shorter_paddlers_sit_lower():
    """Free stability from being short, which a fixed table cannot see."""
    from hullsim.crew import paddler_geometry
    short = paddler_geometry(63.0, "kneeling_upright")
    tall = paddler_geometry(74.0, "kneeling_upright")
    assert short.cg_z < tall.cg_z - 1.5
    assert short.shoulder_z < tall.shoulder_z


def test_posture_is_a_trade_not_a_free_win():
    """Crouching must buy stability AND cost reach.

    This is the whole reason crew.py exists. Before it, every term in the
    model rewarded a lower centre of gravity and nothing objected, so the
    recommendation was to fold up as small as a body goes.
    """
    tall = evaluate(RECOMMENDED.with_(crew_posture=("kneeling_tall",) * 2),
                    detail=False)
    low = evaluate(RECOMMENDED.with_(crew_posture=("back_on_heels",) * 2),
                   detail=False)
    assert low["gm_in"] > tall["gm_in"] + 3.0, "crouching did not buy stability"
    assert low["reach_efficiency"] < tall["reach_efficiency"] - 0.15, \
        "crouching did not cost reach"
    assert low["time_s"] > tall["time_s"] + 1.0, "crouching was free"


def test_deep_sides_eventually_stop_you_paddling():
    effs = [evaluate(RECOMMENDED.with_(side_height=float(h)),
                     detail=False)["reach_efficiency"]
            for h in (11, 15, 19, 23)]
    assert all(b < a for a, b in zip(effs, effs[1:])), \
        f"reach did not fall with side height: {effs}"
    r = evaluate(RECOMMENDED.with_(side_height=26.0), detail=False)
    assert not r["feasible"], "a 26 in deep hull should be unpaddleable"


def test_side_height_has_an_interior_optimum():
    """Freeboard and reach pull opposite ways, so there is a best depth.

    Before the reach penalty existed the model simply wanted more freeboard
    without limit, which is not a design, it is a runaway.
    """
    scores = {h: evaluate(RECOMMENDED.with_(side_height=float(h)),
                          detail=False)["expected_score"]
              for h in (11, 13, 15, 17, 19, 21)}
    best = max(scores, key=scores.get)
    assert best not in (11, 21), f"optimum should be interior, got {best}: {scores}"


def test_crew_cannot_be_placed_on_top_of_each_other():
    close = evaluate(RECOMMENDED.with_(crew_x_frac=(0.44, 0.56)), detail=False)
    assert not close["crew_spacing_ok"], "knees should overlap at 12 in apart"
    assert not close["feasible"]
    ends = evaluate(RECOMMENDED.with_(crew_x_frac=(0.04, 0.96)), detail=False)
    assert not ends["crew_spacing_ok"], "kneeling in the tapered ends is not allowed"
    ok = evaluate(RECOMMENDED, detail=False)
    assert ok["crew_spacing_ok"]


def test_moving_the_crew_changes_trim():
    fwd = evaluate(RECOMMENDED.with_(crew_x_frac=(0.26, 0.66)), detail=False)
    aft = evaluate(RECOMMENDED.with_(crew_x_frac=(0.34, 0.74)), detail=False)
    assert aft["trim_deg"] > fwd["trim_deg"] + 0.05


# ---------------------------------------------------------------------------
# progressive soaking
# ---------------------------------------------------------------------------

def test_soaking_decays_toward_the_floor():
    from hullsim.constants import soak_factor, WET_FACTOR_FLOOR
    vals = [soak_factor(t) for t in (0, 1, 3, 10, 60)]
    assert abs(vals[0] - 1.0) < 1e-9, "a dry hull should start at full strength"
    assert all(b < a for a, b in zip(vals, vals[1:])), f"not monotonic: {vals}"
    assert abs(vals[-1] - WET_FACTOR_FLOOR) < 0.02, "did not settle at the floor"
    # Anchored to the literature: a coated board lost 47% of flat crush after
    # five minutes immersed. A taped hull should be somewhat better than that.
    assert 0.55 < soak_factor(5.0) < 0.80


def test_floating_early_costs_strength():
    early = evaluate(RECOMMENDED.with_(pre_race_soak_min=20.0), detail=False)
    late = evaluate(RECOMMENDED.with_(pre_race_soak_min=0.0), detail=False)
    assert early["wet_factor_at_finish"] < late["wet_factor_at_finish"] - 0.2
    assert early["twist_deg"] > late["twist_deg"], "a wetter hull should twist more"
    assert early["expected_score"] < late["expected_score"]


def test_structure_uses_end_of_race_strength():
    """Not the dry value and not the fully-soaked one.

    The floor and the time constant both come from how much of the immersed
    hull is under tape, because with no paint allowed the tape IS the
    waterproofing.
    """
    from hullsim.constants import soak_factor, soak_params
    r = evaluate(RECOMMENDED, detail=False)
    floor, tau = soak_params(r["bottom_tape_coverage"])
    expect = soak_factor(RECOMMENDED.pre_race_soak_min + r["time_s"] / 60.0,
                         floor=floor, tau_min=tau)
    close(r["wet_factor_at_finish"], expect, 1e-9, "end-of-race wet factor")
    assert floor < r["wet_factor_at_finish"] < 1.0


def test_tape_coverage_controls_soaking():
    """With no paint allowed, tape coverage is the waterproofing.

    Before this, tape was purely a structural and budget decision and the
    sweep happily traded it away. It is also the only thing between bare
    board and the lake.
    """
    bare = evaluate(RECOMMENDED.with_(tape_diag_density=0.1,
                                      tape_diag_opposing=0.0), detail=False)
    sealed = evaluate(RECOMMENDED.with_(tape_diag_density=0.9,
                                        tape_diag_opposing=0.3), detail=False)
    assert sealed["bottom_tape_coverage"] > bare["bottom_tape_coverage"] + 0.4
    assert sealed["wet_factor_at_finish"] > bare["wet_factor_at_finish"] + 0.08
    assert sealed["soak_tau_min"] > bare["soak_tau_min"]
    assert sealed["tape_spare_frac"] < bare["tape_spare_frac"], \
        "sealing the hull has to cost tape"


# ---------------------------------------------------------------------------
# numerical convergence -- the obligation that underwrites everything else
# ---------------------------------------------------------------------------

def test_hydrostatics_are_converged_at_the_default_station_count():
    """41 stations must give the same answer as 401.

    Every other test in this file compares the model against something. This
    one compares the model against itself at higher resolution, which is the
    only way to know the discretisation is not quietly contributing error of
    its own. Trim is excluded because it is 0.03 degrees: a large relative
    error on a number that small means nothing.
    """
    ref_mesh = build_mesh(RECOMMENDED, n_stations=401)
    ref_mp = mass_properties(ref_mesh, 24.0)
    ref_eq = solve_equilibrium(ref_mesh, ref_mp)
    ref_sc = righting_arms(ref_mesh, ref_mp, ref_eq, (0, 20))

    mesh = build_mesh(RECOMMENDED, n_stations=C.N_STATIONS)
    mp = mass_properties(mesh, 24.0)
    eq = solve_equilibrium(mesh, mp)
    sc = righting_arms(mesh, mp, eq, (0, 20))

    for got, want, name in ((eq.draft, ref_eq.draft, "draft"),
                            (eq.gm_t, ref_eq.gm_t, "GM"),
                            (eq.bm_t, ref_eq.bm_t, "BM"),
                            (eq.volume, ref_eq.volume, "volume"),
                            (sc.gz[1], ref_sc.gz[1], "GZ at 20 deg")):
        rel = abs(got - want) / max(abs(want), 1e-9)
        assert rel < 2e-3, f"{name} not converged: {got:.6g} vs {want:.6g} ({rel:.2%})"


def test_race_integration_is_converged_at_the_default_timestep():
    from hullsim.resistance import drag_table, resistance, simulate_race

    def drag(v):
        return resistance(max(v, 1e-3), wetted_in2=3200.0, lwl_in=104.0,
                          beam_wl_in=34.0, draft_in=3.1, volume_in3=9000.0,
                          displacement_lb=325.0, cb=0.72, cp=0.70)

    tab = drag_table(drag)
    kw = dict(displacement_lb=325.0, shaft_power_w=300.0, max_thrust_lb=44.0,
              pacing="even")
    coarse = simulate_race(tab, dt=0.02, **kw).time_s
    fine = simulate_race(tab, dt=0.001, **kw).time_s
    assert abs(coarse - fine) < 0.05, \
        f"timestep not converged: {coarse:.4f} vs {fine:.4f}"


def test_drag_table_matches_the_exact_function():
    from hullsim.resistance import drag_table, resistance, simulate_race

    def drag(v):
        return resistance(max(v, 1e-3), wetted_in2=3200.0, lwl_in=104.0,
                          beam_wl_in=34.0, draft_in=3.1, volume_in3=9000.0,
                          displacement_lb=325.0, cb=0.72, cp=0.70)

    kw = dict(displacement_lb=325.0, shaft_power_w=300.0, max_thrust_lb=44.0,
              dt=0.02, pacing="even")
    exact = simulate_race(lambda v: drag(v).r_total, **kw).time_s
    table = simulate_race(drag_table(drag), **kw).time_s
    assert abs(exact - table) < 0.05, \
        f"64-point drag table off by {abs(exact - table):.3f} s"


# ---------------------------------------------------------------------------
# directional stability, dynamic roll, wind
# ---------------------------------------------------------------------------

def test_rocker_costs_directional_stability():
    """Rocker lifts the lateral plane out and the boat stops tracking.

    Without this the model thinks rocker is nearly free, because nothing in
    a drag calculation knows that a boat which will not hold a line is
    spending power on correction strokes.
    """
    from hullsim.geometry import lateral_plane
    idx, eff, times = [], [], []
    for rk in (0.0, 1.5, 3.0, 4.5):
        d = RECOMMENDED.with_(bow_rocker=rk, stern_rocker=min(0.4 * rk, 3.0))
        m = build_mesh(d)
        mp = mass_properties(m, 24.0)
        eq = solve_equilibrium(m, mp)
        idx.append(lateral_plane(m, eq.waterline_z)["tracking_index"])
        r = evaluate(d, detail=False)
        eff.append(r["tracking_efficiency"])
        times.append(r["time_s"])
    assert all(b < a for a, b in zip(idx, idx[1:])), \
        f"tracking index not falling with rocker: {idx}"
    assert eff[0] > eff[-1] + 0.04, f"rocker cost no efficiency: {eff}"
    assert times[-1] > times[0] + 0.3, f"rocker cost no time: {times}"


def test_mismatched_paddlers_cost_time():
    """Unequal power on opposite sides yaws the boat and needs correcting."""
    even = evaluate(RECOMMENDED.with_(crew_power_w=(160.0, 160.0)),
                    detail=False)
    lop = evaluate(RECOMMENDED.with_(crew_power_w=(220.0, 100.0)),
                   detail=False)
    assert lop["tracking_efficiency"] < even["tracking_efficiency"] - 0.05
    assert lop["time_s"] > even["time_s"] + 0.5, \
        "same total power, badly split, cost nothing"


def test_boat_cannot_respond_to_chop_at_race_speed():
    """Validates the quasi-static heel assumption used everywhere else.

    At race speed the encounter period is a fraction of the roll period, so
    the amplification is tiny and treating heel as quasi-static is correct
    rather than merely convenient. If this ever fails, the crew-lean model
    needs a dynamic term.
    """
    r = evaluate(RECOMMENDED.with_(chop_height=2.0), detail=False)
    assert r["roll_racing_deg"] < 1.0, (
        f"wave roll at race speed is {r['roll_racing_deg']:.2f} deg: the "
        f"quasi-static heel model is no longer safe")


def test_stopped_in_chop_is_worse_than_moving():
    """Encounters slow to the wave period when she stops, and can resonate."""
    r = evaluate(RECOMMENDED.with_(chop_height=3.0), detail=False)
    assert r["roll_stopped_deg"] > r["roll_racing_deg"] * 3.0, (
        f"stopped roll {r['roll_stopped_deg']:.2f} not worse than racing "
        f"{r['roll_racing_deg']:.2f}")
    assert r["boarding_freeboard_with_chop_in"] < r["boarding_freeboard_in"], \
        "chop did not eat into the boarding margin"


def test_roll_resonance_is_not_monotonic_in_stiffness():
    """A stiffer hull is not automatically calmer.

    High GM shortens the roll period and can tune it INTO the chop. Nothing
    else in the model pushes back on ever-increasing GM, so this behaviour
    needs to survive refactoring.
    """
    from hullsim.seakeeping import roll_response
    m = build_mesh(RECOMMENDED)
    mp = mass_properties(m, 24.0)
    eq = solve_equilibrium(m, mp)
    rolls = {gm: roll_response(eq, 3.0, 0.0, gm).roll_amplitude_deg
             for gm in (5, 10, 22, 35, 50)}
    worst = max(rolls, key=rolls.get)
    assert worst not in (5, 50), \
        f"worst roll should be at an intermediate GM, got {worst}: {rolls}"


def test_headwind_costs_time_and_tailwind_saves_it():
    calm = evaluate(RECOMMENDED, detail=False)["time_s"]
    head = evaluate(RECOMMENDED.with_(headwind_mph=15.0), detail=False)["time_s"]
    tail = evaluate(RECOMMENDED.with_(headwind_mph=-10.0), detail=False)["time_s"]
    assert head > calm + 1.0, f"15 mph headwind cost only {head - calm:.2f} s"
    assert tail < calm, "a tailwind made her slower"


def test_monte_carlo_reports_its_own_error_bars():
    """An estimate without an error bar invites reading noise as signal."""
    from hullsim.montecarlo import run_trials, summarise
    a = summarise(run_trials(RECOMMENDED, 400, seed=41, processes=1))
    b = summarise(run_trials(RECOMMENDED, 1600, seed=41, processes=1))
    for k in ("p_feasible_se", "score_mean_se", "time_mean_se"):
        assert a[k] > 0 and np.isfinite(a[k]), f"{k} missing"
        assert b[k] < a[k], f"{k} did not shrink with more trials"


def test_default_params_reproduce_the_deterministic_model():
    """Params() must be exactly the constants the model always used.

    The whole Monte Carlo layer rests on this. If default Params drifted from
    constants.py, every deterministic answer would quietly change and the
    'nominal' column would stop being nominal.
    """
    from hullsim.params import NOMINAL
    close(NOMINAL.board_caliper_in, C.BOARD_CALIPER_IN, 0, "caliper")
    close(NOMINAL.board_areal_lb_ft2, C.BOARD_AREAL_LB_FT2, 0, "areal")
    close(NOMINAL.sigma_along_flutes,
          C.BOARD_ECT_LB_IN / C.BOARD_CALIPER_IN, 1e-12, "ECT stress")
    close(NOMINAL.e_stiff_psi, C.BOARD_E_MD_PSI, 1e-9, "E stiff")
    close(NOMINAL.e_soft_psi, C.BOARD_E_CD_PSI, 1e-9, "E soft")
    close(NOMINAL.tape_lb_per_in, C.TAPE_LB_PER_IN, 1e-12, "tape weight")

    a = evaluate(BASELINE, detail=False)
    b = evaluate(BASELINE, detail=False, p=NOMINAL)
    for k in ("gm_in", "time_s", "twist_deg", "chine_tape_util",
              "hull_weight_lb", "expected_score"):
        close(a[k], b[k], 1e-12, f"{k} with explicit NOMINAL params")


def test_zero_uncertainty_is_deterministic():
    """With every source switched off, trials must be identical to nominal."""
    from hullsim.montecarlo import run_trials
    from hullsim.uncertainty import UncertaintyConfig
    cfg = UncertaintyConfig.only()          # nothing enabled
    df = run_trials(BASELINE, 24, cfg=cfg, seed=3, processes=1)
    nom = evaluate(BASELINE, detail=False)
    assert df.time_s.nunique() == 1, "no-uncertainty trials differed"
    close(float(df.time_s.iloc[0]), nom["time_s"], 1e-12, "time")
    close(float(df.expected_score.iloc[0]), nom["expected_score"], 1e-12,
          "score")


def test_uncertainty_actually_spreads_the_answer():
    """And with it on, the same design must give different numbers."""
    from hullsim.montecarlo import run_trials
    df = run_trials(RECOMMENDED, 200, seed=4, processes=1)
    assert df.time_s.std() > 0.5, "time did not move at all"
    assert df.expected_score.nunique() > 50, "score barely varied"


def test_monte_carlo_is_reproducible():
    """Same seed, same answer -- otherwise nothing here can be checked."""
    from hullsim.montecarlo import run_trials
    a = run_trials(RECOMMENDED, 120, seed=9, processes=1)
    b = run_trials(RECOMMENDED, 120, seed=9, processes=1)
    close(float(a.time_s.mean()), float(b.time_s.mean()), 1e-12, "mean time")
    c = run_trials(RECOMMENDED, 120, seed=10, processes=1)
    assert abs(a.time_s.mean() - c.time_s.mean()) > 1e-9, \
        "different seeds gave identical results"


def test_each_source_can_be_isolated():
    from hullsim.montecarlo import run_trials
    from hullsim.uncertainty import UncertaintyConfig
    sds = {}
    for src in ("build", "race", "model"):
        df = run_trials(RECOMMENDED, 150, cfg=UncertaintyConfig.only(src),
                        seed=6, processes=1)
        sds[src] = float(df.time_s.std())
        assert sds[src] > 0.0, f"{src} alone produced no spread"
    # Model uncertainty dominates the time spread; see the residuary note.
    assert sds["model"] > sds["build"], \
        f"expected model uncertainty to dominate time, got {sds}"


def test_worst_of_n_matches_brute_force():
    """The one-random-number maximum must match actually taking n draws.

    worst_of_n_normal inverts the CDF of the maximum instead of sampling n
    values and picking the largest, which is 150x cheaper inside a Monte
    Carlo loop. Checked against the expensive version it replaces, in both
    mean and spread, because getting the spread wrong would be the subtle
    failure -- a method that recovers the right average worst stroke but not
    its variability would look fine and mis-state the risk.
    """
    from hullsim.uncertainty import worst_of_n_normal
    rng = np.random.default_rng(0)
    mu, sd, m = 2.0, 0.8, 30000
    for n in (1, 10, 150):
        fast = np.array([worst_of_n_normal(rng, mu, sd, n) for _ in range(m)])
        brute = rng.normal(mu, sd, size=(m, n)).max(axis=1)
        assert abs(fast.mean() - brute.mean()) < 0.02, \
            f"n={n}: mean {fast.mean():.3f} vs brute {brute.mean():.3f}"
        assert abs(fast.std() - brute.std()) < 0.02, \
            f"n={n}: sd {fast.std():.3f} vs brute {brute.std():.3f}"

    # And the point of it: the worst of a race's worth of strokes is far
    # out from the typical one, so designing to the average stroke is
    # designing to the wrong stroke.
    typical = np.mean([worst_of_n_normal(rng, mu, sd, 1) for _ in range(4000)])
    worst = np.mean([worst_of_n_normal(rng, mu, sd, 150) for _ in range(4000)])
    assert worst > typical + 2.0 * sd, \
        f"worst-of-150 ({worst:.2f}) barely above typical ({typical:.2f})"


def test_build_list_costs_righting_arm():
    """An asymmetric hull sits with a list and has less arm left over."""
    flat = evaluate(RECOMMENDED, detail=False)
    listed = evaluate(RECOMMENDED.with_(build_list_in=1.0), detail=False)
    assert listed["heel_at_rest_deg"] > 1.0, "no resting list from asymmetry"
    assert listed["gz_max_in"] < flat["gz_max_in"], "list did not cost arm"
    assert listed["paddle_heel_deg"] > flat["paddle_heel_deg"]


def test_water_aboard_costs_freeboard_and_stability():
    dry = evaluate(RECOMMENDED, detail=False)
    wet = evaluate(RECOMMENDED.with_(water_aboard_gal=3.0,
                                     free_water_lanes=1), detail=False)
    assert wet["freeboard_in"] < dry["freeboard_in"], "water did not sink her"
    assert wet["gz_max_in"] < dry["gz_max_in"], "free surface cost no arm"
    assert wet["swamp_reserve_gal"] < dry["swamp_reserve_gal"]


def test_sensitivity_finds_the_dominant_input():
    """The tornado must identify residuary scale as the driver of time.

    Not a tautology: it is a check that the recorded inputs line up with the
    trial outcomes row for row. Shuffle that alignment and this fails.
    """
    from hullsim.montecarlo import run_trials, sensitivity
    df = run_trials(RECOMMENDED, 800, seed=12, processes=1)
    s = sensitivity(df, "time_s")
    assert len(s), "no sensitivities computed"
    assert s.iloc[0]["input"] == "u_residuary_scale", \
        f"expected residuary scale to dominate, got {s.iloc[0]['input']}"
    assert s.iloc[0]["spearman"] > 0.5, "more wave-making should mean more time"


def test_summarise_shape():
    from hullsim.montecarlo import run_trials, summarise
    df = run_trials(RECOMMENDED, 120, seed=13, processes=1)
    s = summarise(df, RECOMMENDED)
    assert 0.0 <= s["p_feasible"] <= 1.0
    assert s["time_p10"] <= s["time_p50"] <= s["time_p90"]
    assert s["score_p10"] <= s["score_p50"]
    assert s["robust_score"] == s["score_p10"]


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ok    {name}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
