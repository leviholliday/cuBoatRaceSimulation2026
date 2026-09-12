"""
Hydrostatics: displacement, trim, and the full large-angle righting-arm curve.

The whole module rests on one primitive -- clip every station polygon against
the waterplane and integrate -- so upright, trimmed and heeled cases all go
through the same code path and cannot disagree with each other.

Why a GZ curve and not just GM: GM is the slope of the righting arm at zero
heel. For a hull with this much flare it understates stability at small angles
and says nothing at all about the angle where the gunwale goes under. For an
open boat the gunwale going under IS the capsize -- there is no reserve
buoyancy above it, only an invitation -- so the downflooding angle is the
number that decides whether the crew gets wet, and GM alone will not find it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .geometry import HullMesh


# ---------------------------------------------------------------------------
# The primitive: clip station polygons against a half-plane and integrate
# ---------------------------------------------------------------------------

def clip_sections(verts: np.ndarray, ny, nz, c) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area and centroid of the part of each section below a plane.

    The submerged half-plane is  ny*y + nz*z <= c,  with (ny, nz) a unit
    vector pointing away from the water. `c` may be a scalar or one value per
    station, which is how trim enters -- a tilted waterline is just a
    different `c` at every station.

    Everything broadcasts over an optional leading batch axis, so a whole
    righting-arm curve is one call rather than one call per heel angle:
    pass `ny`/`nz` of shape (n_angles,) and `c` of shape (n_angles, n_stations)
    and get areas back as (n_angles, n_stations).

    Returns (area, cy, cz). Area is zero where the section is entirely dry,
    and the centroid there is meaningless (set to zero) but is always
    multiplied by that zero area downstream.

    Method: Sutherland-Hodgman clipping, vectorised. Instead of producing a
    variable-length vertex list per station -- which numpy hates -- every
    input edge emits exactly two output vertices:

        1. the vertex itself if it is underwater, otherwise its perpendicular
           projection onto the waterplane;
        2. the exact crossing point if the edge crosses the waterplane,
           otherwise a repeat of (1).

    Repeats contribute nothing to a shoelace sum, and every substitute point
    lies ON the waterline, so the extra vertices are collinear with the true
    clipped edge. Signed area and first moments along a straight chain of
    collinear points telescope exactly to the same values as the single edge
    they replace, so the result is not an approximation -- it is the exact
    clipped area and centroid, computed with a fixed-size array.
    """
    y = verts[..., 0]
    z = verts[..., 1]

    ny = np.asarray(ny, dtype=float)
    nz = np.asarray(nz, dtype=float)
    if ny.ndim:                       # batch over heel angles
        ny = ny[..., None, None]
        nz = nz[..., None, None]

    c_arr = np.asarray(c, dtype=float)
    if c_arr.ndim:                    # last axis is stations
        c_arr = c_arr[..., None]

    s = ny * y + nz * z - c_arr          # > 0 means above water
    above = s > 0.0

    # (1) clamp dry vertices down onto the waterplane
    push = np.where(above, s, 0.0)
    y1 = y - push * ny
    z1 = z - push * nz

    # (2) exact crossing point where an edge pierces the surface
    y_next = np.roll(y, -1, axis=-1)
    z_next = np.roll(z, -1, axis=-1)
    s_next = np.roll(s, -1, axis=-1)

    crosses = above != (s_next > 0.0)
    denom = s - s_next
    t = np.divide(s, denom, out=np.zeros_like(s), where=np.abs(denom) > 1e-15)
    y2 = np.where(crosses, y + t * (y_next - y), y1)
    z2 = np.where(crosses, z + t * (z_next - z), z1)

    y1, z1, y2, z2 = np.broadcast_arrays(y1, z1, y2, z2)
    nv = y1.shape[-1]
    py = np.empty(y1.shape[:-1] + (2 * nv,))
    pz = np.empty_like(py)
    py[..., 0::2], py[..., 1::2] = y1, y2
    pz[..., 0::2], pz[..., 1::2] = z1, z2

    # shoelace over the (possibly degenerate) clipped polygon
    py_n = np.roll(py, -1, axis=-1)
    pz_n = np.roll(pz, -1, axis=-1)
    cross = py * pz_n - py_n * pz

    a2 = cross.sum(axis=-1)               # 2 * signed area
    area = 0.5 * np.abs(a2)

    safe = np.abs(a2) > 1e-12
    denom3 = np.where(safe, 3.0 * a2, 1.0)
    cy = np.where(safe, ((py + py_n) * cross).sum(axis=-1) / denom3, 0.0)
    cz = np.where(safe, ((pz + pz_n) * cross).sum(axis=-1) / denom3, 0.0)

    return area, cy, cz


def _integrate(mesh: HullMesh, ny, nz, c):
    """Volume and centre of buoyancy for one waterplane, or a batch of them."""
    a, cy, cz = clip_sections(mesh.verts, ny, nz, c)
    w = mesh.w_int
    vol = a @ w
    safe = vol > 1e-9
    div = np.where(safe, vol, 1.0)
    xb = np.where(safe, (a * mesh.x) @ w / div, 0.0)
    yb = np.where(safe, (a * cy) @ w / div, 0.0)
    zb = np.where(safe, (a * cz) @ w / div, 0.0)
    if np.ndim(vol) == 0:
        return float(vol), float(xb), float(yb), float(zb), a
    return vol, xb, yb, zb, a


# ---------------------------------------------------------------------------
# Mass properties
# ---------------------------------------------------------------------------

@dataclass
class MassProps:
    weight: float          # lb, total afloat weight
    kg: float              # in, vertical CG above baseline
    lcg: float             # in, longitudinal CG from the bow
    hull_weight: float
    crew_weight: float
    water_weight: float    # shipped water aboard
    parts: dict = field(default_factory=dict)


def hull_cg(mesh: HullMesh) -> tuple[float, float]:
    """Where the hull's own structure sits, from the actual section girth.

    Better than assuming a fraction of side height: with rocker and sheer the
    ends of a real hull are both higher and lighter than midship, and with
    plan taper there is simply less material there.
    """
    d = mesh.design
    depth = np.maximum(mesh.gunwale_z - mesh.keel_z, 1e-9)
    hc = np.clip(d.chine_frac, 1e-3, 1.0) * depth

    w_bot = 2.0 * mesh.half_bottom
    z_bot = mesh.keel_z

    rise = hc
    run = mesh.half_beam - mesh.half_bottom
    w_flare = 2.0 * np.hypot(rise, run)
    z_flare = mesh.keel_z + 0.5 * hc

    w_side = 2.0 * np.maximum(depth - hc, 0.0)
    z_side = mesh.keel_z + hc + 0.5 * np.maximum(depth - hc, 0.0)

    girth = w_bot + w_flare + w_side
    mz = w_bot * z_bot + w_flare * z_flare + w_side * z_side

    total = float(np.trapezoid(girth, mesh.x))
    if total < 1e-9:
        return 0.5 * d.side_height, 0.5 * d.length
    kz = float(np.trapezoid(mz, mesh.x) / total)
    lx = float(np.trapezoid(girth * mesh.x, mesh.x) / total)
    return kz, lx


def mass_properties(mesh: HullMesh, hull_weight: float,
                    water_weight: float = 0.0,
                    water_depth: float = 0.0) -> MassProps:
    """Combine crew, hull, paddles and any shipped water into one CG."""
    d = mesh.design

    x_crew = np.array(d.crew_x_frac) * d.length
    floor_at_crew = np.interp(x_crew, mesh.x, mesh.keel_z)
    w_crew = np.array(d.crew_weights, dtype=float)
    # A paddler's CG is given above the floor they kneel on, and rocker lifts
    # that floor. Sitting near a lifted end genuinely raises the crew CG.
    z_crew = floor_at_crew + np.array(d.crew_kg, dtype=float)

    w_paddle = d.n_crew * C.PADDLE_WEIGHT_LB
    z_paddle = float(np.interp(0.5 * d.length, mesh.x, mesh.keel_z)) + 2.0
    x_paddle = 0.5 * d.length

    z_hull, x_hull = hull_cg(mesh)

    # Shipped water lies on the floor; its CG is half its own depth up.
    x_water = 0.5 * d.length
    z_water = float(mesh.keel_z.min()) + 0.5 * max(water_depth, 0.0)

    w = np.concatenate([w_crew, [w_paddle, hull_weight, water_weight]])
    z = np.concatenate([z_crew, [z_paddle, z_hull, z_water]])
    x = np.concatenate([x_crew, [x_paddle, x_hull, x_water]])

    total = float(w.sum())
    return MassProps(
        weight=total,
        kg=float((w * z).sum() / total),
        lcg=float((w * x).sum() / total),
        hull_weight=hull_weight,
        crew_weight=float(w_crew.sum()),
        water_weight=water_weight,
        parts={
            "crew_z": z_crew.tolist(), "crew_x": x_crew.tolist(),
            "hull_z": z_hull, "hull_x": x_hull,
        },
    )


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------

@dataclass
class Equilibrium:
    solved: bool
    draft: float             # in, at the deepest point of the keel
    trim_slope: float        # in of waterline rise per in aft; + = down by stern
    trim_deg: float
    waterline_z: np.ndarray  # water surface height above baseline, per station
    volume: float            # in^3
    lcb: float
    kb: float
    aw: float                # waterplane area, in^2
    bm_t: float
    bm_l: float
    gm_t: float              # small-angle transverse metacentric height
    gm_l: float
    freeboard_min: float     # in, at the lowest point of the gunwale line
    freeboard_at: float      # in, x where that minimum occurs
    bow_draft: float
    stern_draft: float
    beam_wl: float
    lwl: float
    cb: float                # block coefficient on the waterline box
    cp: float                # prismatic coefficient
    downflooding: bool
    note: str = ""


def _solve_sinkage(mesh: HullMesh, v_req: float, tau: float,
                   x_ref: float) -> tuple[float, float]:
    """Find the waterline height that displaces exactly v_req, at fixed trim.

    Plain bisection. Displacement is monotonic in sinkage, so this cannot fail
    and cannot oscillate, and each step costs one clip of a 41 x 6 array --
    so the extra evaluation a Newton step needs for its derivative costs more
    than the iterations it saves.
    """
    offs = tau * (mesh.x - x_ref)
    lo = 0.0
    hi = float(mesh.gunwale_z.max() + max(0.0, -offs.min()) + 1.0)

    v_lo, *_ = _integrate(mesh, 0.0, 1.0, lo + offs)
    v_hi, *_ = _integrate(mesh, 0.0, 1.0, hi + offs)
    if v_hi < v_req:
        return hi, v_hi                      # cannot displace enough: swamped

    for _ in range(40):
        h = 0.5 * (lo + hi)
        vol, *_ = _integrate(mesh, 0.0, 1.0, h + offs)
        if vol > v_req:
            hi, v_hi = h, vol
        else:
            lo, v_lo = h, vol
        if hi - lo < 1e-4:      # a ten-thousandth of an inch of draft
            break

    # Finish with one linear interpolation between the brackets instead of
    # taking the midpoint. Volume is very nearly linear in sinkage over a
    # bracket this narrow, so this costs nothing and takes the residual from
    # about one part in 1e5 to one part in 1e9 -- which matters only because
    # the tests check against closed-form answers, and a solver that is
    # merely close is a solver whose errors nobody notices growing.
    span = v_hi - v_lo
    h = lo + (v_req - v_lo) * (hi - lo) / span if abs(span) > 1e-12 else 0.5 * (lo + hi)
    vol, *_ = _integrate(mesh, 0.0, 1.0, h + offs)
    return h, vol


def solve_equilibrium(mesh: HullMesh, mp: MassProps,
                      free_to_trim: bool = True) -> Equilibrium:
    """Float the boat: find the draft and trim where buoyancy balances weight
    and the centre of buoyancy sits directly under the centre of gravity."""
    d = mesh.design
    v_req = mp.weight / C.RHO_LB_IN3
    x_ref = 0.5 * d.length

    tau = 0.0
    h = 0.0
    if not free_to_trim:
        h, _ = _solve_sinkage(mesh, v_req, 0.0, x_ref)
    else:
        # Secant iteration on trim slope to drive LCB onto LCG.
        def residual(t):
            hh, _ = _solve_sinkage(mesh, v_req, t, x_ref)
            _, xb, _, _, _ = _integrate(mesh, 0.0, 1.0, hh + t * (mesh.x - x_ref))
            return xb - mp.lcg, hh

        # LCB moves very nearly linearly with trim slope over the range a
        # loaded canoe ever sees, so the secant converges in two or three
        # steps. A thousandth of an inch of LCB/LCG mismatch is far below the
        # uncertainty in where a paddler's mass actually sits.
        t0, t1 = 0.0, 0.01
        r0, h0 = residual(t0)
        r1, h1 = residual(t1)
        tau, h = t1, h1
        for _ in range(12):
            if abs(r1) < 1e-3 or abs(r1 - r0) < 1e-14:
                break
            t2 = float(np.clip(t1 - r1 * (t1 - t0) / (r1 - r0), -0.25, 0.25))
            r2, h2 = residual(t2)
            t0, r0 = t1, r1
            t1, r1, h1 = t2, r2, h2
            tau, h = t1, h1

    wl = h + tau * (mesh.x - x_ref)
    vol, xb, _, zb, area = _integrate(mesh, 0.0, 1.0, wl)

    # --- waterplane properties -------------------------------------------
    # Half-breadth at the waterline, per station, from the section geometry.
    depth = np.maximum(mesh.gunwale_z - mesh.keel_z, 1e-9)
    hc = np.clip(d.chine_frac, 1e-3, 1.0) * depth
    dl = np.clip(wl - mesh.keel_z, 0.0, depth)
    frac = np.clip(np.divide(dl, hc, out=np.ones_like(dl), where=hc > 0), 0.0, 1.0)
    b_half = np.where(
        dl <= 0.0, 0.0,
        mesh.half_bottom + (mesh.half_beam - mesh.half_bottom) * frac,
    )

    wet = dl > 1e-9
    aw = float(np.trapezoid(2.0 * b_half, mesh.x))
    it = float(np.trapezoid((2.0 / 3.0) * b_half ** 3, mesh.x))
    if aw > 1e-9:
        lcf = float(np.trapezoid(2.0 * b_half * mesh.x, mesh.x) / aw)
        il = float(np.trapezoid(2.0 * b_half * (mesh.x - lcf) ** 2, mesh.x))
    else:
        lcf, il = x_ref, 0.0

    bm_t = it / vol if vol > 1e-9 else 0.0
    bm_l = il / vol if vol > 1e-9 else 0.0

    gunwale_fb = mesh.gunwale_z - wl
    i_min = int(np.argmin(gunwale_fb))

    lwl = float(mesh.x[wet].max() - mesh.x[wet].min()) if wet.any() else 0.0
    beam_wl = float(2.0 * b_half.max())
    draft = float(np.max(wl - mesh.keel_z)) if wet.any() else 0.0

    amid = float(area.max()) if area.size else 0.0
    cb = vol / (lwl * beam_wl * draft) if min(lwl, beam_wl, draft) > 1e-6 else 0.0
    cp = vol / (amid * lwl) if amid > 1e-6 and lwl > 1e-6 else 0.0

    v_cap, *_ = _integrate(mesh, 0.0, 1.0, mesh.gunwale_z)
    note = ""
    solved = True
    if v_cap < v_req:
        solved = False
        note = ("cannot displace her own weight before the gunwale goes under: "
                "this hull sinks with this load")

    return Equilibrium(
        solved=solved,
        draft=draft,
        trim_slope=tau,
        trim_deg=math.degrees(math.atan(tau)),
        waterline_z=wl,
        volume=vol,
        lcb=xb,
        kb=zb,
        aw=aw,
        bm_t=bm_t,
        bm_l=bm_l,
        gm_t=zb + bm_t - mp.kg,
        gm_l=zb + bm_l - mp.kg,
        freeboard_min=float(gunwale_fb[i_min]),
        freeboard_at=float(mesh.x[i_min]),
        bow_draft=float(wl[0] - mesh.keel_z[0]),
        stern_draft=float(wl[-1] - mesh.keel_z[-1]),
        beam_wl=beam_wl,
        lwl=lwl,
        cb=cb,
        cp=cp,
        downflooding=bool((gunwale_fb <= 0).any()),
        note=note,
    )


# ---------------------------------------------------------------------------
# Large-angle stability
# ---------------------------------------------------------------------------

@dataclass
class StabilityCurve:
    heel_deg: np.ndarray
    gz: np.ndarray               # in, positive = righting
    gm0: float                   # in, slope at the origin
    gz_max: float
    heel_at_gz_max: float
    downflood_deg: float         # heel at which the gunwale first goes under
    vanishing_deg: float         # heel at which GZ returns to zero
    area_to_downflood: float     # in.deg -- the energy reserve against a gust
    usable_deg: float            # min(downflood, vanishing)
    heel_at_rest: float = 0.0    # steady list from asymmetry or loose water


def righting_arms(mesh: HullMesh, mp: MassProps, eq: Equilibrium,
                  angles_deg=C.HEEL_ANGLES_DEG) -> StabilityCurve:
    """The righting-arm curve, computed by actually heeling the hull.

    Trim is frozen at the upright equilibrium. Free-to-trim GZ is the textbook
    refinement but it changes very little on a hull this short and costs an
    order of magnitude more solving.
    """
    d = mesh.design
    v_req = mp.weight / C.RHO_LB_IN3
    x_ref = 0.5 * d.length
    trim_off = eq.trim_slope * (mesh.x - x_ref)

    def heel_states(phis_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Righting arm and gunwale-immersion margin, for a whole set of heel
        angles at once.

        The margin is the perpendicular height of the lowest point of either
        gunwale above the heeled waterplane. It passes smoothly through zero
        as she rolls down, so the downflooding angle can be interpolated
        rather than snapped to whichever angle happened to be on the grid.

        Every angle is solved in the same vectorised bisection. The clipping
        primitive broadcasts, so the cost of the whole curve is close to the
        cost of one angle.
        """
        phi = np.radians(np.asarray(phis_deg, dtype=float))
        ny, nz = np.sin(phi), np.cos(phi)

        span = (np.abs(mesh.verts[..., 0]).max() * np.abs(ny)
                + mesh.gunwale_z.max() * np.abs(nz)) + 2.0
        lo, hi = -span, span
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            vol, *_ = _integrate(mesh, ny, nz, mid[:, None] + nz[:, None] * trim_off)
            too_deep = vol > v_req
            hi = np.where(too_deep, mid, hi)
            lo = np.where(too_deep, lo, mid)
            if np.all(hi - lo < 1e-5):
                break

        cc = 0.5 * (lo + hi)[:, None] + nz[:, None] * trim_off
        _, _, yb, zb, _ = _integrate(mesh, ny, nz, cc)

        # Righting arm: the earth-horizontal offset between B and G. Positive
        # is righting; tests/test_physics.py pins this against GM*sin(phi)
        # near the origin so the sign convention cannot drift.
        arm = (zb - mp.kg) * np.sin(phi) - yb * np.cos(phi)

        s_port = ny[:, None] * mesh.verts[None, :, 0, 0] + nz[:, None] * mesh.verts[None, :, 0, 1]
        s_stbd = ny[:, None] * mesh.verts[None, :, 5, 0] + nz[:, None] * mesh.verts[None, :, 5, 1]
        margin = np.minimum((s_port - cc).min(axis=-1), (s_stbd - cc).min(axis=-1))
        return arm, margin

    angles = np.asarray(angles_deg, dtype=float)
    gz, margins = heel_states(angles)
    gz = np.asarray(gz, dtype=float).copy()
    gz[angles == 0.0] = 0.0

    def _zero_crossing(vals: np.ndarray, grid: np.ndarray) -> float:
        """First downward crossing of zero, linearly interpolated in angle."""
        for i in range(1, len(grid)):
            if vals[i] <= 0.0 < vals[i - 1]:
                f = vals[i - 1] / (vals[i - 1] - vals[i])
                return float(grid[i - 1] + f * (grid[i] - grid[i - 1]))
        return float("nan")

    downflood = _zero_crossing(margins, angles)
    if not math.isnan(downflood):
        # The margin is smooth but not linear, so refine on a fine grid inside
        # the bracket -- one more batched solve rather than a scalar bisection.
        lo_a = float(angles[angles < downflood].max(initial=0.0))
        hi_a = float(angles[angles >= downflood].min(initial=float(angles[-1])))
        fine = np.linspace(lo_a, hi_a, 13)
        _, fine_margin = heel_states(fine)
        refined = _zero_crossing(fine_margin, fine)
        if not math.isnan(refined):
            downflood = refined

    vanishing = _zero_crossing(gz, angles)

    # Whichever limit she reaches first. Both can be absent: a hull with
    # negative GM never has a positive righting arm to lose, and one with
    # huge freeboard may not immerse a gunwale inside the angle range. In
    # either case the end of the computed range is the honest answer.
    candidates = [a for a in (downflood, vanishing) if not math.isnan(a)]
    usable = min(candidates) if candidates else float(angles[-1])

    # Energy reserve: the area under the righting-arm curve out to whichever
    # comes first, the gunwale going under or the arm vanishing. This is the
    # number that says whether she survives a gust or a bad stroke, as opposed
    # to GM, which only says how stiffly she resists the first degree of it.
    m = angles <= usable
    a_int = np.append(angles[m], usable)
    g_int = np.append(gz[m], float(np.interp(usable, angles, gz)))
    area = float(np.trapezoid(np.clip(g_int, 0.0, None), a_int)) if a_int.size > 1 else 0.0

    imax = int(np.argmax(gz))
    gm0 = eq.gm_t

    return StabilityCurve(
        heel_deg=angles, gz=gz, gm0=gm0,
        gz_max=float(gz[imax]), heel_at_gz_max=float(angles[imax]),
        downflood_deg=downflood, vanishing_deg=vanishing,
        area_to_downflood=area, usable_deg=float(usable),
    )


def apply_upsetting(sc: "StabilityCurve", fs_loss_in: float = 0.0,
                    list_in: float = 0.0) -> "StabilityCurve":
    """Correct a righting-arm curve for loose water and a built-in list.

    Both are steady upsetting influences rather than changes to the hull, and
    both reduce the arm available for everything else:

        GZ_effective(phi) = GZ(phi) - fs_loss*sin(phi) - list*cos(phi)

    The free-surface term is the standard correction -- an effective rise in
    the centre of gravity, so it acts like a GM reduction and grows with
    heel. The list term is a fixed transverse offset of the centre of
    gravity, so it bites hardest when she is UPRIGHT and eases as she rolls
    down onto it. That is why a listing boat has a resting angle: the arm is
    negative at zero heel and only reaches zero once she has leaned into it.

    Everything downstream -- the angle a paddle stroke reaches, whether she
    can be boarded, the energy reserve -- then reads off the corrected curve
    rather than a flat-water, empty-boat one.
    """
    if fs_loss_in <= 0.0 and list_in <= 0.0:
        return sc

    ang = sc.heel_deg
    rad = np.radians(ang)
    gz = sc.gz - fs_loss_in * np.sin(rad) - list_in * np.cos(rad)

    # She settles at the first angle where the arm comes back to zero.
    rest = 0.0
    for i in range(1, len(ang)):
        if gz[i - 1] < 0.0 <= gz[i]:
            f = (0.0 - gz[i - 1]) / max(gz[i] - gz[i - 1], 1e-12)
            rest = float(ang[i - 1] + f * (ang[i] - ang[i - 1]))
            break
    else:
        if gz[0] < 0.0:
            rest = float(ang[-1])        # never recovers: she lies over

    vanishing = float("nan")
    for i in range(1, len(ang)):
        if ang[i] > rest and gz[i] <= 0.0 < gz[i - 1]:
            f = gz[i - 1] / (gz[i - 1] - gz[i])
            vanishing = float(ang[i - 1] + f * (ang[i] - ang[i - 1]))
            break

    cands = [a for a in (sc.downflood_deg, vanishing) if not math.isnan(a)]
    usable = min(cands) if cands else float(ang[-1])

    m = (ang >= rest) & (ang <= usable)
    area = 0.0
    if m.sum() > 1:
        area = float(np.trapezoid(np.clip(gz[m], 0.0, None), ang[m]))

    imax = int(np.argmax(gz))
    return StabilityCurve(
        heel_deg=ang, gz=gz, gm0=sc.gm0 - fs_loss_in,
        gz_max=float(max(gz[imax], 0.0)), heel_at_gz_max=float(ang[imax]),
        downflood_deg=sc.downflood_deg, vanishing_deg=vanishing,
        area_to_downflood=area, usable_deg=usable, heel_at_rest=rest,
    )


def free_surface_loss(mesh: HullMesh, eq: Equilibrium, water_depth: float,
                      n_lanes: int = 1, heel_deg: float = 5.0,
                      weight_lb: float | None = None) -> float:
    """GM lost to water sloshing across the floor, in inches.

    THE TEXTBOOK FORMULA IS WRONG FOR A THIN LAYER, AND A CANOE ONLY EVER HAS
    A THIN LAYER

    The classical free-surface correction, i/V with i the water's own surface
    inertia, is derived assuming the water surface still touches both walls of
    its tank when the boat heels. It is famously independent of HOW MUCH water
    is aboard, which is why 'a few inches of loose water sinks ships' is true.

    But a gallon in a canoe is a tenth of an inch deep. Heel it by even half a
    degree and that layer has entirely run to the low side -- the surface no
    longer reaches the high wall, so it cannot keep tilting, and the heeling
    moment stops growing. Applying the classical formula past that point
    predicts tens of inches of GM loss from a cupful, which is nonsense.

    So this solves the actual geometry of water in a heeled rectangular
    channel, at a reference heel angle:

      Deep case   (d > (b/2)tan(theta)): surface spans the channel. The water
                  is a trapezoid, its centroid offsets by (b^2/12d)tan(theta),
                  and the result reduces exactly to the classical i/V.
      Wedge case  (thin layer): the water is a triangle against the low wall
                  with wetted floor width w = sqrt(2*A/tan(theta)). The lever
                  is b/2 - w/3, and it is capped by the geometry.

    The two agree exactly at the transition, which is the check that the case
    split is right.

    Lanes still help, and this is where you can see how much: a divider cuts
    both the lever and the amount of water free to reach it.
    """
    if water_depth <= 1e-6 or eq.volume <= 1e-9:
        return 0.0

    theta = math.radians(max(heel_deg, 0.05))
    tan_t = math.tan(theta)
    n = max(int(n_lanes), 1)
    w_total = weight_lb if weight_lb else eq.volume * C.RHO_LB_IN3

    # Channel width at each station, and the water area in ONE channel.
    depth = np.maximum(mesh.gunwale_z - mesh.keel_z, 1e-9)
    hc = np.clip(mesh.design.chine_frac, 1e-3, 1.0) * depth
    frac = np.clip(water_depth / hc, 0.0, 1.0)
    b_full = 2.0 * (mesh.half_bottom + (mesh.half_beam - mesh.half_bottom) * frac)
    b = b_full / n
    a_w = b * water_depth                       # water area per channel

    deep = water_depth > 0.5 * b * tan_t
    lever_deep = np.divide(b ** 2 * tan_t, 12.0 * max(water_depth, 1e-9))

    w_wedge = np.sqrt(np.maximum(2.0 * a_w / max(tan_t, 1e-9), 0.0))
    w_wedge = np.minimum(w_wedge, b)
    lever_wedge = 0.5 * b - w_wedge / 3.0

    lever = np.where(deep, lever_deep, lever_wedge)

    # Total transverse moment: n channels, each with its own water and lever.
    moment_per_in = C.RHO_LB_IN3 * n * a_w * np.maximum(lever, 0.0)
    moment = float(np.trapezoid(moment_per_in, mesh.x))

    return moment / (w_total * math.sin(theta))


def inclining_prediction(mp: MassProps, gm: float, shift_lb: float,
                         lever_in: float, plumb_in: float = 24.0) -> dict:
    """What the inclining test should read if the model is right.

        GM = (w * d) / (W * tan(theta))

    Run this on the finished boat before race day. A larger swing than
    predicted means the real GM is lower, and the usual culprit is a crew CG
    higher than assumed -- which is the cheapest thing left to fix.
    """
    if gm <= 0:
        return {"heel_deg": float("nan"), "plumb_swing_in": float("nan")}
    th = math.atan((shift_lb * lever_in) / (mp.weight * gm))
    return {
        "heel_deg": math.degrees(th),
        "plumb_swing_in": plumb_in * math.tan(th),
        "shift_lb": shift_lb,
        "lever_in": lever_in,
        "plumb_in": plumb_in,
    }
