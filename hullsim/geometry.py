r"""
Parametric hull geometry.

Turns a HullDesign into a stack of transverse section polygons, one per
station. Everything downstream -- displacement, stability, wetted surface,
panel sizes -- is computed from those polygons by numerical integration, so
there is exactly one place where the hull shape is defined and one definition
of what the boat is.

Why polygons rather than closed-form area formulas: the moment you want heel,
trim, rocker and flare at the same time, the closed forms turn into a case
analysis that is very easy to get subtly wrong. Clipping a polygon against the
waterplane is one small routine that is right for every one of those cases.

The section at each station, from port gunwale around the bottom and back up:

        (-hB, gz) .______________________. (hB, gz)      gunwale
                  |                      |
        (-hB, kz+hc)\                    /(hB, kz+hc)    chine
                     \                  /
             (-hb, kz) `--------------' (hb, kz)         flat bottom

    hb  = half-width of the flat bottom at this station
    hB  = half-beam at the gunwale at this station
    kz  = height of the bottom above the baseline here (rocker)
    gz  = height of the gunwale above the baseline here (sheer)
    hc  = chine_frac * (gz - kz), the height over which the side flares

With chine_frac = 1 the chine vertices coincide with the gunwale vertices and
the section is a plain trapezoid. Degenerate (repeated) vertices are harmless
in every routine here, which is what makes that work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .design import HullDesign

# Vertex order around the section. Index 0 is the port gunwale and the polygon
# closes from index 5 back to index 0 across the open top.
N_SECTION_VERTS = 6


def trapz_weights(x: np.ndarray) -> np.ndarray:
    """Weights w such that `f @ w` is the trapezoid integral of f over x."""
    w = np.zeros_like(x, dtype=float)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


@dataclass
class HullMesh:
    """A discretised hull. All arrays are indexed by station, bow to stern."""

    x: np.ndarray            # (ns,)  station positions, 0 = bow tip
    half_bottom: np.ndarray  # (ns,)  hb
    half_beam: np.ndarray    # (ns,)  hB
    keel_z: np.ndarray       # (ns,)  kz, bottom height above baseline
    gunwale_z: np.ndarray    # (ns,)  gz, gunwale height above baseline
    verts: np.ndarray        # (ns, 6, 2) section polygons as (y, z)
    design: HullDesign
    w_int: np.ndarray = None  # (ns,) trapezoid weights for integrating over x

    def __post_init__(self):
        # Precomputed trapezoid weights. The hydrostatic solver integrates
        # over stations tens of thousands of times per design, and a dot
        # product against fixed weights is several times cheaper than
        # re-deriving the spacing inside np.trapezoid on every call.
        if self.w_int is None:
            self.w_int = trapz_weights(self.x)

    # -- convenient scalars -------------------------------------------------
    @property
    def n_stations(self) -> int:
        return self.x.size

    @property
    def length(self) -> float:
        return float(self.x[-1] - self.x[0])

    @property
    def depth(self) -> np.ndarray:
        """Local depth, bottom to gunwale, at each station."""
        return self.gunwale_z - self.keel_z

    @property
    def min_gunwale_z(self) -> float:
        """The lowest point of the gunwale line. This is where she floods
        first, so it is the freeboard that actually matters."""
        return float(self.gunwale_z.min())


def _plan_taper(x: np.ndarray, length: float, bow_taper: float,
                stern_taper: float, bow_exp: float, stern_exp: float,
                stem_frac: float, transom_frac: float) -> np.ndarray:
    """Fraction of full width at each station, 0..1, from the plan-view taper.

    Inside the bow taper the width runs from `stem_frac` at the stem to 1.0
    where the taper ends, following (s ** exp) where s is the fraction of the
    way through the taper. exp < 1 is a full, blunt entry; exp > 1 is a fine
    entry that gives away buoyancy forward.
    """
    f = np.ones_like(x)

    if bow_taper > 1e-6:
        m = x < bow_taper
        s = np.clip(x[m] / bow_taper, 0.0, 1.0)
        f[m] = stem_frac + (1.0 - stem_frac) * s ** bow_exp

    if stern_taper > 1e-6:
        x_start = length - stern_taper
        m = x > x_start
        s = np.clip((length - x[m]) / stern_taper, 0.0, 1.0)
        f[m] = np.minimum(f[m], transom_frac + (1.0 - transom_frac) * s ** stern_exp)

    return f


def _end_curve(x: np.ndarray, length: float, bow_rise: float, stern_rise: float,
               exp: float) -> np.ndarray:
    """A curve that is zero amidships and rises to `bow_rise` / `stern_rise`
    at the ends. Used for both rocker (the keel lifting) and sheer (the
    gunwale lifting), which are the same shape function applied to different
    lines.

    exp = 1 gives a straight ramp from midship to the end; exp = 2 is
    parabolic, flat over the middle of the boat and curving up near the ends,
    which is what a rocker actually looks like on a real hull.
    """
    xm = 0.5 * length
    out = np.zeros_like(x)

    fwd = x < xm
    if xm > 1e-9:
        out[fwd] = bow_rise * ((xm - x[fwd]) / xm) ** exp

    aft = ~fwd
    run = length - xm
    if run > 1e-9:
        out[aft] = stern_rise * ((x[aft] - xm) / run) ** exp

    return out


def build_mesh(d: HullDesign, n_stations: int = C.N_STATIONS) -> HullMesh:
    """Discretise a design into station polygons."""
    x = np.linspace(0.0, d.length, n_stations)

    width_frac = _plan_taper(
        x, d.length, d.bow_taper, d.stern_taper,
        d.bow_entry_exp, d.stern_exit_exp,
        d.stem_width_frac, d.transom_width_frac,
    )

    half_bottom = 0.5 * d.bottom_width * width_frac
    half_beam = 0.5 * d.beam * width_frac

    # The gunwale must never be narrower than the bottom, or the section turns
    # inside out. Taper can make that happen right at the stem where both are
    # tiny, so clamp.
    half_beam = np.maximum(half_beam, half_bottom)

    keel_z = _end_curve(x, d.length, d.bow_rocker, d.stern_rocker, d.rocker_exp)
    gunwale_z = d.side_height + _end_curve(
        x, d.length, d.bow_sheer, d.stern_sheer, d.sheer_exp
    )

    # Baseline is the lowest point of the keel. _end_curve is already zero at
    # its minimum, but shift anyway so the invariant holds no matter what
    # shape function is used here later.
    z0 = keel_z.min()
    keel_z = keel_z - z0
    gunwale_z = gunwale_z - z0

    depth = np.maximum(gunwale_z - keel_z, 1e-6)
    hc = np.clip(d.chine_frac, 1e-3, 1.0) * depth

    ns = n_stations
    v = np.empty((ns, N_SECTION_VERTS, 2))
    v[:, 0] = np.stack([-half_beam, gunwale_z], axis=1)        # port gunwale
    v[:, 1] = np.stack([-half_beam, keel_z + hc], axis=1)      # port chine
    v[:, 2] = np.stack([-half_bottom, keel_z], axis=1)         # port bottom
    v[:, 3] = np.stack([half_bottom, keel_z], axis=1)          # stbd bottom
    v[:, 4] = np.stack([half_beam, keel_z + hc], axis=1)       # stbd chine
    v[:, 5] = np.stack([half_beam, gunwale_z], axis=1)         # stbd gunwale

    return HullMesh(
        x=x, half_bottom=half_bottom, half_beam=half_beam,
        keel_z=keel_z, gunwale_z=gunwale_z, verts=v, design=d,
    )


# ---------------------------------------------------------------------------
# Panel developability
#
# Cardboard is a flat sheet. It will bend in one direction all day and it will
# not stretch in any direction. A surface you can fold from flat stock is
# called developable, and a ruled surface is developable only if consecutive
# rulings are coplanar.
#
# Rocker, plan taper and flare each tilt the rulings differently, and combining
# all three produces a side panel that simply cannot be folded from flat board
# without darts, wrinkles, or crushing the flutes. That is a real and commonly
# fatal cardboard-boat problem, and nothing in a hydrostatics model will
# otherwise notice it -- so it gets measured here.
#
# The measure: take the quadrilateral strip of panel between two adjacent
# stations and measure how far it is from planar, normalised by the strip's
# own size. Zero means it folds flat.
#
# Worth knowing which panels can ever be a problem, because two of the three
# cannot:
#
#   Bottom panel    Its two edges are transverse lines at constant x. Two
#                   parallel lines are always coplanar, so the bottom is a
#                   cylinder no matter how much rocker or plan taper it has.
#                   Rolling a sheet is the one thing cardboard is good at.
#   Topsides        Above the chine the sides are vertical, so the panel's
#                   edges are two vertical lines -- coplanar again. Always
#                   developable, however the plan view curves.
#   Flare panel     Between the bottom edge and the chine the panel moves
#                   outboard AND upward at the same time, and the rate of each
#                   changes along the hull. This is the one that warps, and
#                   it is exactly the panel builders fight with.
#
# So combining rocker, plan taper and flare is what makes a cardboard hull
# unbuildable, and the amount of flare-panel warp is the number that says so.
# ---------------------------------------------------------------------------

def panel_warp(mesh: HullMesh) -> np.ndarray:
    """Non-planarity of each flare-panel strip, dimensionless.

    Returns one value per strip (n_stations - 1). Rough reading:
        < 0.01   folds flat, no complaint
        < 0.03   forces a little; a wet score line handles it
        < 0.06   visible wrinkling along the chine
        > 0.06   the panel wants darts. Expect crushed flutes and a leak path.
    """
    x = mesh.x
    depth = np.maximum(mesh.gunwale_z - mesh.keel_z, 1e-9)
    hc = np.clip(mesh.design.chine_frac, 1e-3, 1.0) * depth

    lower = np.stack([x, mesh.half_bottom, mesh.keel_z], axis=1)
    upper = np.stack([x, mesh.half_beam, mesh.keel_z + hc], axis=1)

    a, b = lower[:-1], lower[1:]
    c, dd = upper[:-1], upper[1:]

    # Out-of-plane distance of the 4th corner from the plane of the first 3.
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-12
    dev = np.zeros(a.shape[0])
    dev[ok] = np.abs(np.einsum("ij,ij->i", n[ok], (dd - a)[ok])) / nn[ok]

    scale = np.maximum(np.linalg.norm(dd - a, axis=1), 1e-9)
    return dev / scale


def wetted_surface(mesh: HullMesh, waterline_z: np.ndarray) -> float:
    """Wetted surface area, in^2, for a given waterline height at each station.

    Integrates the girth of the immersed part of each section along the hull.
    `waterline_z` is the water surface height above the baseline at each
    station, so trim comes along for free.
    """
    girth = np.zeros(mesh.n_stations)
    depth = np.maximum(mesh.gunwale_z - mesh.keel_z, 1e-9)
    hc = np.clip(mesh.design.chine_frac, 1e-3, 1.0) * depth

    dl = np.clip(waterline_z - mesh.keel_z, 0.0, depth)   # local draft

    # flat bottom, always wet where there is any draft at all
    girth += np.where(dl > 0, 2.0 * mesh.half_bottom, 0.0)

    # flared part of the side, up to the chine
    rise = np.minimum(dl, hc)
    run = (mesh.half_beam - mesh.half_bottom) * np.divide(
        rise, hc, out=np.zeros_like(rise), where=hc > 0
    )
    girth += 2.0 * np.hypot(rise, run)

    # vertical topsides above the chine
    girth += 2.0 * np.maximum(dl - hc, 0.0)

    return float(np.trapezoid(girth, mesh.x))


# ---------------------------------------------------------------------------
# Directional stability
#
# A canoe that will not hold a line is slow, and not because of drag. Every
# correction stroke is energy spent turning the boat instead of moving it,
# and two untrained paddlers on opposite sides with different strengths will
# be correcting constantly.
#
# What resists yaw is the submerged profile -- the lateral plane -- and
# specifically how much of it sits far from the middle, where it has leverage.
# Rocker lifts exactly that area out of the water. So rocker trades away
# directional stability for turning ability, and on a straight 140 m sprint
# turning ability is worth nothing.
#
# Nothing else in this model knows that. Without it, rocker looks almost free.
# ---------------------------------------------------------------------------

def lateral_plane(mesh: HullMesh, waterline_z: np.ndarray) -> dict:
    """Submerged profile area and how far from midships it is distributed.

    `tracking_index` is the second moment of the lateral plane about its own
    centroid, normalised so that a constant-draft hull with no rocker scores
    exactly 1.0. Less than 1 means the ends have lifted and there is less
    left to resist a yaw.

        index = [ integral d(x) (x - xc)^2 dx ] / [ A_lat * L^2 / 12 ]

    A plain rectangular profile gives L^3/12 over L * L^2/12 = 1, which is
    the reference. Heavy bow rocker with the stem clear of the water can drop
    it below 0.6.
    """
    d_local = np.clip(waterline_z - mesh.keel_z, 0.0, None)
    x = mesh.x
    w = mesh.w_int

    a_lat = float(d_local @ w)
    if a_lat < 1e-9:
        return {"area_in2": 0.0, "centroid_in": float(x.mean()),
                "tracking_index": 0.0, "lwl_in": 0.0}

    xc = float((d_local * x) @ w / a_lat)
    i_lat = float((d_local * (x - xc) ** 2) @ w)
    wet = d_local > 1e-9
    lwl = float(x[wet].max() - x[wet].min()) if wet.any() else 0.0
    ref = a_lat * max(lwl, 1e-9) ** 2 / 12.0
    return {"area_in2": a_lat, "centroid_in": xc,
            "tracking_index": i_lat / max(ref, 1e-9), "lwl_in": lwl}


def tracking_efficiency(index: float, crew_power_w, base_loss: float = 0.06,
                        rocker_slope: float = 0.35,
                        asymmetry_slope: float = 0.30) -> dict:
    """Fraction of shaft power that ends up going forwards.

    Three losses, all of them real and none of them in the drag calculation:

      base        Even in a boat that tracks perfectly, two paddlers who are
                  not racers waste some effort on steering. 6% is a modest
                  allowance for a crew who have practised a bit.
      rocker      Lost lateral plane means the boat wanders and needs
                  correcting. Scales with how far the tracking index has
                  fallen below a flat-keeled reference.
      asymmetry   Two paddlers on opposite sides with different power leave a
                  net turning moment that somebody has to cancel on every
                  stroke. Scales with the imbalance between them.

    [TYPICAL] coefficients, not measured. The STRUCTURE is right -- more
    rocker and more mismatched paddlers both cost real speed -- but the
    magnitudes are engineering judgement. Paddle a straight line and count
    your correction strokes if you want to replace them.
    """
    idx = float(np.clip(index, 0.0, 1.5))
    rocker_loss = rocker_slope * max(0.0, 1.0 - idx)

    p = np.asarray(crew_power_w, dtype=float)
    if p.size >= 2 and p.sum() > 1e-9:
        imbalance = float(abs(p.max() - p.min()) / p.sum())
    else:
        imbalance = 0.0
    asym_loss = asymmetry_slope * imbalance

    total = base_loss + rocker_loss + asym_loss
    eff = float(np.clip(1.0 - total, 0.45, 1.0))
    return {"efficiency": eff, "base_loss": base_loss,
            "rocker_loss": rocker_loss, "asymmetry_loss": asym_loss,
            "imbalance": imbalance, "tracking_index": idx}
