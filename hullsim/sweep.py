"""
Sweep drivers: generate designs, evaluate them, rank them.

Three strategies, and they are for different jobs:

  GRID        Every combination of a few named axes. Use it when you want to
              SEE a trade-off cleanly -- beam against length with everything
              else pinned gives a readable surface. Grows as the product of
              the axes, so three axes of ten points is already a thousand
              designs and five axes is a hundred thousand.

  RANDOM      Latin-hypercube sampling over many axes at once. Use it to
              explore, when you do not yet know which parameters matter. Ten
              thousand samples over fifteen dimensions tells you far more
              about the shape of the space than a grid of the same size,
              because a grid spends all its samples on a few axes.

  HILL CLIMB  Local refinement from a starting design. Use it last, to polish
              whatever the random search found.

The usual order is random, then look at the charts, then grid the two or three
axes that turned out to matter, then hill-climb the winner.

A note on what 'best' means here: the driver sorts on `ranking_score`, which
is the expected competition score with an unbridgeable penalty for every
violated constraint. That is a preference living in scoring.py, not a physical
fact. The whole results table is returned, not just the winner, precisely so
the trade-offs stay visible -- which was the point of building this rather
than an optimiser that hands back one hull and no reasoning.
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import replace
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .design import HullDesign
from .evaluate import evaluate
from .params import NOMINAL, Params
from .scoring import Limits, Rubric
from .structure import load_joint_data

_WORKER_STATE: dict = {}


def _init_worker(limits, rubric, joint_data, water_depth_ft, params):
    _WORKER_STATE.update(
        limits=limits, rubric=rubric, joint_data=joint_data,
        water_depth_ft=water_depth_ft, params=params,
    )


def _eval_one(design: HullDesign) -> dict:
    st = _WORKER_STATE
    try:
        return evaluate(
            design, detail=False,
            limits=st.get("limits"), rubric=st.get("rubric"),
            joint_data=st.get("joint_data"),
            water_depth_ft=st.get("water_depth_ft", 0.0),
            p=st.get("params", NOMINAL),
        )
    except Exception as exc:                    # one bad hull must not kill a sweep
        return {"name": design.name, "feasible": False, "n_violations": 99,
                "violations": f"evaluation error: {exc}",
                "expected_score": float("nan"), "ranking_score": -1e9}


def run(designs: Sequence[HullDesign], *, limits: Limits | None = None,
        rubric: Rubric | None = None, joint_data: dict | None = None,
        water_depth_ft: float = 0.0, residuary_scale: float | None = None,
        params: Params | None = None,
        processes: int | None = None, progress: bool = True) -> pd.DataFrame:
    """Evaluate a list of designs, in parallel, and return a sorted table."""
    limits = limits or Limits()
    rubric = rubric or Rubric()
    joint_data = joint_data if joint_data is not None else load_joint_data()
    params = params or NOMINAL
    if residuary_scale is not None:
        params = params.with_(residuary_scale=residuary_scale)

    n = len(designs)
    if processes is None:
        processes = max(1, (os.cpu_count() or 2) - 1)
    processes = min(processes, max(1, n))

    t0 = time.perf_counter()
    if processes == 1 or n < 8:
        _init_worker(limits, rubric, joint_data, water_depth_ft, params)
        rows = [_eval_one(d) for d in designs]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        with ctx.Pool(processes, initializer=_init_worker,
                      initargs=(limits, rubric, joint_data,
                                water_depth_ft, params)) as pool:
            rows = []
            chunk = max(1, n // (processes * 8))
            for i, r in enumerate(pool.imap_unordered(_eval_one, designs, chunk)):
                rows.append(r)
                if progress and (i + 1) % max(1, n // 20) == 0:
                    el = time.perf_counter() - t0
                    print(f"\r  {i + 1}/{n}  ({el:.0f}s, "
                          f"eta {el / (i + 1) * (n - i - 1):.0f}s)", end="", flush=True)
    if progress:
        print(f"\r  {n}/{n} evaluated in {time.perf_counter() - t0:.1f}s"
              f" on {processes} process(es)" + " " * 20)

    df = pd.DataFrame(rows)
    if "ranking_score" in df:
        df = df.sort_values("ranking_score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def grid(base: HullDesign, axes: dict[str, Iterable],
         filt: Callable[[HullDesign], bool] | None = None) -> list[HullDesign]:
    """Every combination of the named axes.

        grid(BASELINE, {"beam": [30, 33, 36], "length": [90, 102, 114]})

    `filt` drops combinations before they are evaluated, which is how you
    keep a grid from spending most of its budget on hulls whose beam is
    narrower than their bottom.
    """
    names = list(axes)
    out = []
    for i, combo in enumerate(itertools.product(*(list(axes[k]) for k in names))):
        kw = dict(zip(names, combo))
        d = replace(base, **kw, name=f"g{i:06d}")
        if filt and not filt(d):
            continue
        if d.validate():
            continue
        out.append(d)
    return out


def latin_hypercube(base: HullDesign, ranges: dict[str, tuple],
                    n: int, seed: int = 0,
                    filt: Callable[[HullDesign], bool] | None = None,
                    ) -> list[HullDesign]:
    """Latin-hypercube sample over continuous or discrete ranges.

    Each range is either (lo, hi) for a continuous parameter or a list/tuple
    of discrete choices. A Latin hypercube stratifies every axis
    independently, so with n samples each axis gets n distinct values however
    many axes there are -- which is exactly the property a grid lacks and the
    reason this explores a fifteen-dimensional space far better than a grid
    of the same size.
    """
    rng = np.random.default_rng(seed)
    names = list(ranges)

    # One stratified, shuffled column per axis.
    cols = {}
    for k in names:
        u = (rng.permutation(n) + rng.random(n)) / n
        cols[k] = u

    out = []
    for i in range(n):
        kw = {}
        for k in names:
            spec = ranges[k]
            u = cols[k][i]
            if isinstance(spec, tuple) and len(spec) == 2 and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in spec):
                lo, hi = spec
                val = lo + u * (hi - lo)
                if isinstance(lo, int) and isinstance(hi, int):
                    val = int(round(val))
                kw[k] = val
            else:
                choices = list(spec)
                kw[k] = choices[min(int(u * len(choices)), len(choices) - 1)]
        d = replace(base, **kw, name=f"r{i:06d}")
        if filt and not filt(d):
            continue
        if d.validate():
            continue
        out.append(d)
    return out


def hill_climb(base: HullDesign, ranges: dict[str, tuple], *,
               steps: int = 40, neighbours: int = 12, seed: int = 0,
               shrink: float = 0.88, **run_kw) -> tuple[HullDesign, pd.DataFrame]:
    """Local refinement: perturb, keep what improves, shrink the step.

    Plain stochastic hill climbing with a shrinking neighbourhood. Not a
    sophisticated optimiser, and deliberately so -- the objective has hard
    cliffs in it (constraints), the evaluation is cheap, and the answer only
    needs to be good rather than provably optimal. Run it from two or three
    different starting points if you want confidence you are not in a local
    pocket.
    """
    rng = np.random.default_rng(seed)
    names = [k for k in ranges
             if isinstance(ranges[k], tuple) and len(ranges[k]) == 2
             and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                     for v in ranges[k])]

    current = base
    df0 = run([replace(current, name="start")], progress=False, **run_kw)
    best_score = float(df0.iloc[0]["ranking_score"])
    history = [df0.iloc[0].to_dict()]
    scale = 0.30

    for step in range(steps):
        cand = []
        for j in range(neighbours):
            kw = {}
            for k in names:
                lo, hi = ranges[k]
                cur = getattr(current, k)
                span = (hi - lo) * scale
                val = float(np.clip(cur + rng.normal(0.0, span), lo, hi))
                kw[k] = int(round(val)) if isinstance(lo, int) and isinstance(hi, int) else val
            d = replace(current, **kw, name=f"h{step:03d}_{j:02d}")
            if not d.validate():
                cand.append(d)
        if not cand:
            scale *= shrink
            continue

        df = run(cand, progress=False, **run_kw)
        top = df.iloc[0]
        history.append(top.to_dict())
        if float(top["ranking_score"]) > best_score:
            best_score = float(top["ranking_score"])
            current = next(c for c in cand if c.name == top["name"])
        else:
            scale *= shrink

    return current, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------

def default_ranges() -> dict:
    """The space this project actually searches.

    Bounds chosen to bracket the team's earlier hand analysis rather than to
    be neutral -- there is no point sampling hulls three feet wide. A few
    bounds deliberately extend into territory the earlier work already showed
    was bad (very narrow bottoms, no flare) so that the failure shows up in
    the charts instead of being asserted.
    """
    return {
        "length": (80.0, 118.0),
        "bottom_width": (20.0, 36.0),
        "beam": (24.0, 42.0),
        "side_height": (9.0, 17.0),
        "bow_taper": (0.0, 36.0),
        "stern_taper": (0.0, 30.0),
        "bow_entry_exp": (0.6, 2.0),
        "bow_rocker": (0.0, 5.0),
        "stern_rocker": (0.0, 4.0),
        "bow_sheer": (0.0, 5.0),
        "stern_sheer": (0.0, 4.0),
        "chine_frac": (0.2, 1.0),
        "n_frames": (2, 10),
        "skin_layers": (1, 2),
        "free_water_lanes": (1, 3),
        "chine_tape_layers": (1, 3),
        "tape_diag_density": (0.2, 1.0),
        "tape_diag_opposing": (0.0, 0.6),
        "decked_ends": (True, False),
        # Posture is a real design choice now, not a free stability win: a
        # lower kneel drops the centre of gravity AND drops the shoulders
        # toward the rail, which costs stroke.
        "crew_posture": (("kneeling_tall",) * 2, ("kneeling_upright",) * 2,
                         ("kneeling_low",) * 2, ("back_on_heels",) * 2),
        # Where they kneel. Sliding the crew moves trim, which moves the
        # waterline, which moves everything. Bounded by knees overlapping and
        # paddles clashing.
        "crew_x_frac": ((0.34, 0.66), (0.30, 0.72), (0.27, 0.75),
                        (0.24, 0.78), (0.31, 0.69), (0.28, 0.70)),
        "pre_race_soak_min": (0.0, 12.0),
    }


def sane(d: HullDesign) -> bool:
    """Cheap filter for combinations not worth evaluating.

    Applied before the physics so a sweep does not spend its budget on hulls
    that are geometrically legal but obviously pointless.
    """
    if d.beam < d.bottom_width + 1.0:
        return False                      # needs some flare to be worth modelling
    if d.beam > d.bottom_width + 14.0:
        return False                      # flare so extreme it will not fold
    if d.bow_taper + d.stern_taper > 0.75 * d.length:
        return False
    if d.bow_rocker + 1.0 > d.side_height or d.stern_rocker + 1.0 > d.side_height:
        return False
    # One roll of board is 41 in wide. A hull whose developed girth needs far
    # more than that has to be pieced from multiple panels, which is a
    # different and much weaker build.
    girth = d.bottom_width + 2.0 * d.side_height
    if girth > 41.0 * 2.2:
        return False
    return True
