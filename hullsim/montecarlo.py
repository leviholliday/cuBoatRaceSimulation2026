"""
Monte Carlo: run the same design many times and see what comes out.

Each trial draws one plausible built boat, one plausible race day and one
plausible set of true material properties, then runs the ordinary
deterministic model on it. Nothing here randomises a result. The physics is
unchanged; only the inputs move.

WHAT YOU GET THAT A SINGLE RUN CANNOT GIVE YOU

  A spread instead of a number.  "74.9 s" becomes "74.9 s, and nine times in
  ten between 68 and 86". The second statement is the one you can plan
  around, and it is the honest one.

  P(feasible) -- robustness.  A design can score beautifully at its nominal
  dimensions and fall apart half an inch off them. That is not a good design,
  it is a lucky point in parameter space, and only resampling the build finds
  it. This is the single most useful output in the module.

  A ranked list of what the answer is most sensitive to.  Because every
  sampled input is recorded alongside the outcome, the correlation between
  them says which unknown is actually driving the spread. That converts
  "there is a lot of uncertainty" into "go and weigh a square of cardboard,
  and time one paddle, and stop worrying about the rest" -- which is a
  shopping list rather than a shrug.

  Which constraint fails, and how often.  Not "it might fail" but "it goes
  over the tape budget in 18% of builds and never fails anything else", which
  tells you what to change.

A WARNING ABOUT READING THE PROBABILITIES

P(feasible) = 0.93 does not mean a 93% chance of finishing the race. It means
that, given the spreads written down in uncertainty.py, 93% of the boats you
might build on the days you might race pass every hard constraint in
scoring.py. Those spreads are considered judgements, not measured
distributions. Compare designs with these numbers; do not quote them as
predictions.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .design import HullDesign
from .evaluate import evaluate
from .params import NOMINAL, Params
from .scoring import Limits, Rubric
from .structure import load_joint_data
from .uncertainty import UncertaintyConfig, sample

# Columns worth keeping from each trial. Keeping all ~90 would make a 2-million
# row table nobody can open.
KEEP = (
    "feasible", "n_violations", "violations",
    "time_s", "terminal_v_ms", "froude",
    "draft_in", "freeboard_in", "trim_deg",
    "gm_in", "gm_effective_in", "gz_max_in", "gz_demand_in",
    "gz_margin_ratio", "downflood_deg", "usable_heel_deg", "heel_at_rest_deg",
    "gz_area", "paddle_heel_deg", "effective_freeboard_in",
    "swamp_reserve_gal", "expected_shipping_events",
    "boarding_freeboard_in", "boarding_heel_deg",
    "boarding_freeboard_with_chop_in", "boarding_heel_with_chop_deg",
    "roll_racing_deg", "roll_stopped_deg", "roll_period_s",
    "tracking_index", "tracking_efficiency", "effective_shaft_w",
    "reach_efficiency", "reach_clearance_in", "crew_spacing_in",
    "crew_kg_mean", "minutes_afloat", "wet_factor_at_finish",
    "panel_warp", "girder_utilisation", "twist_deg", "chine_tape_util",
    "panel_deflection_in", "hull_weight_lb", "all_up_lb",
    "board_spare_frac", "tape_spare_frac",
    "p_upright", "p_dry", "p_intact", "p_finish", "expected_score",
)

_W: dict = {}


def _init_worker(design, params, cfg, limits, rubric, joint_data,
                 water_depth_ft):
    _W.update(design=design, params=params, cfg=cfg, limits=limits,
              rubric=rubric, joint_data=joint_data,
              water_depth_ft=water_depth_ft)


def _run_block(args) -> list[dict]:
    """One worker's slice of trials. Seeded so the whole run is reproducible."""
    seed, n = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d, p, rec = sample(_W["design"], _W["params"], rng, _W["cfg"])
        try:
            r = evaluate(d, detail=False, limits=_W["limits"],
                         rubric=_W["rubric"], joint_data=_W["joint_data"],
                         water_depth_ft=_W["water_depth_ft"], p=p)
        except Exception as exc:                 # a bad draw must not kill the run
            r = {"feasible": False, "n_violations": 99,
                 "violations": f"evaluation error: {exc}",
                 "expected_score": float("nan"), "time_s": float("nan")}
        row = {k: r.get(k, float("nan")) for k in KEEP}
        row.update(rec)
        # A few of the perturbed design's own dimensions, so the sensitivity
        # analysis can see build error directly rather than only through the
        # deltas recorded by the sampler.
        row["b_length"] = d.length
        row["b_beam"] = d.beam
        row["b_bottom_width"] = d.bottom_width
        row["b_side_height"] = d.side_height
        out.append(row)
    return out


def run_trials(design: HullDesign, n_trials: int = 2000, *,
               cfg: UncertaintyConfig | None = None,
               params: Params | None = None,
               limits: Limits | None = None, rubric: Rubric | None = None,
               joint_data: dict | None = None, water_depth_ft: float = 0.0,
               seed: int = 0, processes: int | None = None,
               progress: bool = False) -> pd.DataFrame:
    """Run `n_trials` perturbed evaluations of one design."""
    cfg = cfg or UncertaintyConfig()
    params = params or NOMINAL
    limits = limits or Limits()
    rubric = rubric or Rubric()
    joint_data = joint_data if joint_data is not None else load_joint_data()

    if processes is None:
        processes = max(1, (os.cpu_count() or 2) - 1)
    processes = min(processes, max(1, n_trials // 50))

    # Each worker gets its own seed stream, so the result depends on `seed`
    # and the number of trials, never on how many cores happened to be free.
    per = max(1, math.ceil(n_trials / max(processes, 1)))
    blocks = []
    left = n_trials
    i = 0
    while left > 0:
        take = min(per, left)
        blocks.append((seed * 1_000_003 + i * 7919 + 11, take))
        left -= take
        i += 1

    t0 = time.perf_counter()
    if processes <= 1 or len(blocks) == 1:
        _init_worker(design, params, cfg, limits, rubric, joint_data,
                     water_depth_ft)
        rows = []
        for b in blocks:
            rows.extend(_run_block(b))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        with ctx.Pool(processes, initializer=_init_worker,
                      initargs=(design, params, cfg, limits, rubric,
                                joint_data, water_depth_ft)) as pool:
            rows = []
            for j, part in enumerate(pool.imap_unordered(_run_block, blocks)):
                rows.extend(part)
                if progress:
                    print(f"\r    {len(rows)}/{n_trials} trials", end="",
                          flush=True)
    if progress:
        print(f"\r    {len(rows)} trials in {time.perf_counter() - t0:.1f}s" + " " * 12)

    df = pd.DataFrame(rows)
    df.attrs["design_name"] = design.name
    df.attrs["sources"] = ",".join(cfg.active)
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame, design: HullDesign | None = None) -> dict:
    """Collapse a trial table into one row of robustness statistics."""
    n = len(df)
    ok = df[df.feasible == True]  # noqa: E712 -- may be object dtype after CSV

    def q(col, p):
        s = df[col].dropna()
        return float(np.percentile(s, p)) if len(s) else float("nan")

    def m(col):
        s = df[col].dropna()
        return float(s.mean()) if len(s) else float("nan")

    out = {
        "name": design.name if design else df.attrs.get("design_name", ""),
        "n_trials": n,
        "sources": df.attrs.get("sources", ""),
        "p_feasible": len(ok) / n if n else float("nan"),

        "time_mean": m("time_s"), "time_sd": float(df.time_s.std()),
        "time_p10": q("time_s", 10), "time_p50": q("time_s", 50),
        "time_p90": q("time_s", 90),

        "score_mean": m("expected_score"),
        "score_p10": q("expected_score", 10),
        "score_p50": q("expected_score", 50),
        "score_p90": q("expected_score", 90),

        "p_finish_mean": m("p_finish"),
        "gm_p10": q("gm_in", 10), "gm_p50": q("gm_in", 50),
        "freeboard_p10": q("freeboard_in", 10),
        "gz_margin_p10": q("gz_margin_ratio", 10),
        "swamp_reserve_p10": q("swamp_reserve_gal", 10),
        "warp_p90": q("panel_warp", 90),
        "twist_p90": q("twist_deg", 90),
        "tape_spare_p10": q("tape_spare_frac", 10),
        "board_spare_p10": q("board_spare_frac", 10),
    }

    # Monte Carlo error bars. Everything above is an estimate from a finite
    # number of draws, and quoting p_feasible = 0.93 without saying whether
    # that is 0.93 +/- 0.01 or +/- 0.08 invites reading noise as signal.
    #
    # The proportion gets the binomial standard error; the percentiles get
    # the asymptotic order-statistic one, which needs the local density and
    # is therefore approximated from the interquartile spread. Both are good
    # enough to answer the only question they need to: is this difference
    # between two designs real, or is it sampling noise?
    if n > 1:
        pf = out["p_feasible"]
        out["p_feasible_se"] = math.sqrt(max(pf * (1 - pf), 0.0) / n)
        sc = df["expected_score"].dropna()
        sc = sc[np.isfinite(sc)]
        if len(sc) > 10:
            iqr = float(np.percentile(sc, 75) - np.percentile(sc, 25))
            dens = max(iqr, 1e-9) / 1.349            # ~ sd for a normal
            out["score_p10_se"] = float(1.71 * dens / math.sqrt(len(sc)))
            out["score_mean_se"] = float(sc.std() / math.sqrt(len(sc)))
        else:
            out["score_p10_se"] = float("nan")
            out["score_mean_se"] = float("nan")
        tm = df["time_s"].dropna()
        tm = tm[np.isfinite(tm)]
        out["time_mean_se"] = (float(tm.std() / math.sqrt(len(tm)))
                               if len(tm) > 1 else float("nan"))
    else:
        out["p_feasible_se"] = out["score_p10_se"] = float("nan")
        out["score_mean_se"] = out["time_mean_se"] = float("nan")

    # Robustness-weighted score. score_p10 is the bad-day answer -- a tenth of
    # the boats you might build and days you might race come out at or below
    # it. Ranking on this rather than on the mean is what makes the choice
    # cautious in the way the brief asked for.
    out["robust_score"] = out["score_p10"]

    if design is not None:
        out["crew_posture"] = "|".join(design.crew_posture)
        out["crew_x_frac"] = "|".join(f"{v:g}" for v in design.crew_x_frac)
        for f in ("length", "bottom_width", "beam", "side_height",
                  "bow_rocker", "chine_frac", "n_frames", "free_water_lanes",
                  "chine_tape_layers", "skin_layers", "decked_ends",
                  "bow_taper", "bow_sheer", "tape_diag_density",
                  "pre_race_soak_min", "pacing"):
            out[f] = getattr(design, f)
    return out


def failure_rates(df: pd.DataFrame) -> pd.Series:
    """How often each constraint is the (or a) reason for failure."""
    bad = df[df.feasible != True]  # noqa: E712
    if not len(bad):
        return pd.Series(dtype=float)
    s = (bad.violations.astype(str).str.split("; ").explode()
         .str.replace(r"-?[\d.]+", "N", regex=True)
         .value_counts() / len(df))
    return s.sort_values(ascending=False)


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def sensitivity(df: pd.DataFrame, target: str = "expected_score",
                min_unique: int = 5) -> pd.DataFrame:
    """Which sampled input moves `target` most. Spearman rank correlation.

    Rank correlation rather than Pearson because several of these
    relationships are strongly monotone but not remotely linear -- residuary
    scale against crossing time, for one. Rank correlation measures "does
    more of this reliably mean more of that", which is the question, and it
    is not thrown by the long tail of a lognormal.

    The sign says direction; the magnitude says how much of the spread that
    one input is responsible for. Read the top three and ignore the rest.
    """
    cols = [c for c in df.columns if c.startswith(("u_", "b_"))]
    y = df[target].to_numpy(dtype=float)
    good = np.isfinite(y)
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        mask = good & np.isfinite(x)
        if mask.sum() < 30 or len(np.unique(x[mask])) < min_unique:
            continue
        rx, ry = _rank(x[mask]), _rank(y[mask])
        sx, sy = rx.std(), ry.std()
        if sx < 1e-12 or sy < 1e-12:
            continue
        rho = float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))
        rows.append({"input": c, "spearman": rho, "abs": abs(rho),
                     "n": int(mask.sum())})
    out = pd.DataFrame(rows)
    return (out.sort_values("abs", ascending=False).reset_index(drop=True)
            if len(out) else out)


# ---------------------------------------------------------------------------
# Comparing designs
# ---------------------------------------------------------------------------

@dataclass
class RobustComparison:
    summary: pd.DataFrame          # one row per design
    trials: dict                   # name -> trial DataFrame


def compare(designs, n_trials: int = 1000, *, cfg=None, seed: int = 0,
            progress: bool = True, keep_trials: bool = True,
            **kw) -> RobustComparison:
    """Monte Carlo several designs and rank them by how well they hold up.

    This is where a deterministic sweep and a probabilistic one disagree, and
    the disagreement is the point: designs that top the nominal ranking are
    regularly NOT the ones that survive being built slightly wrong.
    """
    rows, trials = [], {}
    for i, d in enumerate(designs):
        if progress:
            print(f"  [{i + 1}/{len(designs)}] {d.name or 'design'}")
        df = run_trials(d, n_trials, cfg=cfg, seed=seed + i * 1013,
                        progress=progress, **kw)
        rows.append(summarise(df, d))
        if keep_trials:
            trials[d.name] = df
    out = pd.DataFrame(rows).sort_values("robust_score", ascending=False)
    return RobustComparison(summary=out.reset_index(drop=True), trials=trials)
