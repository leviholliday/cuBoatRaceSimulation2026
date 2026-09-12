#!/usr/bin/env python3
"""
Monte Carlo: run the same design many times and see what actually comes out.

    # One design, 5000 different builds and days
    python scripts/run_montecarlo.py --named recommended -n 5000

    # Where does the spread come from: building, racing, or not knowing?
    python scripts/run_montecarlo.py --named recommended --decompose

    # Rank the top designs from a sweep by how well they hold up
    python scripts/run_montecarlo.py --from-sweep out/sweep.csv --top 30 -n 1500

    # Leave it running overnight. Sweeps wide, then Monte Carlos the leaders,
    # checkpointing as it goes so Ctrl-C never loses work.
    python scripts/run_montecarlo.py --overnight --hours 8

WHY THIS IS DIFFERENT FROM run_sweep.py

The sweep asks "which design is best at its drawn dimensions?". This asks
"which design is still good after you have built it slightly wrong and raced
it on an ordinary day?". Those are different questions with different
answers, and the second one is the one you are actually going to live with.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from hullsim import montecarlo as mc
from hullsim import report, sweep
from hullsim.design import NAMED, HullDesign
from hullsim.evaluate import evaluate
from hullsim.params import NOMINAL, Params
from hullsim.scoring import Limits, Rubric
from hullsim.uncertainty import UncertaintyConfig

_STOP = False


def _on_sigint(signum, frame):
    global _STOP
    if _STOP:
        raise KeyboardInterrupt
    _STOP = True
    print("\n  stopping after the current design; press Ctrl-C again to "
          "abandon immediately", flush=True)


def load_designs(args) -> list[HullDesign]:
    if args.named:
        return [NAMED[args.named]]
    if args.from_sweep:
        df = pd.read_csv(args.from_sweep)
        pick = df[df.feasible] if df.feasible.any() else df
        pick = pick.head(args.top)
        return [HullDesign.from_row(r.to_dict()) for _, r in pick.iterrows()]
    return [NAMED["recommended"]]


def build_cfg(args) -> UncertaintyConfig:
    if args.sources == "all":
        return UncertaintyConfig()
    return UncertaintyConfig.only(*[s.strip() for s in args.sources.split(",")])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--trials", type=int, default=4000)
    ap.add_argument("--named", type=str, default=None, choices=sorted(NAMED))
    ap.add_argument("--from-sweep", type=str, default=None)
    ap.add_argument("--top", type=int, default=20,
                    help="how many designs from the sweep to test")
    ap.add_argument("--sources", type=str, default="all",
                    help="which uncertainty sources: all, or a comma list of "
                         "build,race,model")
    ap.add_argument("--decompose", action="store_true",
                    help="also run each source alone, to see where the "
                         "spread comes from")
    ap.add_argument("--overnight", action="store_true",
                    help="sweep wide, then Monte Carlo the leaders, "
                         "checkpointing as it goes")
    ap.add_argument("--hours", type=float, default=8.0,
                    help="time budget for --overnight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=float, default=0.0)
    ap.add_argument("--material", type=str, default=None,
                    help="board preset; see scripts/identify_material.py. "
                         "This changes the physics, not just a number: "
                         "corrugated and solid board fail in different ways.")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--out", type=str, default="out/mc")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(args)
    # The team confirmed the stock is paperboard, not corrugated, so that
    # is the default now rather than NOMINAL (which stays corrugated only
    # so old tests and --material corrugated_c_flute keep working).
    params = Params.of(args.material) if args.material else Params.of("paperboard_unknown")
    print(f"board: {params.material_name} "
          f"({params.board_caliper_in * 25.4:.2f} mm, "
          f"{'corrugated' if params.is_corrugated else 'solid'})")
    kw = dict(limits=Limits(), rubric=Rubric(), water_depth_ft=args.depth,
              params=params, processes=args.processes)

    if args.overnight:
        return overnight(args, outdir, cfg, kw)

    designs = load_designs(args)
    signal.signal(signal.SIGINT, _on_sigint)

    # ---- many designs: rank by robustness -------------------------------
    if len(designs) > 1:
        print(f"Monte Carlo on {len(designs)} designs, {args.trials:,} trials "
              f"each ({len(designs) * args.trials:,} evaluations)")
        rows, keep = [], {}
        for i, d in enumerate(designs):
            if _STOP:
                print("  stopped early")
                break
            print(f"  [{i + 1}/{len(designs)}] {d.name}")
            df = mc.run_trials(d, args.trials, cfg=cfg, seed=args.seed + i * 1013,
                               progress=True, **kw)
            rows.append(mc.summarise(df, d))
            keep[d.name] = df
            pd.DataFrame(rows).to_csv(outdir / "robust_summary.csv", index=False)

        summary = (pd.DataFrame(rows).sort_values("robust_score", ascending=False)
                   .reset_index(drop=True))
        summary.to_csv(outdir / "robust_summary.csv", index=False)

        best = summary.iloc[0]
        df_best = keep[best["name"]]
        df_best.to_csv(outdir / "trials_best.csv", index=False)

        path = report.write_report(
            outdir / "montecarlo.html",
            title="Robustness ranking",
            subtitle=(f"{len(summary)} designs × {args.trials:,} trials · "
                      f"sources: {', '.join(cfg.active)}"),
            sections=[("mc", report.montecarlo_html(
                df_best, None, mc.summarise(df_best), comparison=summary))],
        )
        print(f"\n  summary   {outdir / 'robust_summary.csv'}")
        print(f"  dashboard {path}")
        print(f"\n  most robust: {best['name']}")
        print(f"    {best['length']:.0f} x {best['bottom_width']:.0f} bottom x "
              f"{best['beam']:.0f} beam x {best['side_height']:.0f} deep")
        print(f"    passes {best['p_feasible']:.0%} of builds, "
              f"median {best['time_p50']:.0f} s, bad-day score "
              f"{best['robust_score']:.1f}")
        _print_disagreement(summary)
        return 0

    # ---- one design: full distribution ----------------------------------
    d = designs[0]
    nominal = evaluate(d, detail=False, limits=kw["limits"], rubric=kw["rubric"],
                       water_depth_ft=args.depth, p=params)
    print(f"Monte Carlo on {d.name}: {args.trials:,} trials, "
          f"sources {', '.join(cfg.active)}")
    print(f"  nominal (no uncertainty): {nominal['time_s']:.1f} s, "
          f"score {nominal['expected_score']:.1f}, "
          f"{'feasible' if nominal['feasible'] else 'FAILS'}")

    df = mc.run_trials(d, args.trials, cfg=cfg, seed=args.seed, progress=True, **kw)
    s = mc.summarise(df, d)
    df.to_csv(outdir / f"trials_{d.name}.csv", index=False)

    by_source = None
    if args.decompose:
        by_source = {}
        for src in ("build", "race", "model"):
            print(f"  decomposing: {src} only")
            by_source[src] = mc.run_trials(
                d, max(args.trials // 2, 400), cfg=UncertaintyConfig.only(src),
                seed=args.seed + 97, progress=True, **kw)
        by_source["all"] = df

    path = report.write_report(
        outdir / f"montecarlo_{d.name}.html",
        title=f"Monte Carlo · {d.name}",
        subtitle=(f"{args.trials:,} trials · sources: {', '.join(cfg.active)} · "
                  f"nominal {nominal['time_s']:.1f} s / "
                  f"score {nominal['expected_score']:.1f}"),
        sections=[("mc", report.montecarlo_html(df, d, s, by_source=by_source))],
    )

    print(f"\n  nominal score {nominal['expected_score']:.1f}   "
          f"mean {s['score_mean']:.1f}   bad day {s['score_p10']:.1f}")
    print(f"  time {s['time_p50']:.1f} s median, {s['time_p10']:.0f}-"
          f"{s['time_p90']:.0f} s across nine trials in ten")
    print(f"  {s['p_feasible']:.0%} of builds pass every constraint")
    fr = mc.failure_rates(df)
    if len(fr):
        print("  what goes wrong:")
        for k, v in fr.head(4).items():
            print(f"    {v:5.1%}  {k}")
    print("\n  biggest drivers of the spread in score:")
    for _, r in mc.sensitivity(df, "expected_score").head(4).iterrows():
        print(f"    {r['spearman']:+.2f}  {report._pretty_input(r['input'])}")
    print(f"\n  trials    {outdir / f'trials_{d.name}.csv'}")
    print(f"  dashboard {path}")
    return 0


def _print_disagreement(summary: pd.DataFrame) -> None:
    """Point out where the robust ranking disagrees with the nominal one."""
    if "score_mean" not in summary or len(summary) < 5:
        return
    by_mean = summary.sort_values("score_mean", ascending=False)
    if by_mean.iloc[0]["name"] != summary.iloc[0]["name"]:
        print(f"\n  Worth noting: ranked on the AVERAGE score the winner "
              f"would be {by_mean.iloc[0]['name']}, which passes only "
              f"{by_mean.iloc[0]['p_feasible']:.0%} of builds. Ranking on the "
              f"bad-day score prefers {summary.iloc[0]['name']} instead.")


def overnight(args, outdir: Path, cfg, kw) -> int:
    """Sweep wide, then Monte Carlo the leaders, inside a time budget.

    Checkpoints after every design, so stopping it at any point leaves a
    usable partial result rather than nothing. Re-running with the same
    output directory picks up where it left off.
    """
    signal.signal(signal.SIGINT, _on_sigint)
    budget_s = args.hours * 3600.0
    t0 = time.perf_counter()
    state_path = outdir / "overnight_state.json"
    sweep_csv = outdir / "stage1_sweep.csv"
    summary_csv = outdir / "robust_summary.csv"

    print(f"Overnight run, budget {args.hours:.1f} h, output {outdir}")

    # ---- stage 1: a wide deterministic sweep ----------------------------
    if sweep_csv.exists():
        sdf = pd.read_csv(sweep_csv)
        print(f"  stage 1: reusing {len(sdf):,} designs from {sweep_csv}")
    else:
        base = NAMED["recommended"]
        n_req = 60000
        print(f"  stage 1: sampling {n_req:,} designs")
        designs = sweep.latin_hypercube(base, sweep.default_ranges(), n_req,
                                        seed=args.seed, filt=sweep.sane)
        sdf = sweep.run(designs, limits=kw["limits"], rubric=kw["rubric"],
                        water_depth_ft=kw["water_depth_ft"],
                        processes=kw["processes"])
        sdf.to_csv(sweep_csv, index=False)
    feas = sdf[sdf.feasible]
    print(f"  stage 1: {len(feas):,} feasible of {len(sdf):,}")
    if not len(feas):
        print("  nothing feasible; widen the ranges or relax a limit")
        return 1

    # ---- size stage 2 to the remaining budget ---------------------------
    elapsed = time.perf_counter() - t0
    left = max(budget_s - elapsed, 60.0)
    rate = _measure_rate(feas, cfg, kw, args)
    total_trials = int(left * rate * 0.92)           # leave room for reporting

    done = {}
    if state_path.exists():
        done = json.loads(state_path.read_text()).get("done", {})
        print(f"  resuming: {len(done)} designs already done")

    n_designs = max(1, min(len(feas), int(total_trials / max(args.trials, 1))))
    per_design = max(400, min(args.trials, total_trials // max(n_designs, 1)))
    print(f"  stage 2: {n_designs} designs x {per_design:,} trials "
          f"= {n_designs * per_design:,} evaluations in ~{left / 3600:.1f} h")

    rows = list(done.values())
    for i in range(n_designs):
        if _STOP or time.perf_counter() - t0 > budget_s:
            print("  budget reached" if not _STOP else "  stopped")
            break
        row = feas.iloc[i]
        name = str(row["name"])
        if name in done:
            continue
        d = HullDesign.from_row(row.to_dict())
        el = time.perf_counter() - t0
        print(f"  [{i + 1}/{n_designs}] {name}  "
              f"({el / 3600:.2f}h elapsed, {(budget_s - el) / 3600:.2f}h left)")
        df = mc.run_trials(d, per_design, cfg=cfg, seed=args.seed + i * 1013,
                           progress=True, **kw)
        summ = mc.summarise(df, d)
        rows.append(summ)
        done[name] = summ
        state_path.write_text(json.dumps({"done": done}, default=float))
        pd.DataFrame(rows).sort_values("robust_score", ascending=False).to_csv(
            summary_csv, index=False)
        if i == 0:
            df.to_csv(outdir / "trials_first.csv", index=False)

    summary = (pd.DataFrame(rows).sort_values("robust_score", ascending=False)
               .reset_index(drop=True))
    summary.to_csv(summary_csv, index=False)

    # Re-run the winner with more trials for a clean headline distribution.
    best_name = str(summary.iloc[0]["name"])
    best_row = feas[feas.name.astype(str) == best_name]
    best_design = (HullDesign.from_row(best_row.iloc[0].to_dict())
                   if len(best_row) else NAMED["recommended"])
    print(f"  final: {min(per_design * 3, 20000):,} trials on the winner")
    df_best = mc.run_trials(best_design, min(per_design * 3, 20000), cfg=cfg,
                            seed=args.seed + 7717, progress=True, **kw)
    df_best.to_csv(outdir / "trials_best.csv", index=False)

    path = report.write_report(
        outdir / "overnight.html",
        title="Overnight robustness study",
        subtitle=(f"{len(sdf):,} designs swept · {len(summary)} Monte Carloed "
                  f"× {per_design:,} trials · "
                  f"{(time.perf_counter() - t0) / 3600:.1f} h"),
        sections=[("mc", report.montecarlo_html(
            df_best, best_design, mc.summarise(df_best, best_design),
            comparison=summary))],
    )
    print(f"\n  summary   {summary_csv}")
    print(f"  dashboard {path}")
    b = summary.iloc[0]
    print(f"\n  most robust: {b['name']}  "
          f"{b['length']:.0f} x {b['bottom_width']:.0f} x {b['beam']:.0f} x "
          f"{b['side_height']:.0f}")
    print(f"    passes {b['p_feasible']:.0%} of builds, median "
          f"{b['time_p50']:.0f} s, bad-day score {b['robust_score']:.1f}")
    _print_disagreement(summary)
    return 0


def _measure_rate(feas: pd.DataFrame, cfg, kw, args) -> float:
    """Trials per second, measured rather than guessed, so the time budget
    means something on whatever machine this is running on."""
    d = HullDesign.from_row(feas.iloc[0].to_dict())
    n = 300
    t = time.perf_counter()
    mc.run_trials(d, n, cfg=cfg, seed=1, progress=False, **kw)
    rate = n / max(time.perf_counter() - t, 1e-6)
    print(f"  measured {rate:.0f} trials/s on this machine")
    return rate


if __name__ == "__main__":
    raise SystemExit(main())
