"""
Uncertainty: turning one answer into a distribution of answers.

THE IDEA, AND WHY IT IS NOT JUST ADDING NOISE

Running the same design twice should give different numbers, because building
the same design twice gives you two different boats and racing on two
different days gives you two different races. The physics is not random. The
INPUTS are. So nothing here perturbs a result -- everything here perturbs an
input and then runs the same deterministic model on it.

That distinction is what makes the output mean something. A spread of five
seconds is not the model being vague; it is the honest consequence of not
knowing your cardboard's stiffness to better than thirty percent and not
knowing how hard you will paddle on the day to better than fifteen.

THREE KINDS OF UNCERTAINTY, KEPT SEPARATE ON PURPOSE

They are separate because you can do something different about each one.

  BuildVariation    The boat you drew versus the boat you cut out. Tape
                    measures, utility knives, and a panel that ended up half
                    an inch narrow. You reduce this with care and jigs.
                    Turning it on alone answers: "is this design forgiving of
                    sloppy building?" -- which is a real design property, and
                    arguably the most useful thing in this whole module.

  RaceDayVariation  How hard you actually paddle, how you actually kneel, what
                    the water is doing, how much splash comes aboard while you
                    get in. You reduce this with practice and by picking your
                    moment. Turning it on alone answers: "how much does the
                    day matter?"

  ModelUncertainty  What nobody knows: your board's real stiffness, the real
                    wave-making of a shoebox. This is ignorance, not
                    variability -- the true value is fixed, we just do not
                    have it. You reduce it with MEASUREMENTS, and the
                    sensitivity output tells you which measurement buys the
                    most.

Mixing all three and reporting one spread would hide that. Keep them separable
and the output tells you whether to buy a better ruler, practise more, or go
weigh a square of cardboard.

CORRELATION MATTERS AND IS EASY TO GET WRONG

Board stiffness, edge crush strength and thickness are properties of ONE piece
of cardboard. Sampling them independently pretends you might get stiff board
that is also weak and thin, which is not a thing. So they share a common
"board quality" factor with independent noise on top. Ignoring that would make
the predicted spread too narrow, which is the worse error: it would make the
model look more confident than it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from .design import HullDesign
from .params import Params, _solid_board

_NORM = NormalDist()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lognorm(rng, cv: float) -> float:
    """A positive multiplier with mean ~1 and coefficient of variation `cv`.

    Lognormal rather than normal because every quantity it is applied to is
    physically positive, and a normal with a 45% CV will hand you a negative
    shear modulus about one time in fifty.
    """
    if cv <= 0:
        return 1.0
    sigma = math.sqrt(math.log1p(cv * cv))
    return float(math.exp(rng.normal(-0.5 * sigma * sigma, sigma)))


def worst_of_n_normal(rng, mean: float, sd: float, n: int) -> float:
    """The largest of `n` draws from Normal(mean, sd), in one random number.

    Used for things that happen repeatedly during a race and only have to go
    wrong once -- a paddle stroke that leans too far. Over 150 strokes the
    worst one is roughly 2.2 standard deviations out, so designing to the
    AVERAGE stroke is designing to the wrong stroke.

    Exact rather than approximate: the maximum of n independent draws has CDF
    F(x)^n, so sampling u uniform and inverting at u^(1/n) is a draw from the
    maximum's distribution.
    """
    if n <= 1 or sd <= 0:
        return mean + (rng.normal(0.0, sd) if sd > 0 else 0.0)
    u = float(rng.random())
    u = min(max(u, 1e-12), 1 - 1e-12)
    return mean + sd * _NORM.inv_cdf(u ** (1.0 / n))


# ---------------------------------------------------------------------------
# the three configs
# ---------------------------------------------------------------------------

@dataclass
class BuildVariation:
    """The boat you drew versus the boat you built.

    Standard deviations in inches, for a hull cut by hand from a roll with a
    tape measure and a utility knife. These are judgement calls, not measured
    build tolerances -- if you cut a few test panels and measure the error,
    replace them.
    """
    enabled: bool = True

    length_sd: float = 0.75
    """Accumulates over a long cut, so larger than the others."""
    bottom_width_sd: float = 0.50
    beam_sd: float = 0.50
    side_height_sd: float = 0.40
    rocker_sd: float = 0.40
    """Rocker is set by how the bottom is persuaded to curve, not by a
       measured cut, so it is among the least repeatable dimensions."""
    sheer_sd: float = 0.30
    chine_frac_sd: float = 0.05
    taper_sd: float = 1.50

    asymmetry_sd: float = 0.35
    """Transverse centre-of-gravity error from one side ending up wider,
       heavier or better taped than the other.

       Worth understanding because it is invisible until you float her: an
       asymmetric hull sits with a permanent list, and that list subtracts
       from the righting arm available for everything else, on every stroke,
       for the whole race. Half an inch of transverse error is a large
       fraction of what one paddler's lean already demands."""

    extra_weight_mean: float = 1.0
    extra_weight_sd: float = 1.0
    """Patches, doubled seams, overlap you did not plan, glue. Always adds,
       never subtracts, so this is clipped at zero."""

    drop_a_frame_prob: float = 0.15
    skimp_chine_tape_prob: float = 0.12
    tape_density_sd: float = 0.08
    wet_factor_sd: float = 0.06
    """Workmanship in sealing seams: how much water the board takes up."""


@dataclass
class RaceDayVariation:
    """Crew and conditions on the day."""
    enabled: bool = True

    power_cv: float = 0.15
    bad_day_prob: float = 0.10
    bad_day_factor: float = 0.75
    """Someone is ill, nervous, or went out too hard and died at 80 m."""

    pacing_slip_prob: float = 0.25
    """Chance the crew goes out harder than they planned.

       Adrenaline off the line is real and it is the most common way a
       planned pace becomes a different one. A slip bumps the strategy one
       step harder: even becomes a fast start, a fast start becomes a sprint.
       Modelled as a race-day variable rather than a design choice because
       nobody decides to do it."""

    anaerobic_fraction_cv: float = 0.20
    """How much of their power is burnable reserve varies between people and
       between days, and it decides how long a hard start can be held before
       it turns into a fade."""

    crew_kg_sd: float = 1.2
    """Posture drifts. Nobody holds the position they planned for 75 seconds
       under load."""

    stroke_lean_sd: float = 0.8
    strokes_per_race: int = 150
    """The lean that matters is not the average stroke, it is the worst one.
       See worst_of_n_normal."""

    chop_levels: tuple = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    chop_weights: tuple = (0.15, 0.25, 0.25, 0.18, 0.12, 0.05)

    soak_extra_min_mean: float = 2.0
    soak_extra_min_sd: float = 2.5
    """How much longer than planned she ends up floating before the gun --
       a slow launch, a delayed start, a boat ahead that needs fishing out.
       Always adds, and it costs real strength."""

    wind_mean_mph: float = 4.0
    wind_sd_mph: float = 3.5
    """Wind along the course on the day, signed: a draw can be a tailwind.
       Modest but not nothing -- fifteen miles an hour of headwind is worth
       several seconds."""

    splash_aboard_mean_gal: float = 0.30
    """Water that comes in while boarding and during the first few strokes,
       before anyone thinks to bail. Exponentially distributed: usually
       nothing, occasionally a lot."""

    boarding_offset_lo: float = 0.15
    boarding_offset_hi: float = 0.45

    water_temp_nu_cv: float = 0.08


@dataclass
class ModelUncertainty:
    """What nobody knows, as opposed to what varies.

    These are the [TYPICAL] and [MEASURE] constants. The sensitivity output
    ranks them, which turns this from hand-wringing into a shopping list of
    measurements worth taking.
    """
    enabled: bool = True

    board_common_cv: float = 0.18
    """Shared 'is this good cardboard' factor. Induces the correlation between
       stiffness, strength and thickness that sampling them independently
       would wrongly destroy."""
    board_caliper_cv: float = 0.05
    board_areal_cv: float = 0.10
    board_ect_cv: float = 0.18
    board_stiffness_cv: float = 0.22
    tape_tensile_cv: float = 0.25
    g_board_cv: float = 0.45
    """The single widest spread in here. The shear modulus of smeared
       corrugated board is barely pinned down at all, and it scales twist
       directly."""
    shell_knockdown_cv: float = 0.20
    flat_crush_cv: float = 0.20
    joint_strength_cv: float = 0.25

    paperboard_thickness_range_mm: tuple = (1.0, 2.4)
    """The team's own description: 'a little thinner than a cereal box... 1
       to 2 and some mm'. Rather than jitter a single guessed thickness by a
       few percent, every trial draws a fresh thickness from this whole
       range -- so a hull that only works at, say, 1.2mm and fails at 2.0mm
       shows up as fragile in the robustness score, which is exactly the
       point of running this instead of trusting one guess."""
    paperboard_density_range_gcm3: tuple = (0.65, 0.95)
    paperboard_stiffness_mult_range: tuple = (0.9, 1.6)
    """The board springs back toward the roll's curvature after being cut,
       which is real elastic memory a limp folding-carton stock would not
       show. That pushes the plausible modulus above a generic paperboard's,
       hence a range centred above 1.0 rather than symmetric around it."""
    paperboard_strength_mult_range: tuple = (0.8, 1.3)

    residuary_lognormal_sigma: float = 0.50
    """NOT a coefficient of variation -- the log standard deviation, which for
       0.50 spans roughly x0.6 to x1.6 at one sigma and x0.37 to x2.7 at two.
       Deliberately wide: the residuary curve is a standard series fitted to
       displacement hulls, extrapolated to a cardboard shoebox. Calibrating
       against a timed paddle is what collapses this, and it is the highest-
       value measurement available."""
    form_factor_cv: float = 0.12
    blade_efficiency_cv: float = 0.08
    max_thrust_cv: float = 0.15


@dataclass
class UncertaintyConfig:
    build: BuildVariation = field(default_factory=BuildVariation)
    race: RaceDayVariation = field(default_factory=RaceDayVariation)
    model: ModelUncertainty = field(default_factory=ModelUncertainty)

    @classmethod
    def only(cls, *which: str) -> "UncertaintyConfig":
        """A config with only the named sources switched on.

            UncertaintyConfig.only("build")          # is it forgiving to build?
            UncertaintyConfig.only("model")          # how much does ignorance cost?
            UncertaintyConfig.only("build", "race")  # everything we cannot measure away
        """
        cfg = cls()
        cfg.build.enabled = "build" in which
        cfg.race.enabled = "race" in which
        cfg.model.enabled = "model" in which
        return cfg

    @property
    def active(self) -> list[str]:
        return [n for n, c in (("build", self.build), ("race", self.race),
                               ("model", self.model)) if c.enabled]


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------

def sample(design: HullDesign, params: Params, rng: np.random.Generator,
           cfg: UncertaintyConfig) -> tuple[HullDesign, Params, dict]:
    """Draw one plausible (built boat, race day, true material) triple.

    Returns the perturbed design, the perturbed material parameters, and a
    record of every value drawn. The record is what makes sensitivity
    analysis possible afterwards: with the inputs and the outcome side by
    side you can ask which input actually moved the answer.
    """
    rec: dict = {}
    d_kw: dict = {}
    p_kw: dict = {}

    # ---------------- build ----------------
    b = cfg.build
    if b.enabled:
        def jitter(name, value, sd, lo=None):
            v = value + rng.normal(0.0, sd)
            if lo is not None:
                v = max(v, lo)
            rec["u_" + name] = v - value
            return v

        d_kw["length"] = jitter("length", design.length, b.length_sd, 24.0)
        d_kw["bottom_width"] = jitter("bottom_width", design.bottom_width,
                                      b.bottom_width_sd, 6.0)
        d_kw["beam"] = max(jitter("beam", design.beam, b.beam_sd, 8.0),
                           d_kw["bottom_width"] + 0.25)
        d_kw["side_height"] = jitter("side_height", design.side_height,
                                     b.side_height_sd, 4.0)
        d_kw["bow_rocker"] = jitter("bow_rocker", design.bow_rocker,
                                    b.rocker_sd, 0.0)
        d_kw["stern_rocker"] = jitter("stern_rocker", design.stern_rocker,
                                      b.rocker_sd, 0.0)
        d_kw["bow_sheer"] = jitter("bow_sheer", design.bow_sheer, b.sheer_sd, 0.0)
        d_kw["stern_sheer"] = jitter("stern_sheer", design.stern_sheer,
                                     b.sheer_sd, 0.0)
        d_kw["bow_taper"] = jitter("bow_taper", design.bow_taper, b.taper_sd, 0.0)
        d_kw["stern_taper"] = jitter("stern_taper", design.stern_taper,
                                     b.taper_sd, 0.0)
        d_kw["chine_frac"] = float(np.clip(
            jitter("chine_frac", design.chine_frac, b.chine_frac_sd), 0.05, 1.0))

        # Rocker cannot eat more depth than there is.
        cap = 0.85 * d_kw["side_height"]
        d_kw["bow_rocker"] = min(d_kw["bow_rocker"], cap)
        d_kw["stern_rocker"] = min(d_kw["stern_rocker"], cap)

        list_in = abs(rng.normal(0.0, b.asymmetry_sd))
        d_kw["build_list_in"] = design.build_list_in + list_in
        rec["u_build_list_in"] = list_in

        extra = max(0.0, rng.normal(b.extra_weight_mean, b.extra_weight_sd))
        rec["u_extra_weight_lb"] = extra
        d_kw["extra_weight_lb"] = design.extra_weight_lb + extra

        if design.n_frames > 0 and rng.random() < b.drop_a_frame_prob:
            d_kw["n_frames"] = design.n_frames - 1
            rec["u_dropped_frame"] = 1
        else:
            rec["u_dropped_frame"] = 0

        if design.chine_tape_layers > 1 and rng.random() < b.skimp_chine_tape_prob:
            d_kw["chine_tape_layers"] = design.chine_tape_layers - 1
            rec["u_skimped_chine_tape"] = 1
        else:
            rec["u_skimped_chine_tape"] = 0

        dens = float(np.clip(design.tape_diag_density
                             + rng.normal(0.0, b.tape_density_sd), 0.0, 1.0))
        rec["u_tape_density"] = dens - design.tape_diag_density
        d_kw["tape_diag_density"] = dens

        wf = float(np.clip(design.wet_factor + rng.normal(0.0, b.wet_factor_sd),
                           0.25, 1.0))
        rec["u_wet_factor"] = wf
        d_kw["wet_factor"] = wf

    # ---------------- race day ----------------
    r = cfg.race
    if r.enabled:
        powers = []
        for w in design.crew_power_w:
            f = _lognorm(rng, r.power_cv)
            if rng.random() < r.bad_day_prob:
                f *= r.bad_day_factor
            powers.append(w * f)
        d_kw["crew_power_w"] = tuple(powers)
        rec["u_power_total_w"] = float(sum(powers))

        kgs = tuple(max(4.0, k + rng.normal(0.0, r.crew_kg_sd))
                    for k in design.crew_kg)
        d_kw["crew_kg"] = kgs
        rec["u_crew_kg_mean"] = float(np.mean(kgs))

        # Pacing discipline: the plan is not always what happens.
        harder = {"negative_split": "even", "even": "fast_start",
                  "fast_start": "hard_start", "hard_start": "hard_start"}
        if rng.random() < r.pacing_slip_prob:
            d_kw["pacing"] = harder.get(design.pacing, design.pacing)
            rec["u_pacing_slip"] = 1
        else:
            rec["u_pacing_slip"] = 0

        af = float(np.clip(design.anaerobic_fraction
                           * _lognorm(rng, r.anaerobic_fraction_cv), 0.05, 0.75))
        d_kw["anaerobic_fraction"] = af
        rec["u_anaerobic_fraction"] = af

        lean = max(0.0, worst_of_n_normal(rng, design.paddle_lean_in,
                                          r.stroke_lean_sd, r.strokes_per_race))
        d_kw["paddle_lean_in"] = lean
        rec["u_worst_stroke_lean_in"] = lean

        chop = float(rng.choice(np.asarray(r.chop_levels, dtype=float),
                                p=np.asarray(r.chop_weights, dtype=float)
                                / np.sum(r.chop_weights)))
        d_kw["chop_height"] = chop
        rec["u_chop_in"] = chop

        splash = float(rng.exponential(max(r.splash_aboard_mean_gal, 1e-9)))
        d_kw["water_aboard_gal"] = design.water_aboard_gal + splash
        rec["u_splash_gal"] = splash

        extra_soak = max(0.0, rng.normal(r.soak_extra_min_mean,
                                         r.soak_extra_min_sd))
        d_kw["pre_race_soak_min"] = design.pre_race_soak_min + extra_soak
        rec["u_extra_soak_min"] = extra_soak

        wind = float(rng.normal(r.wind_mean_mph, r.wind_sd_mph))
        d_kw["headwind_mph"] = wind
        rec["u_headwind_mph"] = wind

        off = float(rng.uniform(r.boarding_offset_lo, r.boarding_offset_hi))
        d_kw["boarding_offset_frac"] = off
        rec["u_boarding_offset"] = off

        p_kw["nu_ft2_s"] = params.nu_ft2_s * _lognorm(rng, r.water_temp_nu_cv)

    # ---------------- model ----------------
    m = cfg.model
    if m.enabled:
        if not params.is_corrugated:
            # Solid-board family with an UNKNOWN thickness, not a measured
            # one: draw the absolute thickness fresh on every trial from the
            # team's own description of it, rather than jittering a few
            # percent around one guessed point. A hull that only survives at
            # one end of this range is fragile, and the robustness score
            # needs to see that, not paper over it.
            t_mm = float(rng.uniform(*m.paperboard_thickness_range_mm))
            dens = float(rng.uniform(*m.paperboard_density_range_gcm3))
            stiff_mult = float(rng.uniform(*m.paperboard_stiffness_mult_range))
            strength_mult = float(rng.uniform(*m.paperboard_strength_mult_range))
            solid = _solid_board(t_mm / 25.4, density_g_cm3=dens,
                                 e_md_psi=780_000.0 * stiff_mult,
                                 e_cd_psi=390_000.0 * stiff_mult,
                                 strength_psi=4200.0 * strength_mult)
            for k, v in solid.items():
                if k not in ("material_name", "is_corrugated", "warp_tolerance",
                            "tape_shear_floor", "sigma_across_ratio"):
                    p_kw[k] = v
            rec["u_paperboard_thickness_mm"] = t_mm
            rec["u_paperboard_density"] = dens
            rec["u_paperboard_stiffness_mult"] = stiff_mult
            rec["u_board_quality"] = stiff_mult          # for the tornado chart
            p_kw["g_board_psi"] = params.g_board_psi * stiff_mult * _lognorm(rng, m.g_board_cv)
            p_kw["shell_buckle_knockdown"] = params.shell_buckle_knockdown * _lognorm(rng, m.shell_knockdown_cv)
            p_kw["tape_tensile_lb_in"] = params.tape_tensile_lb_in * _lognorm(rng, m.tape_tensile_cv)
        else:
            # One shared "is this good cardboard" factor, so stiffness, strength
            # and thickness move together the way properties of a single sheet do.
            q = _lognorm(rng, m.board_common_cv)
            rec["u_board_quality"] = q

            p_kw["board_caliper_in"] = params.board_caliper_in * q ** 0.4 * _lognorm(rng, m.board_caliper_cv)
            p_kw["board_areal_lb_ft2"] = params.board_areal_lb_ft2 * q ** 0.5 * _lognorm(rng, m.board_areal_cv)
            p_kw["board_ect_lb_in"] = params.board_ect_lb_in * q * _lognorm(rng, m.board_ect_cv)
            ds = q * _lognorm(rng, m.board_stiffness_cv)
            p_kw["board_d_stiff_lbin"] = params.board_d_stiff_lbin * ds
            p_kw["board_d_soft_lbin"] = params.board_d_soft_lbin * ds * _lognorm(rng, 0.08)
            p_kw["g_board_psi"] = params.g_board_psi * q * _lognorm(rng, m.g_board_cv)
            p_kw["flat_crush_psi"] = params.flat_crush_psi * q * _lognorm(rng, m.flat_crush_cv)
            p_kw["shell_buckle_knockdown"] = params.shell_buckle_knockdown * _lognorm(rng, m.shell_knockdown_cv)
            p_kw["tape_tensile_lb_in"] = params.tape_tensile_lb_in * _lognorm(rng, m.tape_tensile_cv)

        res = float(math.exp(rng.normal(0.0, m.residuary_lognormal_sigma)))
        p_kw["residuary_scale"] = params.residuary_scale * res
        rec["u_residuary_scale"] = p_kw["residuary_scale"]

        p_kw["form_factor_scale"] = params.form_factor_scale * _lognorm(rng, m.form_factor_cv)
        p_kw["blade_efficiency"] = min(0.95, params.blade_efficiency
                                       * _lognorm(rng, m.blade_efficiency_cv))
        p_kw["max_thrust_per_paddler_lb"] = (params.max_thrust_per_paddler_lb
                                             * _lognorm(rng, m.max_thrust_cv))
        rec["u_g_board_psi"] = p_kw["g_board_psi"]
        rec["u_board_ect"] = p_kw["board_ect_lb_in"]
        rec["u_board_stiffness"] = p_kw["board_d_stiff_lbin"]
        rec["u_tape_tensile"] = p_kw["tape_tensile_lb_in"]
        rec["u_board_areal"] = p_kw["board_areal_lb_ft2"]
        rec["u_blade_efficiency"] = p_kw["blade_efficiency"]

    d = design.with_(**d_kw) if d_kw else design
    p = params.with_(**p_kw) if p_kw else params
    return d, p, rec
