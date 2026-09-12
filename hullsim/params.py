"""
Params: the physical numbers the model is not certain about.

Until now these lived as module-level constants, which is fine for a single
deterministic answer and useless for asking "how wrong could this be?". Pulling
them into one value object means a Monte Carlo run can hand every trial a
different, plausible set of material properties and see what the answer does.

WHAT BELONGS IN HERE AND WHAT DOES NOT

In here: anything labelled [TYPICAL] or [MEASURE] in constants.py. Board
stiffness, tape strength, the shear modulus, blade efficiency, the residuary
scale -- things where a published range exists but your particular cardboard
was never measured.

Not in here: the density of water, gravity, the size of the cardboard roll,
the length of the course. Those are known, and pretending to be uncertain
about them would just add noise that hides the uncertainty that is real.

`NOMINAL` is the same set of numbers the deterministic model always used, so
running with default Params reproduces the old answers exactly. There is a
test for that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import constants as C


@dataclass(frozen=True)
class Params:
    """Uncertain physical properties, one coherent set per evaluation."""

    # ---- corrugated board -------------------------------------------------
    board_caliper_in: float = C.BOARD_CALIPER_IN
    board_areal_lb_ft2: float = C.BOARD_AREAL_LB_FT2
    board_ect_lb_in: float = C.BOARD_ECT_LB_IN
    board_d_stiff_lbin: float = C.BOARD_D_MD_LBIN
    """Flexural rigidity with the span running ACROSS the flutes (the stiff
       way). Named by flute direction rather than MD/CD, which is a reliable
       source of confusion -- see the note at the top of structure.py."""
    board_d_soft_lbin: float = C.BOARD_D_CD_LBIN
    sigma_across_ratio: float = 0.35
    """Edgewise compressive strength across the flutes, as a fraction of the
       along-flute (ECT) value. The liners buckle between flute contacts."""
    flat_crush_psi: float = 25.0
    poisson: float = C.BOARD_POISSON

    material_name: str = "corrugated_c_flute"
    is_corrugated: bool = True
    """Whether the board gets its stiffness from flutes.

       This is not a label, it changes the physics. A corrugated board is two
       thin liners held apart by a fluted core: light, very stiff in bending,
       weak edgewise, strongly direction-dependent, and it cannot be rolled
       or forced into double curvature without crushing. A SOLID board of the
       same weight is roughly a tenth as stiff in bending, several times
       stronger edgewise, far less direction-dependent, and happily takes
       curvature. Those point in opposite directions for almost every design
       decision in this project."""

    warp_tolerance: float = 0.055
    """How much flare-panel non-developability the material will take before
       it darts or crushes. Corrugated is intolerant because forcing double
       curvature crushes the flutes. Thin solid board simply bends."""

    # ---- tape -------------------------------------------------------------
    tape_tensile_lb_in: float = C.TAPE_TENSILE_LB_IN
    tape_areal_lb_ft2: float = C.TAPE_AREAL_LB_FT2

    # ---- structural behaviour ---------------------------------------------
    g_board_psi: float = 2000.0
    """Shear modulus of the smeared board. The least certain number in the
       project, and it scales twist angle directly."""
    shell_buckle_knockdown: float = 0.20
    """Real thin shells buckle well below the classical stress. 0.15-0.25 is
       the usual range for imperfection-sensitive shells."""
    tape_shear_floor: float = 0.45
    """Fraction of shear capacity the skin retains with no diagonal tape."""

    # ---- water and propulsion ---------------------------------------------
    nu_ft2_s: float = C.NU_FT2_S
    blade_efficiency: float = C.BLADE_EFFICIENCY
    max_thrust_per_paddler_lb: float = 22.0
    """Stroke-cycle-average thrust one paddler can make at a standstill, lbf.

       This is a LOW-SPEED limit and it must behave like one. Power over
       speed goes to infinity as speed goes to zero, which is not what a
       paddle does, so thrust is capped -- but if the cap is set too low it
       keeps binding at race speed, and then the boat is thrust-limited for
       the whole crossing and the crew's power stops mattering at all. That
       is wrong, and it is a quiet kind of wrong: the times still look
       plausible.

       22 lbf is about 98 N per paddler averaged over the whole stroke cycle,
       recovery included. Sprint canoe blade forces peak several times higher;
       halving for duty cycle and again for two untrained paddlers lands
       here. It stops binding above about 1.1 m/s, so it shapes the first
       second or two off the line and nothing after."""
    form_factor_scale: float = 1.0
    """Multiplier on the Watanabe form-factor estimate."""
    residuary_scale: float = 1.0
    """Multiplier on the standard-series residuary curve. The single most
       uncertain term in the speed prediction, and at race speed the dominant
       one. scripts/calibrate.py turns a stopwatch into a value for this."""

    # ---- derived ----------------------------------------------------------
    @property
    def e_stiff_psi(self) -> float:
        """In-plane modulus of the board treated as a solid plate, back-
        calculated from D = E t^3 / (12 (1 - v^2)). A smeared-plate
        idealisation with no meaning outside that idealisation -- it exists to
        keep the buckling and girder formulas dimensionally honest."""
        return (12.0 * (1 - self.poisson ** 2) * self.board_d_stiff_lbin
                / self.board_caliper_in ** 3)

    @property
    def e_soft_psi(self) -> float:
        return (12.0 * (1 - self.poisson ** 2) * self.board_d_soft_lbin
                / self.board_caliper_in ** 3)

    @property
    def sigma_along_flutes(self) -> float:
        """Edgewise compressive strength along the flutes, as a stress."""
        return self.board_ect_lb_in / self.board_caliper_in

    @property
    def sigma_across_flutes(self) -> float:
        return self.sigma_across_ratio * self.sigma_along_flutes

    @property
    def tape_lb_per_in(self) -> float:
        """Weight of one linear inch of tape. Not negligible: a full 100 m
        roll is about two pounds."""
        return self.tape_areal_lb_ft2 * C.TAPE_WIDTH_IN / 144.0

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    @classmethod
    def of(cls, material: str, **overrides) -> "Params":
        """Build a Params from a named material preset."""
        if material not in MATERIALS:
            raise KeyError(f"unknown material {material!r}; "
                           f"known: {', '.join(MATERIALS)}")
        return cls(**{**MATERIALS[material], **overrides})


def _solid_board(caliper_in: float, density_g_cm3: float = 0.75,
                 e_md_psi: float = 580_000.0, e_cd_psi: float = 290_000.0,
                 strength_psi: float = 3600.0) -> dict:
    """Derive a solid-paperboard preset from thickness and density.

    Solid board is an ordinary elastic sheet, so everything follows from
    D = E t^3 / 12(1-v^2) and a material strength, rather than from the
    sandwich tests (ECT, flat crush) that only mean something for corrugated.

    Defaults are mid-range for coated kraft or folding boxboard: density
    around 0.75 g/cm3, machine-direction modulus about 4 GPa with the cross
    direction near half, and a compressive strength around 25 MPa.
    """
    t = float(caliper_in)
    gsm = (t * 25.4) * density_g_cm3 * 1000.0
    nu2 = 1.0 - C.BOARD_POISSON ** 2
    d_md = e_md_psi * t ** 3 / (12.0 * nu2)
    d_cd = e_cd_psi * t ** 3 / (12.0 * nu2)
    return {
        "board_caliper_in": t,
        "board_areal_lb_ft2": gsm * 0.0002048,
        "board_ect_lb_in": strength_psi * t,   # force per inch of width
        "board_d_stiff_lbin": d_md,
        "board_d_soft_lbin": d_cd,
        "sigma_across_ratio": 0.55,            # fibre anisotropy, not flutes
        "flat_crush_psi": strength_psi * 0.25,
        "is_corrugated": False,
        "warp_tolerance": 0.16,                # thin sheet takes curvature
        "tape_shear_floor": 0.55,
    }


MATERIALS: dict[str, dict] = {
    # What the model assumed until the material was described properly.
    "corrugated_c_flute": {
        "material_name": "corrugated_c_flute",
        "is_corrugated": True,
        "warp_tolerance": 0.055,
    },
    # Coated one side, supplied on a roll. Corrugated cannot be rolled without
    # crushing its flutes, so a roll is strong evidence the stock is solid
    # board. Calibers here span thin carton board to heavy chipboard.
    "paperboard_0.6mm": {"material_name": "paperboard_0.6mm",
                         **_solid_board(0.6 / 25.4)},
    "paperboard_1.0mm": {"material_name": "paperboard_1.0mm",
                         **_solid_board(1.0 / 25.4)},
    "paperboard_1.5mm": {"material_name": "paperboard_1.5mm",
                         **_solid_board(1.5 / 25.4)},
    "paperboard_2.0mm": {"material_name": "paperboard_2.0mm",
                         **_solid_board(2.0 / 25.4)},
    # The team's actual stock, as best it is known without a scale: thinner
    # than a cereal box, "1 to 2 and some mm", and it springs back toward the
    # roll's curvature after being cut -- a sign of real bending stiffness,
    # more than a generic paperboard of the same thickness would show. That
    # is why the modulus here is set above _solid_board's default rather than
    # at it: a board that curls back that hard is fighting you elastically,
    # which means a higher E than a limp folding-carton stock.
    #
    # This ONE POINT is what a deterministic sweep needs to search against.
    # For anything that should hold up across the real uncertainty --
    # Monte Carlo runs, robustness ranking -- uncertainty.ModelUncertainty
    # resamples thickness across the whole 1.0-2.4 mm range on every trial
    # instead of jittering around this point, so the final answer does not
    # depend on this guess being right.
    "paperboard_unknown": {"material_name": "paperboard_unknown",
                           **_solid_board(1.5 / 25.4, density_g_cm3=0.80,
                                          e_md_psi=780_000.0,
                                          e_cd_psi=390_000.0,
                                          strength_psi=4200.0)},
}


NOMINAL = Params()
