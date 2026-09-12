# hullsim — cardboard canoe design simulator

A parameter sweep over cardboard canoe hulls. You describe a space of possible
boats; it builds thousands of them, floats each one, heels it, paddles it
across 140 m, checks whether the cardboard can actually hold that shape, and
hands back a dashboard of the trade-offs.

It does **not** hand back one winner. What you gain in one place you lose in
another, and the point of the thing is to make those trades visible.

---

## Getting it running

Two commands. The first sets up an isolated Python environment so nothing
touches your system Python; the second installs the three libraries used.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Then check it works:

```bash
.venv/bin/python tests/test_physics.py
```

You should see `78/78 passed`. Those tests check the hydrostatics against
closed-form answers for a rectangular box, so if they pass, the numerical
machinery is sound.

Now run something:

```bash
.venv/bin/python scripts/run_sweep.py -n 5000
```

About half a minute on a modern laptop. It writes two files:

| File | What it is |
|---|---|
| `out/sweep.csv` | Every design, every computed number. Open in Excel. |
| `out/sweep.html` | The readable version: charts, rankings, what blocked what. |

Open `out/sweep.html` in a browser. Everything is embedded in the one file,
so you can email it to a teammate and it still works.

Then look at the winner in detail:

```bash
.venv/bin/python scripts/report_design.py --from-sweep out/sweep.csv --rank 1
```

That writes `out/design_<name>.html`: hull drawings, righting-arm curve, drag
breakdown, flooding cascade, and the full cut list with tape lengths.

---

## The commands

```bash
# Explore. Latin-hypercube sample over ~20 parameters at once.
.venv/bin/python scripts/run_sweep.py -n 20000

# Isolate. Grid two or three axes with everything else pinned.
# --base matters here: a grid pins everything it does not vary, so centre it
# on a hull that already passes rather than on one that does not.
.venv/bin/python scripts/run_sweep.py --base recommended --grid beam,length

# Or centre it on whatever your last sweep found.
.venv/bin/python scripts/run_sweep.py --base-from out/sweep.csv --grid beam,n_frames

# Polish. Hill-climb from the best design found.
.venv/bin/python scripts/run_sweep.py -n 8000 --refine

# Look at one hull, or compare several side by side.
.venv/bin/python scripts/report_design.py --named recommended
.venv/bin/python scripts/report_design.py --set length=96 beam=36 n_frames=6
.venv/bin/python scripts/report_design.py --from-sweep out/sweep.csv --rank 1,2,3

# Rebuild the dashboard from an existing CSV without re-evaluating.
.venv/bin/python scripts/run_sweep.py --from-csv out/sweep.csv
```

Useful flags on `run_sweep.py`: `--chop 2` for rougher water, `--crew 130,175`
for different paddlers, `--power 200,180` for different fitness, `--depth 3`
for a shallow course, `--residuary-scale` for a calibrated drag model.

The usual working order is: explore wide, read the charts, grid the two or
three axes that turned out to matter, then refine.

### Three named hulls to compare against

| Name | Dimensions | What it is |
|---|---|---|
| `baseline` | 102 × 29 × 35 × 13 | Where the team's earlier hand analysis landed. Kept unchanged, and the tool reports honestly that it is over the tape roll at full diagonal coverage. |
| `recommended` | 104 × 33 × 36 × 14 | What a 20,000-design sweep landed on, rounded to numbers you can cut to. |
| `fast` | 112 × 31 × 34 × 14 | The same idea stretched. About six seconds quicker with less of every margin — an illustration of the trade, not a recommendation. |

Two things about `recommended` are worth noticing, because neither is what
canoe intuition suggests. **The section is nearly a box** — only three inches
of total flare. Flare makes a canoe seaworthy in real water and gives a
paddler somewhere to reach, but on flat water it costs waterplane inertia for
a given beam *and* it is the single thing that makes a side panel
non-developable. On a pond, in cardboard, a near-box section wins twice.
**The rocker is small** — an inch and a half at the bow. Tipping the bow up
buys a little in chop and costs buoyancy exactly where the bow needs it, so
the sweep keeps only a token amount.

---

## The board: measure it before you trust anything

**This is the largest open question in the project and it takes two minutes
to close.**

The model was built assuming single-wall corrugated cardboard. Three things
say the actual stock is not that: it is described as thin and professional
rather than box-like, it is "coated one side" — which is paperboard
terminology, not corrugated — and it comes on a **roll**. Corrugated cannot
be rolled without crushing its flutes.

If it is thin solid paperboard instead, several conclusions invert:

| | Corrugated | Thin solid board |
|---|---|---|
| Bending stiffness | very high | roughly a tenth |
| Edgewise strength | weak | several times stronger |
| Direction matters | strongly | mildly, and unchooseable |
| Forced curvature | crushes the flutes | bends happily |
| **What governs** | **panel strength** | **torsion** |

Run this with two measurements and it tells you which:

```bash
.venv/bin/python scripts/identify_material.py --area 12x12 --weight-g 46 --caliper-mm 1.1
```

Cut a square you can measure, weigh it in grams, and measure the thickness
(stack ten sheets and divide). Density separates the families cleanly: solid
paperboard runs 0.6 to 0.9 g/cm³, corrugated under 0.25, because corrugated
is mostly air. Then pass `--material` to any script.

### Why it matters this much

The same recommended hull, changing nothing but the board:

| Material | Hull weight | Twist | Bad-day score | Builds that pass |
|---|---|---|---|---|
| corrugated C-flute | 24 lb | 2.5° | 75.2 | **100%** |
| solid 0.6 mm | 17 lb | **10.0°** | 13.6 | **41%** |
| solid 1.0 mm | 27 lb | 6.2° | 31.9 | 87% |
| solid 1.5 mm | 39 lb | 4.3° | 60.2 | 99% |
| solid 2.0 mm | 52 lb | 3.4° | 70.8 | 100% |

And the sting: if the board is thin and solid, **every way of stiffening her
runs out of cardboard**. Bigger gunwale tubes, more frames, a second skin
layer — each fixes the twist and each busts the one roll you have. The hull
would have to get smaller to free up the material. That is a different
optimum from the one currently recommended, and it cannot be found until the
board is known.

---

## Running it like a real simulator: Monte Carlo

Everything above gives one answer per design. That answer is a fiction in one
specific way: it assumes you build the boat at exactly the drawn dimensions,
that you paddle exactly as hard as planned, and that your cardboard has
exactly the textbook stiffness. None of those will be true.

So run the same design a few thousand times, each time drawing one plausible
built boat, one plausible race day, and one plausible set of true material
properties:

```bash
.venv/bin/python scripts/run_montecarlo.py --named recommended -n 5000
```

**Nothing here randomises a result.** The physics is untouched; only the
inputs move. That is what makes the spread mean something — five seconds of
scatter is not the model being vague, it is the honest consequence of not
knowing your board's stiffness to better than thirty percent.

```bash
# Where does the spread come from: building, racing, or not knowing?
.venv/bin/python scripts/run_montecarlo.py --named recommended --decompose

# Rank a sweep's leaders by how well they hold up, not how well they draw
.venv/bin/python scripts/run_montecarlo.py --from-sweep out/sweep.csv --top 30 -n 1500

# Leave it running overnight. Sweeps 60,000 designs, then Monte Carlos the
# leaders until the budget runs out, checkpointing after every one.
.venv/bin/python scripts/run_montecarlo.py --overnight --hours 8
```

An 8-hour run does roughly 7 million evaluations: about 1,700 designs at
4,000 trials each. It measures its own speed first and sizes the work to fit,
writes a checkpoint after every design, and resumes if you point it at the
same output directory. Ctrl-C once and it finishes the design it is on and
writes the report.

### What the extra runtime actually buys

**Robustness, which is invisible to a single run.** On the 24 leading designs
from a 12,000-design sweep, the nominal ranking and the robust ranking
disagree badly:

| | Nominal rank | Robust rank | Nominal score | Builds that pass |
|---|---|---|---|---|
| r001986 | 1 | 1 | 87.8 | **100%** |
| r018264 | 2 | 7 | 86.6 | **20%** |
| r017334 | 3 | 13 | 85.7 | **25%** |
| r014817 | 9 | 23 | 84.1 | **1%** |

Four of the nominal top five fall apart once you allow for half an inch of
cut error. They are not good designs; they are lucky points in parameter
space, and only resampling the build finds that out.

**A shopping list instead of a shrug.** Every sampled input is recorded
alongside its trial's outcome, so a rank correlation says which unknown is
actually driving the scatter. Two dominate everything else: the wave-making
term (ρ = 0.91 against crossing time) and the board shear modulus. Timing one
paddle over a known distance and twisting one taped test tube would collapse
most of the uncertainty in this model. Everything else is noise you can live
with.

**Failures come from coincidences.** Switch on each source alone and the
recommended hull passes 100%, 98% and 100% of trials. Switch on all three and
it passes 99% — but a marginal hull drops far below the product of its
individual rates. What sinks a trial is a slightly crooked hull *and* a bad
stroke *and* softer board than assumed, each of which would have been
survivable alone. That is why margins cannot be reasoned about one at a time.

**Crew power was nearly irrelevant, and that was a bug.** The low-speed
thrust cap — the limit that stops power over speed going to infinity at a
standing start — was set tight enough that it kept binding at race speed. The
boat was thrust-limited the whole way, so how hard the crew paddled barely
moved the predicted time. The times still looked plausible, which is what
made it hard to spot; it surfaced only when the question "are we simulating
how hard people paddle?" was asked directly. Fixed, with a test that fails if
the cap ever creeps back up. Doubling crew power now buys about nine seconds,
and crew power is the second-largest driver of the spread in crossing time.

**The cheapest fix on the whole boat turned out to be posture.** Kneeling low
(15.5 in crew CG) and kneeling lower (13 in) score identically on a single
deterministic run — 85.7 against 85.8. Across thousands of builds and days
they do not: the pass rate goes from 88% to 99%. Two and a half inches of
crew height costs no cardboard, no tape and no speed, and buys more margin
than two inches of extra beam, which costs all three. That result does not
exist in a deterministic model.

### Pacing, and why going out hard costs you

Power varies both *between* races (±15% plus a one-in-ten bad day) and
*within* them. The within-race part matters more than it sounds:

| Strategy | Time | Power start / finish | Fade |
|---|---|---|---|
| Even the whole way | **72.6 s** | 320 / 320 W | 0% |
| Brief lift off the line | 72.6 s | 363 / 306 W | 16% |
| Hold back, then build | 72.7 s | 282 / 358 W | −27% |
| Sprint it and hang on | 73.6 s | 458 / 256 W | 44% |

Even pacing wins, and sprinting the start costs a full second. This boat
punishes a hard start more than most would: residuary resistance climbs very
steeply past Froude 0.4, which is exactly where this hull settles, so
overspeeding early does not cost a little extra drag — it costs a lot, and
that energy comes out of a reserve that does not refill inside seventy-five
seconds. The Monte Carlo also models *not executing the plan*: a one-in-four
chance of going out harder than intended, because adrenaline off the line is
real.

**Start controlled and hold it.** A brief lift off the line is free; a sprint
is not.

### Posture, depth and the clock: three free decisions

None of these cost cardboard, tape or speed, and all three came out of
modelling the crew properly.

**Kneel upright, not tall and not crouched.** Kneeling tall scores best on a
single deterministic run and passes about a fifth of simulated builds.
Crouching right down is stable and slow. Upright gives up a few tenths of a
point nominally and passes nearly every build:

| Posture | Nominal | Bad day | Builds that pass | Median time |
|---|---|---|---|---|
| kneeling tall | **86.1** | 76.3 | **21%** | 75.7 s |
| kneeling upright | 85.7 | **78.8** | **99%** | 77.5 s |
| kneeling low | 85.3 | 78.5 | 100% | 78.9 s |
| back on heels | 84.8 | 77.8 | 100% | 81.4 s |

**Side height has an interior optimum**, which it did not before. Freeboard
wants deeper sides; a five-foot-three paddler reaching over the rail wants
shallower ones. Fourteen inches is the balance for this crew. A taller crew
would want a deeper boat.

**Launch one minute before the gun, not ten.** Strength decays with a
six-minute time constant and the race lasts seventy-five seconds, so nearly
all the soaking happens while you wait:

| Minutes afloat before the start | Strength at the finish | Twist |
|---|---|---|
| 0 | 90% | 1.8° |
| 5 | 64% | 2.6° |
| 20 | 47% | 3.5° |

### Two things that only show up once you look for them

**She is a different boat standing still.** At race speed the chop is short
and arrives fast, and the hull physically cannot respond — wave-induced roll
comes out under a tenth of a degree. Stopped at the start line, the
encounters slow to the wave period, and for a hull this size in three inches
of chop that lands almost exactly on roll resonance: twelve degrees of roll,
which eats three and a half inches of freeboard. That is also precisely when
somebody is climbing in over the gunwale, so the two stack. The boarding
check now includes it.

**A stiffer hull is not automatically calmer.** Roll period goes as one over
the square root of GM, so piling on stability shortens the period and can
tune it *into* the chop. Stopped in three inches of chop:

| GM | Roll period | Roll amplitude |
|---|---|---|
| 5 in | 1.79 s | 1.5° |
| 10 in | 1.27 s | 4.2° |
| 22 in | 0.85 s | **12.3° — resonant** |
| 35 in | 0.68 s | 8.5° |
| 50 in | 0.57 s | 6.8° |

The worst case is in the middle, not at either end. Nothing else in the model
pushes back on ever-increasing GM; this does.

### The three sources, and why they are kept separate

| Source | What it is | How you reduce it |
|---|---|---|
| `build` | The boat you drew versus the boat you cut. Tape measures, knives, a panel half an inch narrow, one side heavier than the other. | Jigs and care |
| `race` | How hard you actually paddle, how you actually kneel, the chop, splash aboard while boarding. | Practice |
| `model` | What nobody knows: your board's real stiffness, the real wave-making of a shoebox. This is ignorance, not variability. | **Measurements** |

`--sources build` runs one alone. `--decompose` runs all three separately and
plots them together. They are separate because the remedies are different, and
reporting one combined spread would hide which of the three to go and fix.

### Reading the probabilities honestly

`P(feasible) = 0.93` does **not** mean a 93% chance of winning. It means that,
given the spreads written down in `hullsim/uncertainty.py`, 93% of the boats
you might build on the days you might race pass every hard constraint. Those
spreads are considered judgements — my estimates of how badly a hand-cut
cardboard panel misses its dimension — not measured distributions. Compare
designs with these numbers. Do not quote them as predictions.

---

## Running it across several machines at once

The overnight Monte Carlo run is embarrassingly parallel across machines —
four computers each exploring a different random slice of the design space
for eight-plus hours finds more than one machine can, and disagreement
between them is itself useful information (see below).

**Getting the code onto each machine**: this repo is on GitHub
(`github.com/leviholliday/cuBoatRaceSimulation2026`, private). On any other
machine —

```bash
git clone https://github.com/leviholliday/cuBoatRaceSimulation2026.git
cd cuBoatRaceSimulation2026
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Give each machine its own `--seed` so they explore *different* designs
rather than duplicating each other's work:

```bash
.venv/bin/python scripts/run_montecarlo.py --overnight --hours 8 --seed 2 \
  --material paperboard_unknown
```

**Getting results back**: a small results site collects them —
`cuboatrace2026-results.netlify.app`, backed by Netlify Functions and Blob
storage (see `netlify/functions/`). Once a machine's run finishes:

```bash
UPLOAD_TOKEN=<ask Levi for this> python scripts/upload_results.py --tag pi
```

`--tag` is just a label so results from different machines don't collide —
use something like `pi`, `laptop2`, `friend`. It zips `out/mc` and posts it;
the results page lists every upload with a download link, no login needed
(the upload token gates *writing*, not reading — the URL is not published
anywhere and the data is not sensitive).

Download each machine's zip, extract them into separately named folders,
then combine them:

```bash
python scripts/merge_runs.py \
  --run laptop1=path/to/laptop1_extracted \
  --run pi=path/to/pi_extracted \
  --run friend=path/to/friend_extracted \
  --out out/merged
```

This is the step that actually matters, not just a formality: every
machine's designs are named `r000000, r000001, ...` starting from zero, so
without tagging, two machines' *completely different* hulls would collide
under the same name. `merge_runs.py` handles that, and — more useful than
the winner it picks — tells you whether the machines *agree*. If four
independent random searches land on similar hulls, that's real confidence.
If they disagree, the ranking is still sensitive to which designs happened
to get sampled, and the honest move is a large confirmatory run on the
consensus pick before trusting it (`merge_runs.py` prints that exact
command).

**What this setup does not do**: the results site is deployed directly, not
wired to auto-redeploy from GitHub pushes — that link is one manual step in
the Netlify dashboard (Site settings → Build & deploy → Link repository) if
future code changes to the site should redeploy automatically. Not needed
for today's functionality; the site already works.

---

## How to read the dashboard

**The trade-off chart** is the one that matters. Every design is a dot. Grey
dots fail a hard constraint and are drawn anyway, because where the feasible
region *ends* is most of what a sweep tells you. The red Pareto front on the
right marks designs where you cannot get faster without getting less reliable
— everything below it is dominated and not worth arguing about.

**The score is an expected value**, not a performance number:

```
expected score  =  P(finish) × (finish points + speed points)  +  build points
```

A hull two seconds faster with a one-in-four chance of swimming scores below a
slower one that always finishes. That was the brief — going for the win, and
sinking is not winning — so it is built into the ranking rather than left for
someone to remember.

**The rubric is a guess.** The weights in `hullsim/scoring.py` are a
reasonable default for a cardboard boat race, not your course's actual scoring
sheet, which was not available when this was written. Edit `Rubric` once you
have the real one. Everything reads from that one place.

---

## What the model actually does

Each design goes through this, in order, with no circular dependencies:

```
design → mesh → material take-off → hull weight → mass properties
       → equilibrium (draft AND trim) → righting-arm curve
       → resistance → race time → seakeeping → swamping reserve
       → structure → score
```

**Geometry.** The hull is defined by about twenty parameters — length, bottom
width, beam, depth, bow and stern taper with an entry shape exponent, rocker,
sheer, and where the chine sits. It is discretised into 41 transverse sections,
each a polygon. Everything downstream is numerical integration over those
polygons, so there is exactly one definition of what the boat is.

**Hydrostatics.** One primitive does all of it: clip every section polygon
against the waterplane and integrate. Upright, trimmed and heeled all go
through the same code, so they cannot disagree. Both draft *and* trim are
solved, so the boat floats where the centre of buoyancy actually sits under
the centre of gravity.

**Stability** is a full large-angle righting-arm curve, not just GM. GM is
only the slope at zero heel; it says nothing about where the gunwale goes
under, and for an open boat that *is* the capsize. The curve also finds the
angle of vanishing stability and the energy reserve — the area underneath,
which is what decides whether she survives a shove rather than how stiffly she
resists the first degree of one.

**Resistance** is ITTC-1957 friction on the wetted surface the geometry
actually produced, a published form-factor correlation, and a standard-series
residuary curve. The race is integrated from a standing start with the boat's
added mass, so a heavier hull genuinely loses time getting off the line.

**The crew** are modelled from their actual height, weight, shin length and
posture rather than from one typed-in centre-of-gravity number. Kneeling
lower drops the centre of gravity, which helps stability, and drops the
shoulders toward the gunwale, which costs stroke. Both come from the same
posture, so they cannot disagree, and the optimum is in the middle instead
of at the bottom. Seating positions are swept too, bounded by knees
overlapping and paddles clashing.

**Soaking is progressive.** Board strength decays toward a wet floor with a
time constant of about six minutes. The structural checks use the strength
left at the finish, not a fixed knockdown, which is what makes launching
early expensive.

**Directional stability.** Rocker lifts the lateral plane out of the water,
the boat stops holding a line, and every correction stroke is power not going
forwards. Mismatched paddlers on opposite sides cost the same way. Nothing in
a drag calculation knows about either, so both are modelled separately as a
loss of propulsive efficiency.

**Dynamic roll.** Heel is treated quasi-statically everywhere, and that is
checked rather than assumed: at race speed the chop arrives far faster than
the hull's roll period, so the amplification is near zero and quasi-static is
right. Stopped at the line it is a different boat — see below.

**Paddling** is not a constant. The crew has a sustainable power and a finite
anaerobic reserve, the standard critical-power and W' description. Spending
above the sustainable level drains the reserve, and when it is gone they are
pinned at that level whether they like it or not. That is what makes a hard
start cost something instead of being free. Four pacing strategies are
modelled and the single-design report compares them.

**Seakeeping** uses Ochi's deck-wetness criterion for water over the gunwale,
and couples the crew's own rolling to the real righting-arm curve — so a lean
that a stiff hull shrugs off will roll a tender one onto its ear.

**Structure** covers the four ways these boats actually fail: torsion, hull
girder bending, bottom-panel deflection, and joints.

---

## Things this model found that are worth knowing

These came out of the physics rather than being assumed, and several are
things you would not get from a hydrostatics calculator.

**Long and skinny is the wrong answer.** The rumoured 115 × 20 × 24 hull
computes to *negative* GM — it is statically unstable sitting flat, before
anyone moves. It is in the test suite as a regression check.

**GM is not enough, and it will quietly pass hulls that capsize.** A hull can
have a healthy GM and a peak righting arm *smaller than what one paddler
leaning two inches off the centreline demands*. Narrow-bottomed flared hulls
do this routinely: steep at the origin, flattening early. They feel stiff for
the first few degrees and then simply keep going. The sweep now rejects them
on a peak-righting-arm-to-demand ratio, which GM cannot see.

**The crew is a bigger wave than the water.** An inch of pond ripple moves the
surface about a quarter inch. A paddler shifting two inches sideways drops the
low gunwale by an inch or more, on every stroke. Freeboard enters the wetness
criterion squared and inside an exponential, so it is a cliff edge, not a
linear margin.

**Rocker, plan taper and flare are each free; combining all three is not.**
Cardboard is a flat sheet — it bends in one direction and does not stretch. A
bottom with rocker is a cylinder and rolls fine. Vertical topsides with a
curved plan are also developable. But the flared panel between bottom edge and
chine moves outboard *and* upward at rates that change along the hull, and
that surface cannot be folded from flat board. The model measures the warp and
rejects hulls that need darts. Nothing in a hydrostatics tool would notice.

**The bottom panel stops being a panel.** Under water pressure plus a kneeling
paddler it deflects well past its own thickness, at which point it is carrying
load as a membrane in tension rather than as a plate in bending — and that
tension is reacted by the chine tape. On a bare single-layer bottom with few
frames, that tape is the first thing to run out. It is the load path nobody
draws.

**Frames stop helping once their spacing drops below the boat's width.** The
load splits between the two spans as the fourth power of the other one, so
past that point you are stiffening the direction that was already stiff.

**Decking the ends is the single highest-leverage structural decision.** An
open trough has essentially no St-Venant torsional stiffness; what saves it is
the gunwale tubes acting as I-beam flanges. Decking closes the section into a
torsion box and the model shows roughly a sevenfold reduction in twist.

**She may not be boardable.** The design condition never sees the moment the
second paddler steps in, with their whole weight a quarter of the half-beam
off the centreline. Several hulls that float and paddle perfectly well put
the rail under while being climbed into — and a boat that starts the race
with water in it has already lost, because the free-surface effect below is
waiting for it.

**Two longitudinal dividers beat any amount of extra beam** for surviving
water aboard. Free-surface loss goes as the cube of the free water's width, so
*n* lanes cut it by *n* squared, for a few square feet of cardboard.

**Full ±45° tape coverage of the bottom does not fit on one roll.** It is over
half the roll on its own. `tape_diag_density` is therefore a real design
variable, and cutting it buys the saving back in twist.

---

## How much to trust each number

Every constant in `hullsim/constants.py` is labelled one of three ways.

| Label | Meaning |
|---|---|
| `[EXACT]` | A definition or physical constant. |
| `[TYPICAL]` | Published for this class of material. Defensible in a report, but not *your* cardboard. |
| `[MEASURE]` | Currently a guess. Replace before trusting anything downstream. |

In descending order of trustworthiness:

1. **Hydrostatics** — displacement, draft, trim, GM, the righting-arm curve,
   waterplane properties. Numerical integration of geometry you specified. The
   tests check these against closed-form answers for a box. Solid.
2. **Frictional resistance** — ITTC-1957 on a computed wetted surface. This is
   what every towing tank reduces data with. Solid.
3. **Material take-off** — real geometry, developed lengths, seam laps.
   Arithmetic. Its accuracy is limited only by the board density constant.
4. **Form factor** — a published correlation applied to a hull bluffer than
   anything it was fitted to. About right.
5. **Structure** — correct mechanics (plate theory with membrane action,
   Bredt torsion, warping restraint) on material properties that are typical
   values rather than measurements. Use for comparing layouts, not for
   certifying a margin.
6. **Residuary resistance** — a standard-series correlation, extrapolated. At
   the speed this boat reaches it is *most* of the drag, so it dominates the
   predicted time. **This is the weakest link and it matters most.**
7. **Failure probabilities** — smooth functions of safety margin, not
   validated reliability models. The shape is right; the numbers are not
   calibrated.

### The four measurements that would improve this most

1. **Weigh a measured square of your actual cardboard**, and caliper it. Two
   minutes. Hull weight feeds straight back into displacement, so this
   tightens more downstream numbers than anything else. Put it in
   `data/joint_tests.json`.
2. **Time a measured paddle** and run `scripts/calibrate.py`. This replaces
   the residuary correlation with something anchored to your crew and your
   hull. It is the difference between the model quoting a textbook at you and
   the model extrapolating from your own measurement.
3. **Load your joints to failure** — static and then cyclic over ~150 strokes,
   which is roughly a race. `data/joint_tests.json` has the protocol. Right
   now that file has the *ordering* you observed with placeholder magnitudes.
4. **Run the inclining test** on the finished boat. The single-design report
   prints exactly what the plumb line should read. A bigger swing than
   predicted means the real GM is lower, almost always because the crew's
   centre of gravity is higher than assumed — the cheapest thing left to fix.

---

## What is still missing

An honest list, because the model looking thorough is not the same as it
being complete.

**Nothing here has ever been checked against a real boat.** This is the
single biggest gap and no amount of extra physics closes it. Every test in
the suite is either internal consistency or a closed-form answer for a
rectangular box. That makes the machinery trustworthy; it says nothing about
whether the model describes a cardboard canoe. One timed paddle and one
inclining test would change that more than anything else in this document.

Also absent, roughly in order of how much they could matter:

- **Which board it actually is.** See above. Bigger than everything else on
  this list combined.
- **Sealed end chambers.** Raised in your own lab discussion. Decked ends
  currently close the torsion box but get no credit as reserve buoyancy. If
  you build them airtight, that changes swamping from losing the boat to
  finishing awash, and the swamping cascade would need to know.
- **Racing another boat.** Two at a time, so there is a wake, and there is
  the pull of trying to match someone. Neither is modelled.
- **Joint fatigue over the race.** About 150 strokes. The joint data has no
  cyclic numbers in it at all, only single-pull rankings.
- **The transom as a panel.** With a full-width transom it is a large
  unsupported sheet under hydrostatic pressure, and it is not checked the way
  the bottom is.
- **Launch, grounding and recovery loads.** Dropping her in, touching bottom
  at a shallow start, lifting her out at the end.
- **Roll damping** is one assumed coefficient and it scales the resonance
  result directly.

Two things were quantified and then deliberately left out, which is different
from forgetting them: **wind heeling** produces under two tenths of an inch of
heeling arm at fifteen miles an hour, against a crew lean that already demands
more than an inch; and **wave-induced roll at race speed** is under a tenth of
a degree. Both are in the code as checks rather than as effects.

**The numerics are converged**, which is worth stating because everything
else rests on it. Forty-one stations match four hundred to within 0.03% on
draft, GM and righting arm; the race timestep and the drag table are both
converged to under 0.05 s. There are tests for all three.

---

## Deliberately not modelled

- **Getting in and out of the boat, and oar handling.** These are coordination
  and technique between two specific people, not hydrodynamics. Modelling them
  would need motion data on you two specifically, and by the time you had that
  you would have just practised it — which is strictly more useful. What the
  design *can* do for these is keep the gunwale low and consistent and the
  hull untippy, which the stability side already optimises for.
- **Steering and passing.** A straight two-at-a-time time trial, so the speed
  model stays a clean straight-line calculation.
- **Transport to the lake.**
- **Full unsteady wave mechanics.** For an inch or two of pond chop, a
  freeboard-margin criterion is the right level of rigour and real wave
  diffraction would be false precision.

---

## Layout

```
hullsim/
  constants.py    physical constants and material properties, each labelled
  design.py       HullDesign: every knob, with what it means
  geometry.py     parametric hull -> station polygons; developability
  hydro.py        polygon clipping, displacement, trim, righting-arm curve
  resistance.py   ITTC friction, form factor, residuary, the race integration
  structure.py    torsion, hull girder, panels, joints
  seakeeping.py   chop, crew-induced roll, the flooding cascade
  materials.py    cut list, tape plan, hull weight
  scoring.py      hard constraints and the expected-score rubric   <- edit this
  params.py       the physical numbers we are not certain about
  uncertainty.py  how much each of those could be wrong          <- edit this
  montecarlo.py   run one design many times; sensitivity; robustness
  evaluate.py     one design in, one complete answer out
  sweep.py        grid / Latin hypercube / hill climb, parallelised
  report.py       drawings, charts, the HTML dashboard
scripts/
  run_sweep.py    the main entry point
  run_montecarlo.py  the same design many times, incl. overnight mode
  report_design.py  deep dive on one hull, or compare several
  calibrate.py    turn a stopwatch into a calibrated drag model
  identify_material.py  work out which board you have, from two measurements
data/
  joint_tests.json  YOUR bench measurements                        <- edit this
tests/
  test_physics.py   78 checks, several against closed-form answers
```

Three files are meant to be edited as you learn things:
`data/joint_tests.json` for measurements, `scoring.py` for what "better"
means, and `uncertainty.py` for how wrong each unknown could be. The physics
modules should not need touching.
