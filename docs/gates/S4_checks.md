# The checks before S4, the inversion's parameters

Not a gate: rules checked on each window before its inversion, so that the inversion's bounds
come from the curve it inverts (the user's decision of milestone 13: per window). Code:
sigpipe's `sigpipe.masw.inversion.priors` (`derive_inversion`), called by `judge_inversions`
and `rerun_inversion` in `inverting.py`; rules: `priors` in `qc_config.json` (`PriorRules`);
tests: sigpipe's `tests/masw/test_inversion_priors.py`. What they change in values given (by the user, or by the loop) is
logged as notes of the window's inversion attempt, and the agent reads them in the summary.

## What they set

Since 2026-09-27 the layers are chosen by the data unless given (`layering`, the rules'
`layering: "free"`): layers given, or the fixed layering named, keep the table below.

### The layers chosen by the data (the default)

| Parameter | Derived (default) | A value given must |
|---|---|---|
| Vs range (`free.vs_min`, `free.vs_max`) | 100 to 2,000 m/s, every layer's; lower (0.8 x the curve's slowest velocity) or higher (1.5 x its fastest) where the curve needs it | reach down to the curve's slowest velocity and up to 1.09 x its fastest; else set to 0.8 x (or 1.5 x) the curve, with a note |
| Interfaces (`free.depth_min`, `free.depth_max`) | from a third of the shortest wavelength (thinner is not resolved) to half the longest (deeper is not resolved); each layer at least 15 % of its depth thick | the deepest no deeper than half the longest wavelength (set to it, with a note) |
| Layers at most (`free.max_layers`) | 8, the half-space among them; G5 allows 2 more where the models pile at the most, up to 10 | kept |
| Vs drop (`max_vs_drop`) | 20 %: a layer's Vs at least 0.8 of the one above (a stiff layer over a much softer one makes the forward model's fundamental mode a wave trapped in the soft layer) | kept |

### The layers given

| Parameter | Derived (default) | A value given must |
|---|---|---|
| Vs bounds | wide: 100 to 1,000 m/s for the layers, 100 to 2,000 m/s for the half-space; lower than 100 (0.8 x the curve's slowest velocity) for a curve slower than that, higher (1.5 x) for a curve the half-space or the top layer could not reach (the steps written, (max - min) x 20/900, are not used: the chains' moves follow the posterior) | the top layer reaches down to the curve's slowest velocity, the half-space up to 1.09 x its fastest (Vs over Vr in a half-space at PAC's Vp/Vs of 1.77); else that bound is set to 0.8 x (or 1.5 x) the curve |
| Thickness bounds | at least a third of the shortest wavelength (thinner is not resolved); the half-space's top at most half the longest wavelength, shared equally between the layers above it | keep the same limits: a thinner minimum is raised, maxima summing deeper are scaled down |
| Layers, the half-space among them | 4 (PAC's 2 until 2026-09-25; the user: "4 5 6, never do 2"); G5 adds one when the model misfits, up to what the curve resolves (at most 10), removes one when a layer piles at its thinnest | be at least 3 (raised, its Vs and thickness ranges derived again), and not exceed what the curve resolves (layers of the thinnest resolved thickness down to the deepest, at least 3): reduced. Several Vs ranges given without a count are that many layers; when the count changes, a range the same for every layer stays every layer's, ranges that differ are derived again (with a note). The default count is fitted to the curve without a note |
| Effort (both) | PAC's: 150,000 iterations, a quarter as burn-in, 4 chains (100,000, a tenth and 5 before 2026-09-27) | keep at least 150 iterations after the burn-in (sigpipe keeps one model every 150) |

A re-inversion starts from the window's latest parameters with the changes on them, through
the same checks; another number of layers spreads the previous ranges over the new count and
keeps the model's depth, other iterations without a burn-in get a quarter of them. A number of
layers asked of a window whose data chose them gives it the layers given, derived from its
curve.

## On the demo profile

The 17 curves of the dense line (see `G5.md`) have velocities of 163 to 291 m/s and
wavelengths of 2 to 11 m: bounds of about 130 to 440 m/s, layers of at least 0.7 to 1 m, the
half-space's top at most 3 to 5.5 m. Against PAC's form (Vs 100 to 1,000 m/s, thicknesses 1 to
10 m), with the same effort:

| | PAC's bounds | Derived bounds |
|---|---|---|
| Interface depth | 2.4 to 7.5 m (6 to 7.5 m on most windows) | 1.9 to 3.9 m |
| Half-space Vs | 153 to 573 m/s | 169 to 237 m/s |
| Its chains' medians apart by | 5 to 97 % | 2 to 11 % |
| Posterior spread over prior spread, below 7 m | 0.6 to 1.1 (the prior's) | (models end at 4 to 6.5 m) |
| Misfit, layered median | 0.29 to 1.28 | 0.23 to 1.08 |
| Misfit, smooth median | 0.31 to 3.19 | 0.23 to 1.49 |
| Acceptance | 22 to 40 % | 47 to 76 % |

With PAC's bounds the sampler puts the interface below what the curves resolve, and the
half-space's velocity is whatever the prior allows; the derived bounds keep the whole model
within reach of the data. Both fit the curves: the uncertainties are 8 to 15 % of the
velocity.

## Values given, checked

On xmid 8.88 (189 to 291 m/s, 3 to 11 m), with 3 layers, Vs bounds of 200 to 300 m/s,
thicknesses of 0.5 to 10 m and 20,000 iterations given:

```
vs_min 200 m/s above the curve's slowest velocity (189 m/s) in layers 1, 2, 3: set to 151 m/s.
vs_max 300 m/s below 1.09 times the curve's fastest velocity (291 m/s) in layers 1, 2, 3: set to 436 m/s.
thickness_min 0.5 m thinner than the curve resolves (1 m) in layers 1, 2: set to 1 m.
thickness_max puts the half-space as deep as 20 m, below the 5.50 m the curve reaches: scaled by 0.27.
```

The 20,000 iterations stay, with 2,000 of burn-in; 12 layers are cut to the 6 the curve
resolves.

## Decided

The first inversion wide, 100 to 1,000 m/s for every layer and 100 to 2,000 m/s for the
half-space (the user, 2026-09-26), in place of 0.8 x to 1.5 x the curve for every layer: with
bounds that narrow the sampler accepted about half its proposals and the per-layer medians fell
at the middle of the prior, flat models. The loop then adapts each window (`G5.md`): steps,
depth, bounds and layer count.

Kept as they are (the user, 2026-09-25): Vs from 0.8 x to 1.5 x the curve (a half-space under a
stiff top reads about 1.09 x the slowest phase velocity at long wavelengths, above 0.8 x), the
half-space's top at most half the longest wavelength, and one set of bounds for every layer
(bounds by layer would impose a trend the data should decide).

At least 3 layers, the inversion starting at 4 (the user, 2026-09-25: "4 5 6, never do 2";
the demo's 2-layer models were too simple to be told from a trend). 4 after the study of 4, 5
and 6 starting layers (`G5.md`): every count fits within the errors, more layers are slower and
converge less often.

Fixed the same night (found by the study): a curve whose depth is just over a whole number of
its thinnest layers got a layer too many, each with a thickness range of a few centimetres or
none once rounded, and failed validation (6 layers on the demo's xmid 16.50). The count the
curve resolves now stops where every layer keeps a range, the thickness step is 1 cm at least,
and a curve too short for 3 layers (a span of wavelengths under 1.33, none on either line: 1.5
at least) is refused with a message saying so.
