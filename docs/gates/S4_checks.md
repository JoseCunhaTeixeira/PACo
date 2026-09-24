# The checks before S4, the inversion's parameters

Not a gate: rules checked on each window before its inversion, so that the inversion's bounds
come from the curve it inverts (the user's decision of milestone 13: per window). Code:
`src/paco/qc/priors.py` (`derive_inversion`), called by `judge_inversions` and
`rerun_inversion` in `inverting.py`; rules: `priors` in `qc_config.json` (`PriorRules`); tests:
`tests/test_qc_priors.py`. What they change in values given (by the user, or by the loop) is
logged as notes of the window's inversion attempt, and the agent reads them in the summary.

## What they set

| Parameter | Derived (default) | A value given must |
|---|---|---|
| Vs bounds, every layer | 0.8 x the curve's slowest velocity to 1.5 x its fastest; steps of (max - min) x 20/900, PAC's proportion | reach down to the curve's slowest velocity, and up to 1.09 x its fastest (Vs over Vr in a half-space at PAC's Vp/Vs of 1.77); else that bound is set to the derived one |
| Thickness bounds | at least a third of the shortest wavelength (thinner is not resolved); the half-space's top at most half the longest wavelength, shared equally between the layers above it; steps of (max - min)/9, PAC's proportion | keep the same limits: a thinner minimum is raised, maxima summing deeper are scaled down |
| Layers | 2, PAC's; G5 adds one when the model misfits, removes one when a layer piles at its thinnest | not exceed what the curve resolves (layers of the thinnest resolved thickness down to the deepest): reduced |
| Effort | PAC's: 100,000 iterations, a tenth as burn-in, 5 chains | keep at least 150 iterations after the burn-in (sigpipe keeps one model every 150) |

A re-inversion starts from the window's latest parameters with the changes on them, through
the same checks; another number of layers derives every layer's bounds again, other iterations
without a burn-in get a tenth of them.

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
vs_min above the curve's slowest velocity (189 m/s) in layers 1, 2, 3: set to 151 m/s.
vs_max below 1.09 times the curve's fastest velocity (291 m/s) in layers 1, 2, 3: set to 436 m/s.
thickness_min thinner than the curve resolves (1 m) in layers 1, 2: set to 1 m.
thickness_max puts the half-space as deep as 20 m, below the 5.50 m the curve reaches: scaled by 0.27.
```

The 20,000 iterations stay, with 2,000 of burn-in; 12 layers are cut to the 6 the curve
resolves.

## To judge

- Vs from 0.8 x to 1.5 x the curve (the spec's example): on the demo's inverse windows the
  half-space's Vs presses against 0.8 x (5 to 9 % of the samples within 2 % of it, `G5.md`).
  A lower factor (0.6) would let it go; is a half-space slower than 0.8 x the slowest phase
  velocity plausible here?
- The half-space's top at λmax/2: the common rules of thumb range from λmax/3 to λmax/2. The
  demo's useful depths (G5) end at 2.3 to 4.3 m where half the longest wavelength is 3 to 5.5 m.
- The bounds are the same for every layer. A stiff top layer over a softer one (11 of the 19
  demo curves) and the reverse both fit within them: should the top layer's bounds follow the
  short wavelengths, the half-space's the long ones?
