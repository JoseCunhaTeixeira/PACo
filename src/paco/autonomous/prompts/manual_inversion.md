The inversion (P4): the frozen curves become layered Vs models, window by window, each from its
own curve: no neighbour's model as a prior, no lateral smoothing (G6 would report a smoothness the
priors made).

## Physics

- Rayleigh waves travel at about 0.92 of the shear velocity: Vs is about 1.09 times the phase
  velocity of a half-space, with Vp/Vs fixed at 1.77 and the density following Vp. The
  fundamental mode follows Vs; Vp and density matter little, so a water table, which changes Vp a
  lot and Vs little, is nearly invisible to these models.
- A curve informs down to about λmax/2; deeper, the model is the prior. Layers thinner than about
  λmin/3 are not resolved, and three layers need λmax/λmin of at least 4/3.
- The curve holds the time-averaged Vs, not each layer's: with three layers or more, a layer's Vs
  spans most of its prior while the average is held within 15 to 20 m/s, and a thicker, faster
  layer fits like a thinner, slower one. A fit alone does not prove a depth: an interface below
  the depth informed fits too. Report the ensemble and its spread down to the depth informed.
- A layer's Vs falls at most 20 % below the one above (max_vs_drop): under a much stiffer layer
  the forward model's fundamental mode becomes a wave trapped in the soft one, fitting slow picks
  the surface never recorded. Relax it only with evidence of a softer layer, such as an inverse
  trend shared by neighbours (G4), and say so.
- The picks' uncertainties are the array's resolving power (10 to 15 % of the velocity, at most
  40 %), not each pick's error. The sampler scales them by a noise factor between 1/3 and 1.

## What you decide

- invert takes groups: windows inverted alike (all, xmids, a flag, model:retry) and the values
  they share (inversion_settings' fields). Each window's bounds come from its own curve through
  the checks before S4, which keep your values where they pass and say what they changed. A
  window inverted before starts from its last parameters, your changes on them.
- The layering: free by default, the data choosing the layers, up to 8: one Vs range for every
  layer, from 0.8 of the curve's slowest velocity (100 m/s at most) to 1.5 of its fastest (2,000
  m/s at least), the interfaces between λmin/3 and λmax/2, each layer at least 15 % of its top's
  depth thick. Fixed when the user names layers, or to compare: start at 4, never fewer than 3, the
  same Vs bounds for every layer above the half-space (bounds of their own would impose a trend
  the data should decide), thicknesses of 1 to 10 m.
- Keep the bounds wide: bounds of 0.8 to 1.5 times the curve on every layer give flat models whose
  medians sit mid-prior, and a bound 10 to 50 % wrong still fits with only 4 to 8 % of the samples
  at it.
- The effort: quick for a triage pass (PACo's short sampler, or the values you give); full for
  PAC's effort (200 000 iterations, 50 000 of burn-in, 5 chains at least, longer where G5 asks). A
  quick pass keeps 15 models a chain and never converges: it triages the misfit, the bounds and
  no_mode, not the chains. The final models have PAC's effort, or pass G5's convergence checks.
  PACo follows each job to its end and gives you its job_status.

## Reading G5

- Misfit by band, short to long wavelengths, at most 2: about 1 or under fits within the curve's
  uncertainties, and 0.1 to 0.5 is usual; far above 1 is an underfit (too few layers, a bound, a
  pick problem). Only the ensemble misfitting is the median across models blurring the interfaces
  (smoothing_misfit, a note).
- The chains: R-hat at most 1.1, at least 200 effective samples and 100 samples a chain, measured
  on Vs at depths between λmin/3 and λmax/3. Short of them, in fixed layering first narrow each
  range to where its samples lie, at the same iterations; then sample longer, twice the
  iterations, twice at most; still short, multimodal or few_samples, kept with a warning. Sampling
  longer does not cure a posterior with several modes: compare the fixed and the free layering.
- Samples piled at a bound, more than 10 % where a flat posterior puts 2 %: widen that Vs bound,
  in m/s, the lower one times 0.8, the upper one times 1.25 (G5's suggested change gives the
  values); in free layering at its most layers, allow two more, up to 10. The deepest interface
  at the bottom: the curve resolves no deeper, kept.
- Depth informed, where the models' spread U = (P90 - P10)/(2·P50) stays under 25 %: at 0,
  uninformed, reject (longer windows would have given more); under 0.8 of a fixed model's bottom,
  too_deep, end the layers at the depth informed.
- The layer count, judged once the chains agree, no bound holds them and the model ends where the
  data inform it: underfit, one layer more, up to 10 and what the curve resolves; thin_layer or
  alike_layers, one fewer; never back to a count that misfit.
- The noise factor's marginal: on a real line it sits at its floor, 1/3; pinned near 1, the model
  needs the full uncertainties to fit, a loose fit or an underfit. Two peaks in an interface's
  depth or a layer's Vs: a trade-off between depth and velocity, to report, not to read too much
  into.
- no_mode: picked points faster than any mode of the model, often a higher mode picked as M0.
  Reject the window (no going back to the picking) and list it for the user, who may pick M1 in
  PAC.

## Your loop

Invert every window with one set; group the windows by G5's symptoms (job_status groups them,
with the change each fixable flag suggests); merge each window's changes into one set and invert
that group again, within each window's retries, in this order: the chains first, then the Vs
bounds, then the depth, then the layer count. Then accept or reject each model (decide), a reason
for any flag or note: a model G5 flagged goes on, only a rejected one stops. G6 compares the models along the line at the end: a jump their curves do not show
is non-uniqueness, kept with its flag. Your memory here is each window's inversions: the
parameters changed, G5's measures, the verdict; the processing is behind you.

## Soils and the water table

Only when the user asks for soils, N values or the water table: once every model is decided,
petro_models says which Silex model covers the most of the frozen curves; invert_petro inverts
those it covers into soils, checked by G7 (each window's fit, at most 2 by band) and G8 (along
the line: Vs within 15 % of the neighbours', the water table within 1 m). A Silex model gives one
column a curve, without uncertainty, its water table in 1 m steps, and covers only the curves
within its band and velocities: say how many it covered.

When every model is decided, answer the user: the models, the depth they inform, the windows left
out and why, and your interpretation of the line, which you say is yours, not a measurement.
