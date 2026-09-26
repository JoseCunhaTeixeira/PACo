# Task: a quality-controlled processing loop for PACo (active MASW first)

Read PROGRESS.md first and keep its working agreement: you build, run and check each step yourself; real design decisions come to me as options, and you say which one you would pick; I commit. Save this spec as `docs/qc_workflow.md` so it outlives the session, and record progress, decisions and open issues in PROGRESS.md as for the earlier milestones.

## Goal

Today the agent processes, judges, picks and inverts with a single quality gate (`dispersion_quality`) and asks me to approve the curves before inverting. I want it to work the way a geophysicist does, on its own: after each stage, check the result; if it is not good, change the parameters of that stage or of an earlier one, and redo only what is affected, until each xmid passes or is rejected with a reason. The end result is one Vs model per xmid, each traceable to the parameters and QC verdicts that produced it, plus a QC report for the whole line.

The default is that I give no parameters at all: "process <profile>" must run the whole loop and end with Vs models and a report, every parameter derived from the data and the geometry, and the agent must not ask me a single question. Settings I type in the chat are the exception, and even those are not a reason to ask (see "Agent side").

Scope: active profiles. Passive will get extra gates later (segment selection, FK selection, cross-correlation convergence), so build the gates as a generic mechanism passive can extend. Gates G2 to G6 must already run on passive runs unchanged; on passive, G1 runs only the checks that do not need a trigger (dead/clipped traces, spectra).

## Pipeline

Artifacts in [ ], gates in < >.

    [raw record per source position] + geometry
      S1 preprocessing, per record (trace editing, filtering, muting, normalisation...)
    [preprocessed record per source position]
      <G1 signal QC, per record>
      S2 phase shift, per xmid
    [dispersion image per xmid]
      <G2 dispersion image QC, per xmid>
      S3 picking, per xmid
    [dispersion curve per xmid]
      <G3 curve QC, per xmid>
      <G4 curve profile QC, whole line>
      gate decision: S4 starts for the xmids G4 passed; the rejected ones are reported, never inverted
      S4 MCMC inversion, per xmid
    [Vs model per xmid: PAC's smooth model is the one monitored and reported]
      <G5 model QC, per xmid>
      <G6 model profile QC, whole line>
    [final Vs model per xmid + QC report]

    Only when the user asks for soils or the water table, from the curves G4 passed (milestone 15):
      range check: the curves the chosen Silex model's trained band and velocities cover; the others are reported, never inverted
      S5 petrophysical inversion, per xmid (Silex: soils, N values, water table; santiludo's rock physics)
    [petrophysical model per xmid]
      <G7 petrophysical model QC, per xmid>
      <G8 petrophysical profile QC, whole line>
    [PAC's petrophysical sections, over the xmids G7 and G8 passed]

## Rules for every gate

1. Output per unit (record or xmid): verdict `pass`, `retry` or `reject`; each metric with its threshold; the flags raised; and for each flag, the stage most likely at fault (this one or an earlier one) and a concrete suggested change as overrides that can be applied as they are, not only a sentence.
2. Fixable vs not fixable: a dead geophone, a shifted trigger, or a window without enough offset coverage does not improve with parameters. Actions a gate can advise: change parameters of stage k, exclude traces, exclude a record, reject the xmid. `reject` is a legitimate end state, with its reason.
3. Cheapest fix first: re-pick before re-running the phase shift, before re-preprocessing. Going back to stage k invalidates everything downstream, for the affected xmids only; the others keep their results.
4. Bounded: a retry budget per gate and per unit, and a total budget per run, all configurable. When a budget is spent, the unit is rejected with "budget spent" and the last attempt's flags. No loop can run forever. Stop at the first attempt that passes.
5. The judge is fixed during the loop: the loop changes processing, picking and inversion parameters, never QC thresholds; only I change thresholds, outside a run. This reverses the milestone-4 decision that let the agent change them: confirm it with me.
6. A fix must not win by throwing data away. Each gate also reports what was kept: usable band, wavelength range, number of points, traces and records. A retry that passes by shrinking this well below the previous attempt, or below the line's median, is flagged. Example: a mute that sharpens the image but loses the low frequencies.
7. Parameters come from the data and the geometry, not from defaults (see "Coherence rules"). Check them before any work, like `resolving.py`, with errors written for the agent. PAC's form defaults (3-receiver windows) are not acceptable derived values.
8. Everything is recorded. Each attempt (unit, stage, parameters, metrics, verdict, what triggered it) goes into a QC log in the run folder. The final report gives, per xmid: final parameters, number of attempts, verdict, reasons for any reject, and for each inverted xmid the wavelength range and point count of the curve it used, so I can spot a bad pick afterwards in PAC's UI and re-invert it. These records are also future labelled data (see "AI picker: later").
9. Thresholds are provisional: all in one place, configurable, recorded with each verdict, tuned on the demo profiles only for now, and to be recalibrated on hand-judged windows.

## Gates

### G1 signal QC (per preprocessed record, with the raw one when useful)
- Dead, clipped or NaN traces; traces whose RMS is an outlier against the amplitude decay with offset; reversed polarity (negative correlation with neighbours).
- SNR: energy in the surface-wave window (between the arrivals at vg_max and vg_min) against a noise window: pre-trigger if the record has one, otherwise after the slowest arrival if the record is long enough. Report which window was used.
- Usable band: frequencies where the signal spectrum exceeds the noise spectrum by N dB, giving fmin_usable and fmax_usable. This band feeds S2.
- Lateral coherence: median correlation between neighbouring traces inside the surface-wave window.
- Trigger consistency: first-break moveout consistent with the source position (a shifted t0 is a bad record, not a parameter problem).
- When a mute is on: share of energy removed, taper present.
- Actions: filter band; mute on/off with vg_min, vg_max and taper (keep t in [x/vg_max - pad, x/vg_min + pad]); exclude traces; exclude the record from the windows that use it.

### G2 dispersion image QC (per xmid, before picking)
- Coherent energy: fraction of frequency columns whose maximum exceeds k times the noise floor 1/sqrt(N) (the floor the picker already uses).
- Energy maximum on the grid edges: at vmin or vmax means the velocity range is too narrow; at fmin or fmax means the band is too narrow (reuse the "pinned" idea).
- Competing ridges of similar strength at the same frequency: a higher mode dominating, or aliasing (compare with the aliasing limit v = 2·dx·f).
- Coherent band much narrower than G1's usable band: the problem is in S2 or S1 (offsets, records used, mute), not in the data.
- Actions: vmin/vmax, fmin/fmax, steps, offset range and records feeding the window, or back to S1 (a surface-wave mute between the expected vg_min and vg_max often cleans the image a lot).
- The current metrics (sharpness, prominence, on_data, constant_wavelength) are computed on the picked M0, so they belong to G3. Don't duplicate them.

### G3 curve QC (per xmid)
Since nobody looks at the curves before inversion any more, G3 is the safety net for the picking risks listed in PROGRESS.md. The current `dispersion_quality` metrics, plus:
- Wavelength range: λmin >= 2·dx; points beyond λmax (between L and 2L, configurable) are flagged and cut. On active_p1, 64 to 91 % of the saved points lie beyond 2L.
- Mode jump: a velocity jump between adjacent frequencies above X %, or a branch continuing on M1 below the aliasing floor.
- Air wave: near-constant velocity around 330 to 345 m/s.
- Normal dispersion expected (velocity rising with wavelength). An inverse trend is flagged, not rejected, since a stiff layer over a soft one produces it.
- Minimum number of kept points; Lorentzian uncertainty not above X % of the velocity.
- Actions: picking parameters (corridor, coherence rule, max wavelength), or back to S2 or S1.

### G4 curve profile QC (whole line)
- Build the pseudo-section Vr(λ, xmid) from the passing curves.
- Compare each curve with the median of its k neighbours at the same wavelengths; a relative misfit above X % marks an outlier.
- An isolated outlier (1 or 2 xmids) is suspect. A change shared by a run of adjacent xmids is geology and is kept.
- Coverage: runs of rejected xmids, and wavelength ranges that vary a lot along the line (depths of investigation not comparable).
- Actions for an outlier: re-pick with the corridor centred on the neighbours' median curve, then G3 again; otherwise reject. Never edit or smooth a curve to make it fit.
- G4's verdict is the go/no-go for S4: its record in the QC log is what `invert` reads.

### Checks before S4 (inversion parameters coherent with the curve)
- Vs bounds bracket the curve: Vs ≈ 1.09·Vr at PAC's Vp/Vs of 1.77, with margin, e.g. [0.8·Vr_min, 1.5·Vr_max], configurable.
- Depth: the top of the half-space (sum of the thicknesses) is not deeper than about λmax/2. Layers thinner than about λmin/3 are not resolved.
- Layers: start from PAC's 2. The loop may add one when the misfit stays high and remove one when it is not resolved.
- The burn-in rule stays (at least SAVE_EVERY = 150 iterations left to sample).

### G5 model QC (per xmid)
- Misfit: RMS, and normalised by the curve's uncertainties (about 1 means a fit within errors; much above 1 is underfit; much below 1 is suspicious). Compute it from the forward-modelled curve of the monitored smooth model.
- Convergence: the chains agree (R-hat on Vs at a few depths, or chain medians within X %); acceptance rate within a configurable band; enough saved samples.
- Posterior piled at a prior bound (Vs or thickness): widen that bound.
- Posterior width against prior width, by depth: report the depth where posterior ≈ prior (the useful depth). Don't fail on this alone.
- Actions: n_iterations and burn-in (convergence), bounds, number of layers. A failing xmid can be re-inverted with its own settings, recorded in the log.

### G6 model profile QC (whole line)
- Vs at fixed depths and interface depths along the line.
- An isolated jump that the curves don't show (G4 found the neighbouring curves agree) is non-uniqueness: re-invert that xmid (more iterations, other bounds or layer count). A jump the curves also show is kept.
- Useful depth consistent along the line.
- Don't build any lateral smoothing, neither by editing models nor by using neighbours as priors: it would create the smoothness it then reports.

### Range check, G7 and G8: the petrophysical inversion (milestone 15)

The user's decision (2026-09-26, "Range, fit, line"), and `docs/gates/G7.md`, `G8.md`:

- Range: a curve outside the chosen Silex model's trained band and velocities (beyond Silex's own 20 % margin) is not inverted; the agent says how the curves fall outside ("5 end below 43 Hz") and chooses the model covering the most (`petro_models`).
- G7, per xmid: the curve the predicted soil column gives back against the pick, by band, as G5 (at most 2 per band); points with no fundamental mode, and a pick with no uncertainty, reject.
- G8, whole line: the rock physics' Vs at fixed depths (G6's rule) and the water table against the neighbours; an outlier the curves do not show is left out, a change they show is kept.
- No retry: a model gives one soil column per curve. A window G7 or G8 rejects is left out of the sections; its flag says what could change it (another model covering the curve, or the pick).

## Coherence rules for S2 (checked before running)
- Window length: one value for the whole profile, so lateral resolution and depth of investigation stay comparable and xmids don't move.
- Step: 1 receiver by default (every xmid). Per-xmid retries change only parameters that don't move the xmid: band, velocity range, offsets, records used, mute.
- Frequency range: fmin >= fmin_usable from G1; fmax <= min(fmax_usable, Nyquist). The lowest useful frequency follows from λmax (L to 2L) and the expected velocities.
- Velocity range: vmin and vmax bracket the expected velocities (from G1's moveout, or a first image).
- Frequency step: a step finer than 1/T (T = record length after the mute) only interpolates.
- Offsets: the nearest source offset is large enough against near-field effects (e.g. >= λmax/2, configurable); the farthest stays within the SNR that G1 found.

## Agent side
Constraints from PROGRESS.md: Qwen3 4B/8B, 16k context, one tool call per reply, PACO_MAX_TOOL_CALLS = 15, the model sees summaries only, tool cards have a measured budget.
- The agent asks me nothing. Every parameter is derived and tuned in the loop; the go/no-go before inversion is G4's verdict. `invert` reads G4's record from the QC log, like `pick` reads `quality.json`, and refuses a run without it. Remove the elicitation in `invert` (milestone 5) and the host's y/N; update the README's safety section.
- Settings I type in the chat are the only exception to "derived": the loop starts from them. If a gate then concludes one must change, the agent changes it and says so in its answer. It still does not ask. Replace the current "ask before changing" rule in the role, and relax the `only_called` checks that enforced it.
- Gates work on a whole run and return one short summary: counts per verdict, flags grouped by runs of xmids (e.g. "xmid 12-18: ridge at vmax"), with the suggested overrides per group. Never one line per xmid.
- Retries target a subset: redo stage k for listed xmids, or for the ones carrying flag X, with overrides.
- Placeholders, not values, in tool descriptions (small models copy example values).
- Raise PACO_MAX_TOOL_CALLS only with evaluation evidence.
- Evaluation scenarios, on synthetic variants of the demo profiles with known defects:
  - Zero settings: "process active_p1 and give me the Vs models" with nothing else; the agent must not ask a single question, and the run must end with models for the xmids G4 passed and reasons for the rest.
  - G1: a dead trace and a shifted trigger.
  - G2: a velocity range too narrow.
  - G3: a mode jump and an air-wave pick.
  - G4: an isolated outlier xmid; and a line whose curves are all bad, where no inversion may start.
  - G5: too few iterations, and a bound too tight.
  - A setting I typed that a gate must change: the answer says so.
  - Every scenario checks that thresholds were never changed and that no question was asked.
- The old approval scenarios (declined, approved) go; `inversion_succeeded` is only checked when G4 passed something.

## Build order (one milestone each; ruff, pyright and pytest green before the next, then `paco-evaluate --repeat 3`)
- M9. Split `run_processing` into preprocessing, with persisted preprocessed records, and phase shift. Results must be identical to today on the demo profiles (parity test).
- M10. The gate framework, domain code only (no mcp/openai): verdict, flag and advice models; QC log; retry budgets; downstream invalidation; partial re-runs on a list of xmids; final report.
- M11. G1 and G2, and move the existing metrics to G3.
- M12. The G3 extensions, and G4.
- M13. The checks before S4, the smooth model as the monitored result, G5 and G6.
- M14. Tools, the removal of the approval, the agent's role and policy (the option chosen below), evaluation scenarios, README.

Tests as before: each metric on inputs with a known answer (synthetic records with a dead trace, a clipped trace, a known SNR; images with the ridge at vmax; curves that carry exactly the picking risks: points beyond 2L, the air wave, a jump onto M1; posteriors piled at a bound), mutation checks, and parity with PAC wherever PAC does the same thing.

## Decisions to bring me as options before building
- Where the parameter search lives:
  - (A) The LLM applies each suggested change and calls the tools again. Most faithful, but many calls and a growing context.
  - (B) Each stage tool runs its own bounded retry loop with its gate's suggestions. The LLM handles backtracking across stages and explains what it did.
  - (C) A mix.
  - Measure calls and tokens per scenario before choosing.
- How attempts are stored: a new run per attempt plus a per-xmid selection, or an attempt history inside one run. Either way, the final result must sit in PAC's xmid folders so PAC's UI opens it.
- Thresholds locked during the loop (reverses the milestone-4 choice).
- Default values for every new threshold and retry budget.
- Which of PAC's five models is the smooth model PAC shows by default (check PAC), and whether `job_status` reports it instead of the median.

## Decisions taken (2026-09-24)

Brought as options, with the evidence, before building; each was the user's choice, and Claude's.

- **Search loop: B.** Each stage tool runs its own bounded retry loop with its gate's suggested
  overrides and returns one summary; the LLM only backtracks across stages and explains.
  Evidence: a straight pass already took 4–8 calls and 2.2–3.4k-token prompts with Qwen3-4B; the
  one play that looped burned all 15 calls and invented a result.
- **Attempts: history inside one run.** The final result keeps PAC's layout at the top of each
  `xmid_<x>/` folder; earlier attempts go under `xmid_<x>/attempts/<n>/` with their files,
  parameters, metrics and verdict; the QC log sits at the run level. One window weighs 2.4 MB.
- **Thresholds: locked during the loop.** One configuration, changed by the user between runs,
  recorded with every verdict; the tools stop taking thresholds. Reverses milestone 4.
- **`job_status` reports the smooth median model**, PAC's default (its visualization page starts on
  "Smooth median layered model", the backend defaults to `smooth_median`), as Vs at a few fixed
  depths and the useful depth, plus each window's curve misfit.
- **Budgets:** 2 retries per gate and unit, 2 × xmids per run, all configurable.
- **Thresholds' values:** proposed per gate at its milestone, measured on the demo profiles and
  the synthetic defects, confirmed by the user; all in one configuration file.

Taken during milestone 9:

- **No padding before the phase shift.** PAC pads every trace (`Pad(n=1000, taper=25)`) before
  the phase shift; the user removed it from both image pipelines (2026-09-24): a frequency step
  finer than 1/T only interpolates (the coherence rules above). Claude would have kept PAC's
  step for parity. So "today's results" for the parity test are PAC's steps without the padding;
  the images now have a 0.5 Hz step on the 2 s demo records (0.4 Hz before), and on the passive
  demo the window at xmid 2.88 went from doubtful to bad.
- **Windows of 5 receivers by default** (the user, 2026-09-24: "24 is a lot, start with 5"),
  where PAC's form has 3. The 24 of the tests and scenarios is an explicit request. The derived
  value comes with the coherence rules.
- **A consistent trigger delay is corrected in S1** (2026-09-24; the user's choice, and Claude's).
  G1 found both demo records triggered about 18 ms late (first breaks at 19–21 ms on the
  0.75 m trace). The active preset gets a `trigger` stage with `t0` (0 by default): G1's flag
  suggests the measured `t0` as an override, and the record is shifted by it. A t0 that differs
  from trace to trace stays a reject. Rejected: rejecting as the spec first said (the demo line
  would have no window left), and flagging without acting (the mute windows x/v would stay
  wrong by the delay).
- **G1, G2 and G3 thresholds** (2026-09-24; proposed from measurements on the demo profiles,
  accepted by the user): see `docs/gates/G1.md`, `G2.md` and `G3.md`, each with the values,
  the reasons, the demo's real outputs and the synthetic cases. Rejected: loosening the
  trigger to 25 ms, flagging second ridges from 50 % of the columns, loosening prominence to
  1.2 for 5-receiver windows (a finding about the window length, not the threshold).
- **The agent's loop: plan, act, observe, adapt** (the user, 2026-09-24). With option B the
  tools run the retries within a stage; the agent keeps that loop across stages: it plans the
  stages, acts by calling them, observes each gate's summary, and adapts by backtracking with
  the suggested overrides. The role and the evaluation of milestone 14 must show all four.

Taken during milestone 12:

- **G3's curve rules and G4's thresholds** (2026-09-24; proposed from measurements on the demo
  profiles, accepted by the user, see `docs/gates/G3.md` and `G4.md`): the cut at 2 window lengths as the picking's
  starting `max_wavelength` (the saved curves reached 33 to 81 m for a 5.75 m window; 5 to 10
  points remain at 2 to 11 m), a mode jump at 30 % between consecutive points of the resampled
  curve (the good demo curves step by up to 26 %, the two bad ones by 31 and 53 %), the air wave
  at 330–345 m/s over half of the points (0 % on the demo), an inverse trend by the rank
  correlation of velocity with wavelength (kept, never rejected: 11 of 19 demo windows), at
  least 5 points, uncertainties at most 50 % of the velocity (8 to 15 % on 24-receiver windows,
  56 % on 5-receiver ones, rejected: only longer windows help). G4 compares each curve with two
  neighbours a side: an outlier is off neighbours that agree with each other and fits no side
  (misfit 15 %; the demo's neighbours agree within 2 to 13 %); a curve off one side and fitting
  the other is a shared change, geology, kept. Rejected: the spec's "1 or 2 xmids" for an
  isolated outlier (two changed windows make their neighbours disagree, so nothing in a run of
  two can be told from geology by its neighbours), and a side of one curve as evidence (it
  agrees with itself: three curves 6 m apart said xmid 8.88 was an outlier).
- **The pick is saved as it is judged.** The picking attempt writes the window's
  `DispersionCurves_0000.csv` (PAC's layout) before G3 judges it; a pick done again archives the
  previous curve and the inversion under `attempts/<n>_picking/`. What G3 and G4 judge is the
  file the inversion reads.
- **G4 runs on the curves that passed G3**, the others are the line's gaps. A window whose G3
  verdict is `retry` is not a curve for the line until its retry passes.

Taken during milestone 13 (2026-09-24; brought as options with measurements on active_p1, each
the user's choice, and Claude's):

- **The window length: the shortest that passes.** Before S2, a ladder of lengths (5, 8, 12,
  16, 24, 32, 48 receivers, from the user's length when one is given) is tried on a few xmids
  spread along the line; the first length at which two thirds of them pass G3 is kept for the
  whole line: lateral resolution first. Measured on active_p1 (every 4 receivers, fmax 100 Hz):
  G3 passes 0 of 23 windows at 5 and 8 receivers, 0 of 22 at 12, 12 of 21 at 16, 17 of 19 at
  24, 15 of 17 at 32, 13 of 13 at 48. Rejected: the deepest length that passes (48, windows of
  12 m on a 24 m line), and full runs escalated by the gates (whole runs and tool calls).
- **The band: capped, never widened by G2.** fmax starts at the preset's (PAC's 100 Hz) and is
  capped at the smaller of G1's usable fmax and Nyquist; G2's `band_at_fmax` and
  `narrower_than_usable` become kept flags. Measured: widening the band made G3 worse at every
  length (24 receivers: 17 of 19 pass at 100 Hz, 15 at 150 Hz, G2's own suggestion, 6 at 324 Hz,
  G1's usable band): above 100 Hz a second ridge competes and the picker jumps. Rejected:
  keeping G2's retries and the better attempt (more phase shifts, a rule to compare attempts),
  and fmax from G1's band.
- **The inversion's bounds: derived per window**, from its own curve: Vs in
  [0.8 Vr_min, 1.5 Vr_max], layers no thinner than λmin/3, the half-space's top no deeper than
  λmax/2, 2 layers to start; bounds the user types are checked against them and changed with a
  message. Measured on the dense demo line's 17 curves: PAC's bounds (Vs 100–1,000 m/s,
  thicknesses 1–10 m) put the interface at 6 to 7.5 m, deeper than the curves resolve, and
  leave the half-space's Vs to the prior (the chains' medians 5 to 97 % apart); derived, the
  interface sits at 1.9 to 3.9 m, the chains agree within 2 to 11 %, the fits improve (0.23 to
  1.08 against 0.29 to 1.28, normalised by the uncertainties). Rejected: one prior for the line,
  and PAC's bounds checked only.
- **G5 judges the smooth median, and keeps what only the smoothing loses.** PAC's smoothing
  spreads a layer boundary over a few metres: with a thin stiff top layer, the smooth median
  misfits 2.3 to 3.2 where the layered median it comes from fits within 1.0 to 1.3 (PAC's
  bounds; 1.4–1.5 against 1.05–1.08 with derived bounds). When the layered median fits and only
  the smooth one does not, G5 keeps the window with a flag (re-inverting cannot fix a smoothing
  effect); when both misfit, it retries. Rejected: judging the layered median only, and a retry
  on any misfit of the smooth median.
- **The ladder's strictness: 9 trial windows, 80 %** (brought with the thresholds, after the
  build). As first built (5 trials, two thirds) it kept 16 receivers on active_p1, where only
  50 of the line's 81 windows passed G3 (62 %); 9 trials at 80 % keep 24, where 66 of 73 pass
  (90 %). Rejected: 5 trials at two thirds, and 9 at two thirds (still 16).
- **The near-offset rule: reported, not applied.** Keeping only shots at least λmax/2 from the
  window cost a third of the demo's curves (17 to 11 of 19 at 24 receivers). G3 reports the
  window's nearest shot against half its longest wavelength (`near_offset`), with a kept flag
  (`near_field`). Rejected: applying it, and dropping it.
- **G5's and G6's thresholds**, proposed from the demo and synthetic defects, accepted by the
  user (see `docs/gates/G5.md` and `G6.md`): misfit at most 2 per band, split R-hat 1.1,
  acceptance 10 %, 100 models a chain, 10 % of samples at a bound, up to 4 layers; models 15 %
  off agreeing neighbours, useful depths spread at most 50 %. Rejected: an at-bound limit of
  6 % (it would flag the demo's inverse windows, whose half-space presses against its lowest
  Vs, and reject them after two widenings although they fit), a misfit limit of 1.5, 20 % for
  models, and one more layer for a non-unique model instead of longer sampling.

Taken at the start of milestone 14 (2026-09-24; brought as options, each the user's choice, and
Claude's):

- **The tools: one per stage, each with its gate's retries, and `redo`.** `run_processing` runs
  S1, G1 and its fixes, the S2 rules, S2 and G2; `pick` runs S3, G3 and G4 (the re-picks
  included); `invert` runs S4, G5 and G6 as a background job, refused unless G4 passed the line;
  `job_status` follows it; `redo` goes back to a stage for some windows (listed, or those
  carrying a flag) with changes. `dispersion_quality`, `quality_settings` and the approval go.
  A straight run is 4 calls and the polling, and the agent observes after each stage.
  Rejected: one tool up to G4 (a single look before the inversion), and one job end to end
  (the agent would only report).
- **Synthetic defects in the data where they are physical**: copies of active_p1 with a zeroed
  trace (G1), a strong air wave (G3), a static shift on a few receivers (G4's outlier); the
  demo's own trigger delay and mode jumps; passive_p1 for a line with no good curve; typed
  settings for the rest (vmax too low for G2, 2,000 iterations and a Vs bound too tight for
  G5). Each defect is checked to trip its gate before it becomes a scenario. Rejected: typed
  settings only.
- **The evaluation runs 8 workers at PAC's effort**: about 6 minutes per play of the
  zero-settings scenario (the whole demo line inverted), where the default single worker takes
  about 50. Rejected: a lower effort in evaluations, and a shorter line.
- **The host refuses a call identical to one that just failed**, with the failure and "change
  it or answer the user". Seen: Qwen3-8B sent the same wrong `invert` call three times after
  milestone 13, Qwen3-4B looped ten times and invented a result. Rejected: leaving it to the
  model and the 15-call budget.

- **The agent asks only when stuck** (the user, during milestone 14: "I want the LLM to be able
  to ask me questions sometimes, if it is needed, so I can help him choose"; brought as options,
  the user's choice, and Claude's). It applies the gates' fixes and goes back a stage by
  itself, saying what it changed. It asks only when the data cannot decide: no image or no curve
  left for the line, windows rejected once the retry budget is spent, or a request the data do
  not allow (windows longer than the line). Then it asks one short question with 2 or 3
  concrete options, its choice first, and waits. This replaces "the agent asks me nothing" of
  the agent side above. Rejected: asking before changing a setting the user typed (the rule
  before milestone 14, which Qwen broke either way), and asking before every step back.

Taken on 2026-09-25, after milestone 14 (the user: "the windows should be smaller 5 7 9 11, you
are using too bigger windows", then "maybe it is better to let the agent think and find out the
best window size, depending on the profile length"; each brought as options with measurements
on active_p1, the user's choice, and Claude's):

- **The pick goes as far as its ridge holds, at both ends.** The cut at 2 window lengths goes
  (the user: "that rule is not good, I like the Nyquist limit"); the aliasing limit (2 x the
  spacing) stays. The pick is the longest continuous run of kept points: a step steeper than
  |d ln v / d ln f| = 2, or more than 2 Hz of columns too weak to keep, ends it. On the demo it
  ends at 12.5 to 16 Hz (medians) on 5- to 11-receiver windows, and under the 50 Hz mains line
  at the high end. Rejected: no bridged gap (ends at 18 to 22 Hz), a fixed 10 Hz floor, the
  continuity at the low end only.
- **A point is kept only where the window resolves velocity at all**: a perfect plane wave for
  the window, at the pick's frequency and velocity, must stand at least 1 % above its column's
  median. Some picks otherwise drifted smoothly to 5 Hz (160 m wavelengths on 3.75 m windows),
  0.5 Hz after a G4 re-pick. At 1 % the lowest points are 8.5 to 10 Hz on every length.
  Rejected: 0.5 %, 2 %, no floor.
- **G3 judges sharpness and prominence against a perfect plane wave for the same window.** On
  the demo both equal a plane wave's (ratio 1.00) at every length from 5 to 24 receivers, where
  the fixed limits failed every 5- to 11-receiver window: they measured the array, not the
  data. Prominence must reach half a plane wave's, counted up to 4 (beyond, any ridge stands
  out: the old fixed limit, 2, is half of it); sharpness 0.8 of its width. Rejected: prominence
  only, both as they were.
- **The ladder: 5, 7, 9, 11 receivers, then 16, 24, 32 ...** for a line where none of the
  short ones passes. Rejected: 5, 7, 9, 11 only.
- **The ladder proposes, the agent decides.** `run_processing` keeps the ladder's length when
  none is given, and returns every length tried (trial windows passing G3, the wavelengths
  their curves reach, windows on the line), one length past the proposed one included, for
  comparison; the agent runs again with another length when the request needs more depth or
  lateral detail, and says why. A length given, by the user or the agent, is kept as it is:
  the ladder no longer climbs past it. Rejected: the agent choosing from the trials on every
  run (a call more each time), the agent exploring whole runs freely, and climbing past a
  length given (the rule of milestone 13).

The review of the gates' "To judge" lists, the same day (brought as one round of options, each
the user's choice, and Claude's; the details in each page under `docs/gates/`):

- **The ladder's trials leave out the line's two end windows**, next to the shots, where every
  length from 5 to 11 receivers keeps 1 or 2 points. Rejected: keeping them. Replaced the same
  day, once measured on the whole line (below).
- **G1's retries count against each record's own budget**, not the windows' run budget.
- **A mode jump's first fix cuts the band where the jump sits**, the corridor halved second.
- **G4 says an inverse trend along the line once** (kept); **G5 reports PAC's residual by
  band** (not judged); **G5's `no_mode` suggests the phase shift again with the band below the
  points no mode reaches** (the window stays rejected; the agent's `redo`).
- **G1 leaving the near field out of its decay fit**: chosen, then dropped by the user after a
  try (`docs/gates/G1.md`): at G1 the longest wavelength is not known, and the record's dominant
  wavelength, tried instead, missed the traces the rule was for. Rejected with it: a fixed
  share of the line, and deciding after the picking.
- Everything else kept as it is, and the lists pruned of what milestone 13 and this day decided.

Then, after the evaluation of these changes (Qwen3-8B, 13 scenarios: 17 of 39 plays once two
check artefacts were fixed, the model reporting the gates' changes in few answers, closing
with a question in 7, inverting unasked in 6, never choosing a length from the table):

- **27 trials spread over the whole line, its ends included.** Without the end windows, nine
  trials all passed at 5 receivers where the whole line gives curves on 65 % of its windows;
  27 trials follow the line within a few % (5: 63 %, 7: 74 %, 9: 74 %, 11: 85 %, 16: 96 %) and
  propose 11 receivers (85 % of the line). The ladder takes a minute on the demo instead of
  30 s. Rejected: keeping the ends out, and 9 trials with the ends in (16 receivers).
- **What the host guarantees, whatever the model says** (`paco.agent.host`): the settings the
  gates changed are listed after every answer, as the tools gave them; an answer that asks or
  offers is asked again once, unless a tool said the agent is stuck; no inversion starts unless
  the user's message asks for models (invert, inversion, model, Vs, shear). The evaluation then
  measures the model and the host together. Rejected: documenting Qwen3-8B's limits only.
- **At least 3 layers, never 2** (the user): the inversions start at 4, 5 or 6 layers, to be
  measured; G5 adds layers up to what each curve resolves and removes one that piles at its
  thinnest, never below 3.

Then on a real survey, `active_p2` (the user's: 96 receivers every 1.5 m, 97 shots, "a more
realistic survey"), which PACo could not process as it stood (each brought as options, the
user's choice, and Claude's):

- **The farthest shot a window stacks comes from the data**: `masw.distance_max` where the
  traces' median SNR, by distance from the shot over all records, falls under 2 dB (63 m on
  `active_p2`, 24 m on the demo), unless the user gives one. Rejected: PAC's 100 m (noise
  beyond about 60 m there), every shot (PACo's 1,000 m until then), 6 dB and 3 dB (at 6 dB the
  demo's end windows lost the far shot that gave them a curve).
- **G1 judges a record's SNR and usable band within that reach**: 52 of 97 records were
  rejected for their far traces, which no window stacks. Rejected: excluding the traces under
  6 dB instead, keeping the rule as it was.
- **The band from the records' median usable band**, not the worst record's: 2.3 to 298 Hz
  against 24 to 86 Hz, which cut the line's long wavelengths. Rejected: keeping the worst.
- **No reversed-polarity check** (the user: "in geophysics this will likely never happen"): at
  1.5 m next to the shot it flagged whole blocks of traces. And G1's decay fit keeps the traces
  it excluded, which stopped a cascade of 21 exclusions around each shot (a fix, not a choice).
- **A window leaves out each record's own excluded traces** (the traces flagged in at least half
  of its records, from all), not every trace any record excluded: with 66 shots a window, every
  window ended empty. Fixed with the two G1 fixes above (the user: "I am sure there was no
  reversed polarity, fix it", against finishing on the demo alone).
- **G2's `band_at_fmin` kept, not retried**, like `band_at_fmax`: lowering fmin to 1 Hz never
  moved the band's low end, and spent the budget on 68 of 92 windows; the picker now decides
  where the low end stops. Rejected: one retry, then kept.

Decided by Claude on the night of 2026-09-26, the user asleep ("make it work well as an agent
for inverting active, passive and active-passive MASW, everything PAC does ... I trust you"),
each with its measurements in `PROGRESS.md` and the gate pages, to review:

- **The inversion starts at 4 layers** (the study of 4, 5 and 6 on both lines, `G5.md`).
- **G6 and `job_status` read the curve's depth of investigation**, half its longest wavelength:
  the posterior's useful depth is 0 m on 62 of the layer study's 72 models of 3 layers or
  more (2.1 to 3.5 m on the others).
- **PAC's passive-active mode** (interferometry on an active profile's shots), with two fixes
  of PAC's chain: the flipped gathers' geometry, and each shot cut to its surface-wave window
  before correlating (`correlation_window`).
- **PACo's passive defaults**: 2 s segments whitened and normalized one-bit, instead of PAC's
  0.1 s segments with neither (no curve on a real ambient-noise line).
- **G2 counts a peak at the grid's top velocity only where the window resolves that velocity**
  from an infinite one (f x aperture above vmax).
- A record G1 rejects goes into no window; the layer count at the curve's limit keeps every
  layer a range; given Vs ranges survive a change of the count.

Added by the user with the decisions:

- The fit between the inverted model's forward-modelled dispersion curve and the picked one is a
  QC in its own right (G5), reported by band, not as one number. PAC already saves the
  forward-modelled curve (`SeismicInversion_DispersionCurves_*_smooth_median.csv`) and compares
  the two in its pseudo-section comparison: match it.
- **Documentation of every QC tool, with examples.** Each gate gets a page under `docs/gates/`:
  what it checks, each metric with its threshold and why, the tool's real summary on the demo
  profiles and on the synthetic defects, with the figures, so the user can judge each one at the
  end. Written at the gate's milestone, from real outputs.
