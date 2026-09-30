# The quality-controlled processing loop

PACo processes, judges, picks and inverts a line the way a geophysicist does, on its own: after each stage it checks the result; if it is not good, it changes the parameters of that stage or of an earlier one, and redoes only what is affected, until each xmid passes or is rejected with a reason. The end result is one Vs model per xmid, each traceable to the parameters and QC verdicts that produced it, plus a QC report for the whole line. Each gate's page under `docs/gates/` gives its checks, thresholds, reasons and the demo's outputs.

By default the user gives no parameters at all: "process <profile>" runs the whole loop and ends with Vs models and a report, every parameter derived from the data and the geometry, without a question to the user. Settings the user types in the chat are the exception, and even those are not a reason to ask (see "Agent side").

Scope: active, passive and passive-active lines, the gates one generic mechanism. G2 to G6 run on every mode; on a passive record, G1 runs only the checks that do not need a trigger (dead, clipped and NaN traces, the spectra). G2 also judges a passive window's FK selection and a passive or passive-active window's stacked correlations, and S2 chooses a passive line's segments (`docs/gates/S2_rules.md`).

## Pipeline

Artifacts in [ ], gates in < >.

    [raw record per source position] + geometry
      S1 preprocessing, per record (trace editing, filtering, muting, normalisation...)
    [preprocessed record per source position]
      <G1 signal QC, per record>
      S2 phase shift, per xmid
    [dispersion image per xmid]
      <G2 dispersion image QC, per xmid>
      S3 picking, per xmid (M0 alone: a higher mode is too hazardous to pick without a person's eye,
         a ridge taken for the wrong mode misleading the inversion; a person picks it in PAC)
    [dispersion curve per xmid]
      <G3 curve QC, per xmid>
      <G4 curve profile QC, whole line>
      gate decision: S4 starts for the xmids G4 passed; the rejected ones are reported, never inverted
      S4 MCMC inversion, per xmid (every mode picked, as PAC inverts them by default: M0, and the
         higher modes a person picked in PAC; G5 judges M0's fit)
    [Vs model per xmid: the ensemble, PAC's default view, is the one monitored and reported]
      <G5 model QC, per xmid>
      <G6 model profile QC, whole line>
    [final Vs model per xmid + QC report]

    Only when the user asks for soils or the water table, from the curves G4 passed:
      range check: the curves the chosen Silex model's trained band and velocities cover; the others are reported, never inverted
      S5 petrophysical inversion, per xmid (Silex: soils, N values, water table; santiludo's rock physics)
    [petrophysical model per xmid]
      <G7 petrophysical model QC, per xmid>
      <G8 petrophysical profile QC, whole line>
    [PAC's petrophysical sections, over the xmids G7 and G8 passed]

## Rules for every gate

1. Output per unit (record or xmid): verdict `pass`, `retry` or `reject`; each metric with its threshold; the flags raised; and for each flag, the stage most likely at fault (this one or an earlier one) and a concrete suggested change as overrides that can be applied as they are, not only a sentence.
2. Fixable vs not fixable: a dead geophone, a trigger inconsistent from trace to trace, or a window without enough offset coverage does not improve with parameters. Actions a gate can advise: change parameters of stage k, exclude traces, exclude a record, reject the xmid. `reject` is a legitimate end state, with its reason.
3. Cheapest fix first: re-pick before re-running the phase shift, before re-preprocessing. Going back to stage k invalidates everything downstream, for the affected xmids only; the others keep their results.
4. Bounded: a retry budget per gate and per unit, and a total budget per run, all configurable. When a budget is spent, the unit is rejected with "budget spent" and the last attempt's flags. No loop can run forever. Stop at the first attempt that passes.
5. The judge is fixed during the loop: the loop changes processing, picking and inversion parameters, never QC thresholds; only the user changes thresholds, between runs (`PACO_QC_CONFIG`).
6. A fix must not win by throwing data away. Each gate also reports what was kept: usable band, wavelength range, number of points, traces and records. A retry that passes by shrinking this well below the previous attempt, or below the line's median, is flagged. Example: a mute that sharpens the image but loses the low frequencies.
7. Parameters come from the data and the geometry, not from defaults (see "Coherence rules for S2"). They are checked before any work, like `resolving.py`, with errors written for the agent. PAC's form defaults (3-receiver windows) are not acceptable derived values.
8. Everything is recorded. Each attempt (unit, stage, parameters, metrics, verdict, what triggered it) goes into a QC log in the run folder. The final report gives, per xmid: final parameters, number of attempts, verdict, reasons for any reject, and for each inverted xmid the wavelength range and point count of the curve it used, so that a bad pick can be spotted afterwards in PAC's UI and inverted again. These records are also labelled data for training a picker.
9. Thresholds are provisional: all in one place, configurable, recorded with each verdict, tuned on the demo profiles and a real line, to be recalibrated on hand-judged windows.

## Gates

### G1 signal QC (per preprocessed record, with the raw one when useful)
- Dead, clipped or NaN traces; traces whose RMS is an outlier against the amplitude decay with offset. No reversed-polarity check: near the shot it flags neighbours shifted by more than half a period.
- SNR: energy in the surface-wave window (between the arrivals at vg_max and vg_min) against a noise window: pre-trigger if the record has one, otherwise after the slowest arrival, on every trace when the record is long enough, else on the traces within the line's reach. Report which window was used. Measured in the part of the usable band the dispersion images use (as a filter to it would give it: a filter common to the traces cannot change the image), the noise measured before the muting.
- Usable band: frequencies where the signal spectrum exceeds the noise spectrum by N dB, giving fmin_usable and fmax_usable. This band feeds S2.
- Lateral coherence: median correlation between neighbouring traces inside the surface-wave window, in the same band as the SNR.
- Trigger consistency: a delay the same on every trace is corrected in S1 (the muting's trigger); read from the direct wave (the traces nearest the shot) on the record before its muting, and judged on muted records only. First breaks that make no line (a t0 differing from trace to trace) reject the record.
- On a muted record, the noise, the usable band, the first breaks and the pulse are measured on the record before its muting.
- Actions: exclude traces; exclude the record from the windows that use it; correct a muted record's trigger. No filter or mute: a record under its limits is left out at once, since neither could change what G1 measures. Details: `docs/gates/G1.md`.

### G2 dispersion image QC (per xmid, before picking)
- Coherent energy: the share of frequency columns whose maximum rises 30 % of the way from the noise floor 1/sqrt(N) (the floor the picker uses) to 1.
- Energy maximum on the grid edges: at vmin or vmax, the velocity range is too narrow; at fmin or fmax, the band may go on beyond the image (kept, never widened).
- Competing ridges of similar strength at the same frequency: a higher mode dominating (kept: the fundamental mode is picked as the slowest ridge), or aliasing (compared with the aliasing limit v = 2·dx·f: the band is cut where the alias starts).
- Coherent band much narrower than G1's usable band: reported; the band is capped, never widened.
- A passive or passive-active window's stacked correlations, measured as a record is against G1's limits, and a passive window's fk selection: falling short, the phase shift again with more of the data, else the window rejected.
- Actions: vmin/vmax, fmax at an alias, a passive window's fk threshold, or back to S1 (a surface-wave mute between the expected vg_min and vg_max often cleans the image a lot; asked only of records not muted yet, and done once by PACo at the end of the picking).
- The metrics computed on the picked M0 (sharpness, prominence, on_data, constant_wavelength) belong to G3. Details: `docs/gates/G2.md`.

### G3 curve QC (per xmid)
No one looks at the curves before the inversion: G3 is the safety net for the picking's risks. The pick's metrics (sharpness and prominence against a perfect plane wave for the same window, on_data, constant_wavelength), plus:
- Wavelength range: the picker follows its ridge as far as it holds, down to one spacing, with no cut at the long end; points under 2·dx (the aliasing zone) and over 3L (beyond the window's reach) are flagged and kept, and PAC draws both limits as λmin and λmax.
- Mode jump: a relative velocity step between consecutive points of the resampled curve above 30 %.
- Air wave: near-constant velocity around 330 to 345 m/s: rejected, since no mute parts it from the surface waves.
- Normal dispersion expected (velocity rising with wavelength). An inverse trend is flagged, not rejected, since a stiff layer over a soft one produces it.
- At least 4 points, and a span of wavelengths the inversion can use (the longest over the shortest at least 1.34); the Lorentzian uncertainty reported, not judged: the picker caps each point's at 40 % of its velocity.
- Actions: picking parameters (corridor, coherence rule, max wavelength, mode rule, resampling), or back to S2 or S1, done once by PACo at the end of the picking (`docs/gates/loop.md`). Details: `docs/gates/G3.md`.

### G4 curve profile QC (whole line)
- Build the pseudo-section Vr(λ, xmid) from the passing curves.
- Compare each curve with the median of its neighbours (two a side, within 3 of the line's steps: across a gap, none) at the same wavelengths; a relative misfit above 15 % puts it off that side.
- An isolated outlier, one xmid off neighbours that agree with each other and fitting no side, is suspect. A change shared with one side is geology and is kept.
- Coverage: runs of rejected xmids, and wavelength ranges that vary a lot along the line (depths of investigation not comparable).
- Actions for an outlier: re-pick with the corridor centred on the neighbours' median curve, then G3 again; otherwise reject. Never edit or smooth a curve to make it fit.
- G4's verdict is the go/no-go for S4: its record in the QC log is what `invert` reads. Details: `docs/gates/G4.md`.

### Checks before S4 (inversion parameters coherent with the curve)
- Vs bounds bracket the curve: Vs ≈ 1.09·Vr at PAC's Vp/Vs of 1.77. Wide by default (100 to 2,000 m/s; with the layers given, 100 to 1,000 m/s above the half-space), widened to 0.8·Vr_min and 1.5·Vr_max where the curve needs it.
- Depth: the top of the half-space (the deepest interface) is not resolved deeper than about λmax/2: the data's own layers end there, values given are scaled to it, and G5 shrinks a model given layers reaching deeper to the depth the data inform. Layers thinner than about λmin/3 are not resolved.
- Layers: chosen by the data by default, up to 8 (sigpipe's reversible-jump chains); when given, at least 3, starting at 4, and the loop adds one when the misfit stays high and removes one when it is not resolved.
- Vs may fall at most 20 % from a layer to the next: a stiff layer over a much softer one makes the forward model's fundamental mode a wave trapped in the soft layer.
- The burn-in rule (at least SAVE_EVERY = 150 iterations left to sample). Details: `docs/gates/S4_checks.md`.

### G5 model QC (per xmid)
- Misfit: RMS normalised by the curve's uncertainties, by band of wavelength (about 1 means a fit within errors; much above 1 is underfit; much below 1 is suspicious), from the forward-modelled curve of the monitored model, the ensemble.
- Convergence: the chains agree on the models' Vs at a few depths the curve resolves (split R-hat), with enough effective and saved samples; the acceptance rate is reported, not judged as a limit (the samplers need no step tuned to a band).
- Posterior piled at a prior bound (Vs, a thickness, or the most layers allowed): widen that bound.
- The depth informed (`depth_informed`): from the kept models' relative uncertainty of Vs alone, no prior; below it the data say little. A model given layers reaching much deeper is shrunk to it, and one the data inform at no depth is rejected (`uninformed`); G6 and `job_status` read it too.
- Actions: n_iterations and burn-in (convergence), bounds, the most layers allowed (or, given, the number of layers). A failing xmid can be re-inverted with its own settings, recorded in the log. Details: `docs/gates/G5.md`.

### G6 model profile QC (whole line)
- Vs at fixed depths along the line, each model down to G5's depth informed; not the interface depths.
- An isolated jump that the curves don't show (G4 found the neighbouring curves agree) is non-uniqueness: kept with its flag, since G5 passed the model converged, and sampling longer gives the same posterior. A jump the curves also show is kept.
- The depth informed consistent along the line.
- No lateral smoothing, neither by editing models nor by using neighbours as priors: it would create the smoothness it then reports. Details: `docs/gates/G6.md`.

### Range check, G7 and G8: the petrophysical inversion

Three steps, the range, the fit and the line (`docs/gates/G7.md`, `G8.md`):

- Range: a curve outside the chosen Silex model's trained band and velocities (beyond Silex's own 20 % margin) is not inverted; the agent says how the curves fall outside ("6 end below 43 Hz") and chooses the model covering the most (`petro_models`).
- G7, per xmid: the curve the predicted soil column gives back against the pick, by band, as G5 (at most 2 per band); points with no fundamental mode, and a pick with no uncertainty, reject.
- G8, whole line: the rock physics' Vs at fixed depths (G6's rule) and the water table against the neighbours; an outlier the curves do not show is left out, a change they show is kept.
- No retry: a model gives one soil column per curve. A window G7 or G8 rejects is left out of the sections; its flag says what could change it (another model covering the curve, or the pick).

## Coherence rules for S2 (checked before running)
- Window length: one value for the whole profile, so lateral resolution and depth of investigation stay comparable and xmids don't move; proposed by a ladder of trial windows, the agent's to change (`docs/gates/S2_rules.md`).
- Step: 1 receiver by default (every xmid). Per-xmid retries change only parameters that don't move the xmid: band, velocity range, offsets, records used, mute.
- Frequency range: fmin >= the records' median fmin_usable from G1; fmax <= min(their median fmax_usable, Nyquist), never widened.
- Velocity range: vmin and vmax bracket the expected velocities: G2 widens them per window where a ridge sits on the grid's edge.
- Frequency step: a step finer than 1/T (T = record length after the mute) only interpolates: no padding before the phase shift.
- Offsets: the nearest shot at least half the trial curves' longest wavelength from the window, against near-field effects, where the window has a farther one; the farthest within the reach G1 found.

## Agent side
Constraints: Qwen3 4B or 8B, a 16k context, one tool call per reply, `PACO_MAX_TOOL_CALLS` = 15; the model sees summaries only, and the tool cards have a measured budget.
- The agent asks only when stuck: when the data cannot decide (no image or no curve left for the line, windows rejected once the retry budget is spent, a request the data do not allow, such as windows longer than the line), it asks one short question with 2 or 3 concrete options, its choice first, and waits. Otherwise it applies the gates' fixes and goes back a stage by itself, saying what it changed.
- Settings the user types in the chat are the only exception to "derived": the loop starts from them. If a gate then concludes one must change, the agent changes it and says so in its answer, without asking.
- The go/no-go before the inversion is G4's verdict: `invert` reads G4's record from the QC log, and refuses a run without it. There is no approval step.
- Gates work on a whole run and return one short summary: counts per verdict, flags grouped by runs of xmids (e.g. "xmid 12-18: ridge at vmax"), with the suggested overrides per group. Never one line per xmid.
- Retries target a subset: redo stage k for listed xmids, or for the ones carrying flag X, with overrides.
- Placeholders, not values, in tool descriptions (small models copy example values).
- Raise `PACO_MAX_TOOL_CALLS` only with evaluation evidence.
- Evaluation scenarios (`paco.evaluation`), on the demo profiles, synthetic variants of the demo profiles with known defects, and settings the user types:
  - Looking around: the profiles, one profile's geometry, a profile that does not exist.
  - Zero settings: "process active_p1 and give me the Vs models" with nothing else; the agent must not ask a single question, and the run must end with models for the xmids G4 passed and reasons for the rest.
  - PAC's three modes, and soils and the water table.
  - G1: a dead trace (a copy of active_p1 with a zeroed trace), and the demo's own trigger delay.
  - G2: a velocity range too narrow.
  - G5: too few iterations, and bounds too tight.
  - The window length the agent chooses for depth, and for lateral detail.
  - Stuck: a passive line with no wave (passive_p1's geometry with white noise for records), and windows longer than the line: the agent asks.
  - A setting the user typed that a gate must change: the answer says so.
  - Every scenario where the data decide checks that thresholds were never changed and that no question was asked; `inversion_succeeded` is checked only when G4 passed a curve.
  - An air wave and an isolated G4 outlier have no scenario: neither can be made a physical defect of the demo's records (`docs/gates/loop.md`); the gates' tests cover both on analytic curves and synthetic lines.

## Tests

Each metric on inputs with a known answer (synthetic records with a dead trace, a clipped trace, a known SNR; images with the ridge at vmax; curves that carry exactly the picking risks: points beyond 3 window lengths, the air wave, a jump onto M1; posteriors piled at a bound), mutation checks, and parity with PAC wherever PAC does the same thing. ruff, pyright and pytest pass before a change goes in, then `paco-evaluate --repeat 3` measures the agent.

## Design decisions

Each decision states what the system does and the evidence it rests on; the gates' pages give the details.

### The loop and the agent

- **The search loop: option B.** Where the parameter search lives: (A) the LLM applies each
  suggested change and calls the tools again: most faithful, but many calls and a growing
  context; (B) each stage tool runs its own bounded retry loop with its gate's suggested
  overrides and returns one summary, and the LLM handles backtracking across stages and
  explains what it did; (C) a mix. B: a straight pass already takes 4–8 calls and 2.2–3.4k-token
  prompts with Qwen3-4B, and a play that loops burns all 15 calls and invents a result. The
  earlier stage G2 or G3 blames is done again once by PACo itself, at the end of `pick`
  (`docs/gates/loop.md`).
- **The agent's loop: plan, act, observe, adapt.** With option B the tools run the retries
  within a stage; the agent keeps that loop across stages: it plans the stages, acts by calling
  them, observes each gate's summary, and adapts by backtracking with the suggested overrides.
- **The tools: one per stage, each with its gate's retries, and `redo`.** `run_processing` runs
  S1, G1 and its fixes, the S2 rules, S2 and G2; `pick` runs S3, G3 and G4 (the re-picks
  included), then the earlier stage G2 or G3 blames, once; `invert` runs S4, G5 and G6 as a
  background job, refused unless G4 passed the line; `job_status` follows it; `redo` goes back
  to a stage for some windows (listed, or those carrying a flag) with changes. A straight run is
  4 calls and the polling, and the agent observes after each stage. Not one tool up to G4 (a
  single look before the inversion), nor one job end to end (the agent would only report).
- **Attempts: history inside one run.** The final result keeps PAC's layout at the top of each
  `xmid_<x>/` folder; earlier attempts go under `xmid_<x>/attempts/<n>_<stage>/` with their
  files, parameters, metrics and verdict; the QC log sits at the run level. One window weighs
  2.4 MB.
- **The pick is saved as it is judged.** The picking attempt writes the window's
  `DispersionCurves_0000.csv` (PAC's layout) before G3 judges it; a pick done again archives the
  previous curve and the inversion under `attempts/<n>_picking/`. What G3 and G4 judge is the
  file the inversion reads.
- **G4 runs on the curves that passed G3**, the others are the line's gaps. A window whose G3
  verdict is `retry` is not a curve for the line until its retry passes.
- **Thresholds: locked during the loop.** One configuration, changed by the user between runs,
  recorded with every verdict; the tools take no thresholds. Their values are measured on the
  demo profiles, a real line (`active_p2`) and synthetic defects, all in one configuration file
  (rule 9); each gate's page gives them with their reasons.
- **Budgets:** 2 retries per gate and unit, 2 × xmids per run, and 6 inversion retries per
  window (G5 or a failed run, outside the run's), all configurable. G1's retries draw on each
  record's own budget: a line of 97 shots would otherwise spend the run's budget on records
  before any window.
- **No retry that cannot change the result.** A retry with the parameters of the attempt before
  is refused (`nothing_to_try`). None of: a filter or a mute for G1, a second mode for G2, a
  filter or an air-wave mute for G3 (the air wave is rejected), a mute of records muted already,
  a vmin under 30 m/s, a longer sampling of a converged model for G6 (`non_unique` is kept).
- **An earlier stage G2 or G3 blames is done again once**, by PACo at the end of `pick`, then
  the window is rejected if it still asks (`redone_once`): left to the agent, G3's are not run,
  and the window is a gap, unsaid (`docs/gates/loop.md`).
- **One name, one meaning**: `trigger_error_s`; `neighbour_misfit` for G4, G6 and G8, "misfit"
  alone a fit's (G5, G7); one depth informed, G5's (`depth_informed`), which G6 compares the
  models down to and `job_status` reports (`depth_informed_m`); G5's `min_vs_ratio`, apart from
  the priors' `max_vs_drop`. G4 and G6 compare no neighbours across a gap (3 of the line's steps,
  as G8). A `qc_config.json` saved by an older version still reads: its retired thresholds
  dropped, its renamed ones read under their new names (`config.py`).
- **The agent asks only when stuck** (see "Agent side"). Not before changing a setting the user
  typed (Qwen3 does not keep such a rule), nor before every step back.
- **The host refuses a call identical to one that just failed**, with the failure and "change
  it or answer the user": Qwen3-8B otherwise sends the same wrong `invert` call three times, and
  Qwen3-4B loops ten times and invents a result.
- **What the host guarantees, whatever the model says** (`paco.agent.host`): the settings the
  gates changed are listed after every answer, as the tools gave them; an answer that asks or
  offers is asked again once, unless a tool said the agent is stuck; no inversion starts unless
  the user's message asks for models (invert, inversion, model, Vs, shear). The evaluation
  measures the model and the host together: left to itself, Qwen3-8B reports the gates' changes
  in few answers, closes with a question in 7 of 39 plays and inverts unasked in 6.
- **Synthetic defects in the data where they are physical** (`paco.evaluation.defects`): a copy
  of active_p1 with a zeroed trace (G1), and passive_p1's geometry with white noise for records
  (a line with no curve); the demo's own trigger delay; typed settings for the rest (a velocity
  range too narrow for G2, too few iterations and bounds too tight for G5). Each defect is
  checked to trip its gate before it becomes a scenario.
- **The evaluation runs 8 workers at PAC's effort**, not a lower effort nor a shorter line: a
  play of the zero-settings scenario (the whole demo line inverted) takes about 6 minutes on 8
  workers, about 50 on one.

### The records (S1, G1)

- **A consistent trigger delay is corrected in S1**, the trigger belonging to the muting: the
  records' files state it (SEG-2's `DELAY`; 20 ms on the demo, whose first breaks on the 0.75 m
  trace come at 19–21 ms), and G1 corrects a muted record's trigger when its first breaks put
  the shot more than 10 ms off. A t0 that differs from trace to trace rejects the record.
  Rejecting a delayed record would leave the demo line no window; flagging it without acting
  would leave the mute's windows x/v wrong by the delay.
- **G1 measures what the images are made of, and leaves a record under its limits out at
  once.** The SNR and the coherence in the part of the usable band the dispersion images use,
  as a filter to it would give them (a filter common to the traces cannot change a phase-shift
  image, which divides each trace's spectrum by its own amplitude); the usable band capped to
  the dispersion band (on `active_p2` it reaches 240 to 270 Hz from energy near the shot that no
  image uses). The noise, the first breaks and the pulse before the muting, which a mute cannot
  change. So no filter or mute retry, which would measure the same; the trigger from the direct
  wave, judged on muted records only (`docs/gates/G1.md`). On two runs of `active_p2`, G1 leaves
  out 4 records and corrects 3 triggers (run 20260927-094830-487e: 112, 114 and 121 kept, their
  coherence 0.51 to 0.60 in 0 to 100 Hz; 130 out at 5.9 dB there), and 3 records (run
  20260928-153714-f215: 117, 120, 132 and 144 kept; 133 out, coherence 0.498), where it finds
  57 records about 11 ms off their trigger.
- **The farthest shot a window stacks comes from the data**: `masw.distance_max` where the
  traces' median SNR, by distance from the shot over all records, falls under 2 dB (63 m on
  `active_p2`, 24 m on the demo), unless the user gives one. Not PAC's 100 m (noise beyond
  about 60 m on `active_p2`), nor every shot, nor 6 dB or 3 dB (at 6 dB the demo's end windows
  lose the far shot that gives them a curve).
- **G1 judges a record's SNR and usable band within that reach**: judged over whole records,
  52 of `active_p2`'s 97 records fall under the SNR limit for their far traces, which no window
  stacks.
- **No reversed-polarity check**: at 1.5 m next to the shot it flags whole blocks of traces,
  neighbours shifted by more than half a period. G1's decay fit keeps the traces it excludes,
  which stops a cascade of 21 exclusions around each shot.
- **G1 keeps the near field in its decay fit**: at G1 the longest wavelength is not known, and
  the record's dominant wavelength misses the traces next to the shot such a rule is for.
- **A window leaves out each record's own excluded traces** (the traces excluded in at least
  half of its records, from all), not every trace any record excluded: with 66 shots a window,
  every window of `active_p2` would end empty.

### The images (S2, G2)

- **No padding before the phase shift.** PAC pads every trace (`Pad(n=1000, taper=25)`) before
  the phase shift; PACo does not, in both image pipelines: a frequency step finer than 1/T only
  interpolates. The images have a 0.5 Hz step on the 2 s demo records; on the passive demo, the
  window at xmid 2.88 is bad without the padding, doubtful with it.
- **The window length: the shortest that passes, as precise as the inversion needs.** A ladder
  of short lengths (5, 7, 9 and 11 receivers; PAC's form has 3) is tried on 27
  windows spread over the whole line, its ends included; the ladder climbs while 80 % of them
  pass G3 and the picks gain precision (`docs/gates/S2_rules.md`). The 27 trials follow the
  line within a few % (5 receivers: 63 %, 7: 74 %, 9: 74 %, 11: 85 %, against 65, 68, 77 and
  85 % of the line's windows). The ladder takes about a minute on the demo. Not a long window
  that passes more often on poor images, nor full runs escalated by the gates (whole runs and
  tool calls). The tests' and scenarios' 24-receiver windows are an explicit request.
- **The ladder proposes, the agent decides.** `run_processing` keeps the ladder's length when
  none is given, and returns every length tried (trial windows passing G3, the wavelengths their
  curves reach, the picks' precision, windows on the line), one length past the proposed one
  included, for comparison; the agent runs again with another length when the request needs
  more depth or lateral detail, and says why. A length given, by the user or the agent, is kept
  as it is. Not the agent choosing from the trials on every run (a call more each time), nor
  the agent exploring whole runs freely.
- **The band: capped, never widened by G2.** fmax starts at the preset's (PAC's 100 Hz) and is
  capped at the smaller of the records' median usable fmax and Nyquist; G2's `band_at_fmax` and
  `narrower_than_usable` are kept flags. Widening the band makes G3 worse at every length (24
  receivers: 17 of 19 pass at 100 Hz, 15 at 150 Hz, 6 at 324 Hz, G1's usable band): above
  100 Hz a second ridge competes and the picker jumps. The median record's band, not the worst
  record's: 2.3 to 298 Hz on `active_p2`, against 24 to 86 Hz, which would cut the line's long
  wavelengths.
- **G2's `band_at_fmin` kept, not retried**, like `band_at_fmax`: lowering fmin to 1 Hz never
  moves the band's low end, and spends the budget on 68 of 92 windows of `active_p2`; the
  picker decides where the low end stops.
- **G2 counts a peak at the grid's top velocity only where the window resolves that velocity**
  from an infinite one (f x aperture above vmax).
- **The near field: the nearest shot a window stacks.** On an active line, a window stacks no
  shot nearer to its nearest receiver than half the longest wavelength the trial curves
  reached, where it has a farther one; a window with near shots only keeps them, and G3
  reports the window's nearest shot against half its longest wavelength (`near_offset`, a kept
  `near_field` flag). Leaving out the near shots of every window, the end windows included,
  costs a third of the demo's curves (17 to 11 of 19 at 24 receivers).
- **One SNR limit for every signal.** A passive or passive-active window's stacked correlations
  are measured as a record is, from their virtual source, against G1's 6 dB, each lag scaled for
  the samples it sums (noise alone reads 4 to 5 dB unscaled, about 2 dB scaled); a muted line's
  on the correlations of its records before their muting (a mute zeroes their noise: some
  300 dB whatever the data). Falling short, the phase shift again with more of the data, then
  the window rejected (`docs/gates/G2.md`).
- **PAC's passive-active mode** (interferometry on an active profile's shots), with a fix of
  PAC's chain: the flipped gathers' geometry. A shot is correlated whole: the muting's
  velocities cut what a surface-wave window of the chain's own would.
- **The passive defaults**, PAC's and PACo's: 1 s segments end to end, the FK selection always
  on (0.2), whitened and normalized one-bit, the correlations stacked phase-weighted (power 2);
  S2 then tries other segment lengths and FK thresholds from them (`docs/gates/S2_rules.md`).

### The picks (S3, G3, G4)

- **The pick goes as far as its ridge holds, at both ends.** No cut at 2 window lengths; the
  picker searches down to one spacing (`min_wavelength` 1). The pick is the longest continuous
  run of kept points (a step steeper than |d ln v / d ln f| = 2, or more than 2 Hz of columns
  too weak to keep, ends it), then its ridge followed on past both ends, a lower maximum taking
  its place only when brighter, since a dimmer one under it is its sidelobe or alias (on 9 m
  windows of `active_p2`, faint lines at 1 to 4 spacings reach 35 % of their column's maximum
  under 25 Hz, and the backward waves' aliases, at 1 to 2 spacings, above 50 Hz). Without a
  bridged gap, the picks end at 18 to 22 Hz on the demo. On `active_p2` (90 windows of 7
  receivers) the high end reaches the grid's top, 100 Hz (lowest 74), against 68.7 Hz (lowest
  58) with a search stopping at 2 spacings; 269 tracked points kept (median), against 169.
  Followed from its widest stretch instead, the pick of the demo's first 24-receiver window is
  the branch above the 50 Hz mains line.
- **Where the window does not resolve, flagged and kept**: points under 2 spacings
  (`aliasing_zone`) and over 3 window lengths (`beyond_reach`: on 83 of those 90 windows, a
  median 63 % of the curve's points, where the ridge climbs out of the grid under 15 Hz). PAC's
  λmin and λmax are where the flags start: 2 spacings and 3 window lengths, on PAC's image and
  sigpipe's figures.
- **PACo's rules read the whole curve**, its points over 3 window lengths included: the
  near-field distance (half the trial curves' longest wavelength), the inversion's depth range,
  the ladder's wavelengths and precision, G4's and G6's spreads. On the demo's ladder (24
  receivers, 5.75 m): 3 trial windows of 3 pass, wavelengths 5-34 m, picks within 40 %, a near
  field of 17 m, where 2 of the 4 windows keep near shots only.
- **A point is kept only where the window resolves velocity at all**: a perfect plane wave for
  the window, at the pick's frequency and velocity, must stand at least 1 % above its column's
  median. Without that floor, some picks drift smoothly to 5 Hz (160 m wavelengths on 3.75 m
  windows), 0.5 Hz after a G4 re-pick. At 1 % the lowest points are 8.5 to 10 Hz on every
  length.
- **G3 judges sharpness and prominence against a perfect plane wave for the same window.** On
  the demo both equal a plane wave's (ratio 1.00) at every length from 5 to 24 receivers, where
  fixed limits fail every 5- to 11-receiver window: they measure the array, not the data.
  Prominence must reach half a plane wave's, counted up to 4 (beyond, any ridge stands out: a
  fixed limit of 2 is half of it); sharpness 0.8 of its width.
- **G3's curve rules and G4's thresholds** (`docs/gates/G3.md`, `G4.md`): a mode jump at 30 %
  between consecutive points of the resampled curve (the good demo curves step by up to 26 %,
  the two bad ones by 31 and 53 %), its first fix a band cut where the jump sits, the corridor
  halved second; the air wave at 330–345 m/s over half of the points, rejected; an inverse trend
  by the rank correlation of velocity with wavelength, kept, never rejected (11 of 19 demo
  windows with the picks cut at 2 window lengths), and said once over the line by G4; at least 4
  points; the uncertainty reported only. G4 compares each curve with two neighbours a side: an
  outlier is off neighbours that agree with each other and fits no side (misfit 15 %; the
  demo's neighbours agree within 2 to 13 %); a curve off one side and fitting the other is a
  shared change, geology, kept. An isolated outlier is one xmid, not two: two changed windows
  make their neighbours disagree, so nothing in a run of two can be told from geology by its
  neighbours. A side of one curve is no evidence: it agrees with itself (three curves 6 m apart
  would call xmid 8.88 an outlier).

### The models (S4, G5, G6)

- **The inversion's bounds come from each window's curve** (`docs/gates/S4_checks.md`): the
  data's own layers between a third of the shortest wavelength and half the longest, Vs 100 to
  2,000 m/s widened where the curve needs it; the layers given start wide (Vs 100 to 1,000 m/s,
  the half-space to 2,000, layers 1 to 10 m thick), and the loop narrows them. Values the user
  types are checked against the curve and changed with a message. With PAC's bounds (Vs 100 to
  1,000 m/s, thicknesses 1 to 10 m) and 2 layers, the demo's dense line puts the interface at
  6 to 7.5 m, deeper than the curves resolve, and leaves the half-space's Vs to the prior (the
  chains' medians 5 to 97 % apart); within half the longest wavelength, the interface sits at
  1.9 to 3.9 m and the chains agree within 2 to 11 %. Not one prior for the whole line, nor
  PAC's bounds merely checked against the curve.
- **Layers: chosen by the data by default**, up to 8 (sigpipe's reversible-jump chains); given,
  at least 3, starting at 4 (the layer study, `docs/gates/G5.md`): every count fits within the
  errors, more layers are slower and converge less often. G5 adds layers up to what each curve
  resolves and removes one that piles at its thinnest, never below 3.
- **G5 judges the ensemble, PAC's default view, and keeps what only the median across the
  models loses.** The fit between the monitored model's forward-modelled curve and the picked
  one is a QC in its own right, reported by band, not as one number, as PAC's pseudo-section
  comparison shows it (PAC saves the forward-modelled curves beside the models). When the
  layered median fits and only the ensemble does not, G5 keeps the window with a flag
  (`smoothing_misfit`: re-inverting cannot fix the median's blurring of the interfaces); when
  both misfit, it retries. Not judging the layered median only, nor a retry on any misfit of the
  monitored model. PAC's residual is reported by band, not judged.
- **`job_status` reports the monitored model** as Vs at a few fixed depths and the depth
  informed (`depth_informed_m`), plus each window's curve misfit.
- **G5's and G6's thresholds** (`docs/gates/G5.md`, `G6.md`): misfit at most 2 per band, split
  R-hat 1.1, 100 models a chain, 200 effective samples, 10 % of samples at a bound, up to 10
  layers; models 15 % off agreeing neighbours, depths informed spread at most 50 %. Not an
  at-bound limit of 6 %: it would flag the demo's inverse windows, whose half-space presses
  against its lowest Vs, and reject them after two widenings although they fit. A non-unique
  model is kept with its flag: its posterior converged, and sampling longer gives it again.
- **G5's `no_mode` suggests the phase shift again with the band below the points no mode
  reaches**: the window stays rejected, the change the agent's to make with `redo`.
- **G6 and `job_status` read G5's depth informed**, from the models' Vs spread (a median of
  7.5 m on `active_p2`), G6 within the curve's depth of investigation.

### The documentation

- **Each gate has a page under `docs/gates/`**: what it checks, each metric with its threshold
  and why, the tool's real summary on the demo profiles and on synthetic defects, with the
  figures, so that each gate can be judged.
