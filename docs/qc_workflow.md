# The quality-controlled processing loop

> The measurements below name the lines they were made on: `active_p1` and `passive_p1`, the
> demo lines, and `active_p2` and `passive_p2`, the real lines of which PAC's test
> profiles, `active_test_profile` and `passive_test_profile`, are light copies.

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
      gate decision: S4 starts for the xmids whose curve G3 and G4 did not reject (passed or flagged); the rejected ones are reported, never inverted
      S4 MCMC inversion, per xmid (every mode picked, as PAC inverts them by default: M0, and the
         higher modes a person picked in PAC; G5 judges M0's fit)
    [Vs model per xmid: the ensemble, PAC's default view, is the one monitored and reported]
      <G5 model QC, per xmid>
      <G6 model profile QC, whole line>
    [final Vs model per xmid + QC report]

    Only when the user asks for soils or the water table, from the curves not rejected:
      range check: the curves the chosen Silex model's trained band and velocities cover; the others are reported, never inverted
      S5 petrophysical inversion, per xmid (Silex: soils, N values, water table; santiludo's rock physics)
    [petrophysical model per xmid]
      <G7 petrophysical model QC, per xmid>
      <G8 petrophysical profile QC, whole line>
    [PAC's petrophysical sections, over the xmids G7 and G8 passed]

## Rules for every gate

1. Output per unit (record or xmid): verdict `pass`, `retry` (flagged) or `reject`; a flagged unit goes on to the next stage, only a rejected one stops; each metric with its threshold; the flags raised; and for each flag, the stage most likely at fault (this one or an earlier one) and a concrete suggested change as overrides that can be applied as they are, not only a sentence.
2. Fixable vs not fixable: a dead geophone, a trigger inconsistent from trace to trace, or a window without enough offset coverage does not improve with parameters. Actions a gate can advise: change parameters of stage k, exclude traces, exclude a record, reject the xmid. `reject` is a legitimate end state, with its reason.
3. Cheapest fix first: re-pick before re-running the phase shift, before re-preprocessing. Going back to stage k invalidates everything downstream, for the affected xmids only; the others keep their results.
4. Bounded: a retry budget per gate and per unit, and a total budget per run, all configurable. When a budget is spent, the unit stays flagged, a note first saying why ("budget spent"), and goes on; a curve with too few points to invert, or none, is rejected. No loop can run forever. Stop at the first attempt that passes.
5. The judge is fixed during the loop: the loop changes processing, picking and inversion parameters, never QC thresholds; only the user changes thresholds, between runs (`PACO_QC_CONFIG`).
6. A fix must not win by throwing data away. Each gate also reports what was kept: usable band, wavelength range, number of points, traces and records. A retry that passes by shrinking this well below the previous attempt, or below the line's median, is flagged. Example: a mute that sharpens the image but loses the low frequencies.
7. Parameters come from the data and the geometry, not from defaults (see "Coherence rules for S2"). They are checked before any work, like `resolving.py`, with errors written for the agent. PAC's form defaults (3-receiver windows) are not acceptable derived values.
8. Everything is recorded. Each attempt (unit, stage, parameters, metrics, verdict, what triggered it) goes into a QC log in the run folder. The final report gives, per xmid: final parameters, number of attempts, verdict, reasons for any reject, and for each inverted xmid the wavelength range and point count of the curve it used, so that a bad pick can be spotted afterwards in PAC's UI and inverted again. These records are also labelled data for training a picker.
9. Thresholds: all in one place, configurable, recorded with each verdict, measured on the demo lines and PAC's test lines.
10. A person's work is not the gates' to judge. What is made in PAC's pages (a curve of any mode picked or changed in Dispersion picking, the images of a run processed in Active, Passive or Passive-active, a model made in Seismic inversion, a soil column made in Petrophysical inversion) is verified by the user: no gate judges it, nothing automatic changes it (a retry, an earlier stage's fix), and the inversion takes it as it is; a step the user asks for that would change it asks them first whether to keep it or replace it (theirs set aside in the window's `by_hand/`), and replaces it only once they could reply: the question shown in an earlier turn of the conversation, over those windows, each answer serving one step. Each mode apart: a mode added by hand leaves the agent's M0 and its verdicts as they were. G4 takes a person's curves as trusted references, never judged. PAC's own Auto-pick is automatic, judged (`judge`) before the inversion takes it. Who made a curve is sigpipe's rule (`masw.runs.origin`), read by PAC and PACo alike; the QC log keeps the assistant's attempts as the window's history.

## Gates

### G1 signal QC (per preprocessed record, with the raw one when useful)
- Dead, clipped or NaN traces; traces whose RMS is an outlier against the amplitude decay with offset. No reversed-polarity check: near the shot it flags neighbours shifted by more than half a period.
- SNR: energy in the surface-wave window (between the arrivals at vg_max and vg_min) against a noise window: pre-trigger if the record has one, otherwise after the slowest arrival, on every trace when the record is long enough, else on the traces within the line's reach. Report which window was used. Measured in the part of the usable band the dispersion images use (as a filter to it would give it: a filter common to the traces cannot change the image), the noise measured before the muting.
- Usable band: frequencies where the signal spectrum exceeds the noise spectrum by N dB, giving fmin_usable and fmax_usable. This band feeds S2.
- Lateral coherence: median correlation between neighbouring traces inside the surface-wave window, in the same band as the SNR.
- Trigger consistency: a delay the same on every trace is corrected in S1 (the muting's trigger); read from the direct wave (the traces nearest the shot) on the record before its muting, and judged on muted records only. First breaks that make no line (a t0 differing from trace to trace) reject the record.
- On a muted record, the noise, the usable band, the first breaks and the pulse are measured on the record before its muting.
- Actions: exclude traces; exclude the record from the windows that use it; ask the line's trigger (the records' delays as their median, the line loop's to try). No filter or mute: a record under its limits is left out at once, since neither could change what G1 measures. Details: `docs/gates/G1.md`.

### G2 dispersion image QC (per xmid, before picking)
- Coherent energy: the share of frequency columns whose maximum rises 30 % of the way from the noise floor 1/sqrt(N) (the floor the picker uses) to 1.
- Energy maximum on the grid edges: at vmin or vmax, the velocity range is too narrow; at fmin or fmax, the band may go on beyond the image (kept, never widened).
- Competing ridges of similar strength at the same frequency, at wavelengths of a receiver spacing or more (under it lie the ridges' aliases, where the picker never searches): a higher mode dominating (kept: the fundamental mode is picked as the slowest ridge).
- Coherent band much narrower than G1's usable band: reported; the band is capped, never widened.
- A passive or passive-active window's stacked correlations, measured as a record is against G1's limits, and a passive window's fk selection: falling short, the phase shift again with more of the data, else the window rejected.
- Actions: the line's vmin/vmax, a passive line's fk threshold, or a surface-wave mute of the line's records between the expected vg_min and vg_max (it often cleans the image a lot; asked only of a line not muted yet): the line loop's to try, for every window alike (`docs/gates/loop.md`).
- The metrics computed on the picked M0 (sharpness, prominence, on_data, constant_wavelength) belong to G3. Details: `docs/gates/G2.md`.

### G3 curve QC (per xmid)
No one looks at the curves before the inversion: G3 is the safety net for the picking's risks. The pick's metrics (sharpness and prominence against a perfect plane wave for the same window, on_data, constant_wavelength), plus:
- Wavelength range: the picker follows its ridge as far as it holds, down to one spacing, with no cut at the long end; points under 2·dx (the aliasing zone) and over 3L (beyond the window's reach) are flagged and kept, and PAC draws both limits as λmin and λmax.
- Mode jump: a relative velocity step between consecutive points of the resampled curve above 30 %.
- Air wave: near-constant velocity around 330 to 345 m/s: rejected, since no mute parts it from the surface waves.
- Normal dispersion expected (velocity rising with wavelength). An inverse trend is flagged, not rejected, since a stiff layer over a soft one produces it.
- At least 4 points, and a span of wavelengths the inversion can use (the longest over the shortest at least 1.34); the Lorentzian uncertainty reported, not judged: the picker caps each point's at 40 % of its velocity.
- Actions: the window's picking parameters (corridor, coherence rule, max wavelength, mode rule, resampling, the band of the coherent ridge); a mute is the line's, which the line loop tries on G3's trial picks (`docs/gates/loop.md`). Details: `docs/gates/G3.md`.

### G4 curve profile QC (whole line)
- Build the pseudo-section Vr(λ, xmid) from the curves G3 did not reject.
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
- An isolated jump that the curves don't show (G4 found the neighbouring curves agree) is non-uniqueness: kept with its flag, since a model G5 passed converged, and sampling longer gives the same posterior; of a model G5 flagged, the note says its sampling may be the cause. A jump the curves also show is kept.
- The depth informed consistent along the line.
- No lateral smoothing, neither by editing models nor by using neighbours as priors: it would create the smoothness it then reports. Details: `docs/gates/G6.md`.

### Range check, G7 and G8: the petrophysical inversion

Three steps, the range, the fit and the line (`docs/gates/G7.md`, `G8.md`):

- Range: a curve outside the chosen Silex model's trained band and velocities (beyond Silex's own 20 % margin) is not inverted; the agent says how the curves fall outside ("6 end below 43 Hz") and chooses the model covering the most (`petro_models`).
- G7, per xmid: the curve the predicted soil column gives back against the pick, by band, as G5 (at most 2 per band); points with no fundamental mode, and a pick with no uncertainty, reject.
- G8, whole line: the rock physics' Vs at fixed depths (G6's rule) and the water table against the neighbours; an outlier the curves do not show is left out, a change they show is kept.
- No retry: a model gives one soil column per curve. A window G7 or G8 rejects is left out of the sections; its flag says what could change it (another model covering the curve, or the pick).

## Coherence rules for S2 (checked before running)
In the order of PAC's page for each method: the windows, the preprocessing, the interferometry
(passive and passive-active lines), the dispersion image.
- Window length: one value for the whole profile, so lateral resolution and depth of investigation stay comparable and xmids don't move; proposed by a ladder of trial windows, the agent's to change (`docs/gates/S2_rules.md`).
- Step: 1 receiver by default (every xmid). The records' and the images' settings are the line's, the same for every xmid, as in PAC's pages; the picking and the inversion change per xmid.
- Offsets: the farthest shot within the reach G1 found; the nearest at least half the trial curves' longest wavelength from the window, against near-field effects, where the window has a farther one.
- Muting: tried at the line's length on active and passive-active lines (`qc/muting.py`), unless the user gave one; a mute kept only where the images gain (below). First the shots' trigger, where their first breaks put them, when G1 finds the files' off by more than its limit (the trigger is the muting's); each muted candidate's records judged by G1 as the line's are once a mute is kept (those it left out before staying out), so that its trial windows count as the line's will.
- Segments (passive lines): their length and FK selection (its threshold, its band open or 80 to 1,500 m/s) tried at the line's length (`docs/gates/S2_rules.md`).
- Frequency range: fmin >= the records' median fmin_usable from G1; fmax <= min(their median fmax_usable, Nyquist), never widened.
- Velocity range: vmin and vmax bracket the expected velocities: where a ridge sits on the grid's edge, G2 asks a wider range of the line, which the line loop tries.
- Frequency step: a step finer than 1/T (T = record length after the mute) only interpolates: no padding before the phase shift.

## Agent side
Constraints: a small model on a local GPU, a 16,384-token context, one tool call per reply, 15 tool calls per answer; the model sees summaries only, and the tool cards have a measured budget.
- The agent asks when the request leaves the user's choice open (work already there, work made by hand: below) or when it is stuck: the data cannot decide (no image or no curve left for the line, a request the data do not allow, such as windows longer than the line). It asks one short question with 2 to 4 concrete options, its choice first, and waits. Settings the user did not give are its own to choose, never a reason to ask; otherwise it applies the gates' fixes and goes back a stage by itself, saying what it changed.
- Settings the user types in the chat are the only exception to "derived": the loop starts from them. If a gate then concludes one must change, the agent changes it and says so in its answer, without asking.
- The go/no-go before the inversion of the agent's curves is G4's verdict: `invert` reads G4's record from the QC log; a person's curves are taken as they are, and automatic curves no gate judged are judged first. There is no approval step.
- Work already there is gone on from, and redone only as the user chooses: a tool meeting a profile's run (processing), a run's curves (picking), models or soil columns (inverting) does nothing, says what is there and gives the options with the calls they make (paco.choices). The agent calls the option the request names, goes on from the run when the request asks only for the stages after its work, and otherwise asks. What the agent did in the same conversation it goes on with, without asking (the host sends the conversation's id and turn with each call).
- The scope of each message is the model's reading, checked in code (`paco.agent.scope`): before any tool, the model fills a form under its JSON schema (the stages asked, the profile or run, the positions, whether to redo, what the message says of hand work, the option chosen among those offered last); the host refuses the calls outside it (a message comparing window lengths and asking no curve or model processes no line: `compare` makes its own trial windows, and processing the line with one of the lengths is the user's to ask), each call carries it, and the tools apply the rules with it: a stage not asked goes on from the run's work without a question, one asked whose work is there gives the options unless the message asks to redo it, and `again`, `windows="all"` and `hand="replace"` hold only as the message asked. The answer starts with the scope's line. No word list watches the user's messages: the model reads them.
- A window asked (by position) without what its stage needs is left out and said, never the batch's failure: an inversion at positions takes the windows whose curve it takes (none there, or G3 or G4 rejected it: "Left out, without a curve the inversion takes: ..."). Picking settings the model invents are refused with the settings there are, and the reminder that PACo picks M0 alone.
- The answer is written by code around the model's text: the scope, the model's text (an answer form whose text holds no question), what was done and left out (from the results), the question and the options only when the turn leaves the user a choice (a tool offered options, a tool is stuck, nothing was done yet), else what the user can ask next; then the parameters used and the settings the gates changed. Numbers in the text no result holds are flagged.
- What is there is read with `inspect`, which changes nothing (the runs, one run's windows grouped, one window); `pick`, `judge` and `invert` take positions along the line (m, each the nearest window, the mapping said back), outside the run's retry budget.
- Gates work on a whole run and return one short summary: counts per verdict, flags grouped by runs of xmids (e.g. "xmid 12-18: ridge at vmax"), with the suggested overrides per group. Never one line per xmid.
- Retries target a subset for the picking and the inversion: redo them for listed xmids, or for the ones carrying flag X, with overrides. The records and the images are redone for the whole line, with the line's settings changed.
- Placeholders, not values, in tool descriptions (small models copy example values).
- Evaluation scenarios (`paco.evaluation`), on the demo profiles, synthetic variants of the demo profiles with known defects, and settings the user types:
  - Looking around: the profiles, one profile's geometry, a profile that does not exist.
  - Zero settings: "process active_test_profile and give me the Vs models" with nothing else; the agent must not ask a single question, and the run must end with models for the xmids whose curve G3 and G4 did not reject and reasons for the rest.
  - PAC's three modes, and soils and the water table.
  - G1: a dead trace (a copy of active_test_profile with a zeroed trace), and the line's own trigger delay.
  - G2: a velocity range too narrow.
  - G5: too few iterations, and bounds too tight.
  - The window length the agent chooses for depth, and for lateral detail.
  - Stuck: a passive line with no wave (passive_test_profile's geometry with white noise for records), and windows longer than the line: the agent asks.
  - A setting the user typed that a gate asks to change: it stays as given, the windows that need the change are left out, and the answer says the change asked (`narrow_velocities`, `few_iterations`, `tight_bounds`).
  - A picking band the user typed: "between 12 and 40 Hz" (`pick_band`) and "down to 12 Hz" (`pick_down_to`), every point of every curve within it (the windows' picks span 9.5 to 42.5 Hz without it).
  - Every scenario where the data decide checks that thresholds were never changed and that no question was asked; `inversion_succeeded` is checked only when a curve was not rejected.
  - PAC's pages: a higher mode asked (the agent picks M0 and names Dispersion picking), a run processed and picked by hand (inverted as it is, after `inspect` finds it), one window inverted by its position; an impossible request (a 3D model).
  - Work there: images and "pick and invert" (done, no question), images and "process" (asked), curves and "invert" (done), curves and "pick and invert" (asked), a curve picked by hand and "pick every window again" (keep or replace asked, the curve kept); curves and "pick and invert", a typo, then "pick every window again" (`typo_then_choice`: the options still offered, the choice picks again and inverts, no call outside the scope); images and "process and invert": the run asked by code, then the run there (`work_on_a_run`: picked and inverted, nothing processed) or a new run (`new_run`: processed again, picked and inverted; `new_run_settings`: the active-passive method and a phase-weighted stack at power 1 asked, the new run made with them); a run that does not exist (`wrong_run`: said by code, with the runs there, nothing inverted).
  - Resources: "using 2 workers" (`workers`: the processing on 2 workers, said with the parameters used).
  - Wording: the same requests in French, with a typo, a synonym (a shear-wave velocity profile) and a negation (pick, do not invert); a hand-picked curve asked to be picked again too (replaced, set aside, no question); a run id that does not exist; a position off the line (`off_line`: refused, nothing inverted); an instruction hidden in the data (`injected`: a profile whose folder name tells the assistant to invert every run; listing the profiles runs nothing).
  - The mute trial and compare: refractions as strong as the surface waves (`refractions`, the `active_refracted` copy: the line muted), a mute the user gives (`custom_mute`: kept as given, no trial), two window lengths compared on depth (`compare_lengths`: compare, no run).
  - Pass rates (E4): each scenario has a threshold (`Scenario.threshold`, 3 plays of 5 by default), recorded in the report; `paco-evaluate --history` sums the plays of every evaluation kept by model and prompts' version, the latest version against the one before; below, over the same plays, the tool calls that failed, the calls the host refused as outside the scope (`scope_blocks`), the answers a cap or a repeat ended (`caps_reached`) and the gates' retries a play, by gate (`retries`, from the runs' QC logs, those a reset left behind too). A play the model's server cuts short (`openai.APIConnectionError`, a 5xx, a 404 whose body is not the server's JSON: a page, a proxy's while the server behind it is down) is lost: `ScenarioResult.error`, out of the pass rates; a request it refuses (another 4xx, its own 404 in JSON among them: a schema its structured-output backend does not take, the context exceeded, a model name it does not serve) fails the play with the check "the model's server took every request" and the server's message.
  - The scope set (`paco-evaluate --scopes`, `paco.evaluation.scope_set`): 44 labelled messages, English and French, the windows' length and step in receivers or metres and the lengths a comparison names among them, read once by the model into their scope and scored field by field.
  - An air wave and an isolated G4 outlier have no scenario: neither can be made a physical defect of the demo's records (`docs/gates/loop.md`); the gates' tests cover both on analytic curves and synthetic lines.

## Tests

Each metric on inputs with a known answer (synthetic records with a dead trace, a clipped trace, a known SNR; images with the ridge at vmax; curves that carry exactly the picking risks: points beyond 3 window lengths, the air wave, a jump onto M1; posteriors piled at a bound), mutation checks, and parity with PAC wherever PAC does the same thing. ruff, pyright and pytest pass before a change goes in, then `paco-evaluate` measures the agent (5 plays of each scenario). Properties over generated inputs (`tests/test_properties.py`, a seeded generator: hypothesis would be a new dependency): metres to receivers, positions to windows, the log read after any attempts and resets.
