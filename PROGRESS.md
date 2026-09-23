# PACo progress

Last updated 2026-09-23. Read this first at the start of each session.

## Working agreement

- Started in tutor mode, from the rules pasted in the first session (not saved as `CLAUDE.md`).
- Since milestone 4 (2026-09-23): Claude builds, runs and checks each step itself, then explains
  it with the real output; it never asks the user to run anything. Real design decisions still go
  to the user as options, and Claude says afterwards which one it would have picked. No check
  questions. The user commits.
- Repos: sigpipe in `../sigpipe` (GitHub HEAD 403fea2), PAC in `../PAC` (372357c, plus one
  uncommitted README change). PACo installs sigpipe from git; `uv.lock` pins 403fea2.

## Status

| Milestone | Status |
|---|---|
| 0. Orientation | Done, with gaps (see open issues) |
| 1. Plain functions | Done: `inspect_profile`, `run_processing` |
| 2. Schemas | Done: generated presets, messages for the agent, checks before any work, schema budget; 173 tests |
| 3. Quality metric and picking | Done: picker, metric, `dispersion_quality`, `pick`; 250 tests |
| 4. MCP server | Done: 7 tools over Streamable HTTP, errors the model reads, progress, instructions; 267 tests |
| 5. Background jobs | Done: PAC's inversion ported, run as a background job after the user's approval; 294 tests |
| 6 to 8 | Not started |

Check questions: milestone 1 asked, answer pending. Milestone 2 (what happens when sigpipe renames
a parameter): answered with hints on 2026-09-23, up to the plain-words rung. Worth revisiting:
why generated models fail at import while hand-written ones fail in every window. Milestone 3
(why `pick` reads its parameters and verdicts from `quality.json`): answered on 2026-09-23 after
the concept hint. The user's point: judging a curve means seeing it, which a text-only LLM
cannot. Completed: so the verdict is the agent's only view of the curve, and it must describe
the exact curve saved. Milestone 4: none, at the user's request.

## Milestone 0: orientation

- **How PAC builds pipelines:** `load_acquisition` builds the acquisition; `ComputingConfig`
  has one class per mode, with stage unions on `method`; `build_windows` cuts the line into
  windows; `PIPELINE_BUILDERS[mode]` builds one pipeline per window, run in a process pool;
  results go to `data/output/<profile>/xmid_<x>/`.
- **Where PAC's defaults live:**
  - form values in the React forms;
  - derived values, also computed in the forms: `tmax` is the record length, `fmax` is Nyquist;
  - the fixed steps, hard-coded in the adapters.
  - PAC's Pydantic models hold constraints only.
- **sigpipe's registries:** dicts mapping method names to functions, in
  `algorithms/<stage>/registry.py`, re-exported by `sigpipe.algorithms`.
  - Transformers take `method` plus `**params` and pass them on unchecked, at transform time.
  - `Pick` and `Stack` choose their registry from the input type.
  - Lazy-import wrappers hide the signatures of beamforming, silex and petro forward modelling.
- **Found:** PAC's default filter `fmax`, equal to Nyquist, always fails sigpipe's IIR check once
  filtering is switched on.

## Milestone 1: plain functions

Code in `src/paco/`, one sub-package per feature:

- `settings.py`: pydantic-settings, prefix `PACO_`. Settings: `input_dir` (default
  `/data/input`), `output_dir` (`data/output`), `workers` (1).
- `profiles/`:
  - `list_profiles` and `load_profile`. Every file that is not `.yaml` or `.json` is a record,
    read with sigpipe's `Load`. Positions become sigpipe `Coordinate`s with y = 0.
  - `inspect_profile` returns a `ProfileSummary` (about 280 characters).
  - `ProfileError` messages are written for the agent.
- `windows/`: a port of PAC's `build_windows`, with `xmid` from sigpipe's `LinearAcquisition`.
  `MASWParameters` holds PAC's form defaults, except `distance_max`, set to 1000 m.
- `presets/`: see milestone 2.
- `pipelines/`: a port of PAC's adapters. `build_pipeline(preset, window, output_folder)`
  refuses presets that were not resolved.
- `runs/`: `run_processing(profile, preset, overrides, settings, on_progress)` returns a
  `RunSummary` and writes `<output_dir>/<profile>/<run_id>/`: PAC's `xmid_<x>/` folders, plus a
  `run.json` manifest.

Checks against PAC:

- **Windows:** identical in 12 cases. The reference is `tests/data/pac_windows.json`.
- **Presets:** resolved presets validate as PAC configs.
- **Pipelines:** identical steps, and bit-identical outputs on demo windows, even though PAC's
  lockfile pins sigpipe 31274d8 and PACo's pins 403fea2. Re-checked after milestone 2.
- **Tests:** each feature was also checked with deliberately broken copies of the code.

## Milestone 2: schemas

All in `src/paco/presets/`:

- **Generated models** (`stages.py`, `generation.py`, `models.py`):
  - method names, parameter names and types are read from sigpipe's functions
    (`inspect.signature`, `typing.get_type_hints`);
  - `stages.py` holds PAC's choices: exposed methods (only PAC's), PAC's defaults, bounds, units,
    parameters the pipeline sets itself;
  - drift fails at import: a parameter or method sigpipe renames or drops raises with
    "update paco/presets/stages.py".
- **Messages for the agent** (`explaining.py`): `make_preset` raises a `PresetError` with one line
  per problem, what is allowed, and "did you mean" for typos; a stage of the other preset says so.
- **Checks before any work** (`resolving.py`): the profile-dependent rules sigpipe checks in every
  window (segment length and step, whitening band against taper and frequency step, dispersion
  `fmin` below Nyquist, filter `fmax` below Nyquist), all listed at once.
- **The schema the agent reads** (`schemas.py`): `override_schema(name)`, the preset's JSON Schema
  without titles and `mode`; derived values say "null: from the profile". About 2,900 characters
  (active) and 6,500 (passive), under a tested budget of 3,000 and 6,600.

## Milestone 3: picking and quality

- `picking/`: `pick_modes(image, parameters)` returns each mode's points with their diagnostics
  (coherence, pinned, kept) and the kept points as a sigpipe `DispersionCurve` (`M<n>`, Lorentzian
  uncertainties, resampled over wavelength).
- `quality/`: `measure_quality(image, m0)` for one image; `dispersion_quality(run_id, settings)`
  picks and measures every window of a run, writes `quality.json` per window and per run (with
  the parameters used), and returns a `QualitySummary` (counts, stretches of good windows, flags,
  advice for the three most frequent flags).
- `picks/`: `pick(run_id, settings)` saves the M0 curve of every good window in PAC's layout
  (`DispersionCurves_0000.csv`, replacing the window's M0 curve and keeping other labels), redraws
  `DispersionImage_0000.png` with the curves as PAC does, writes `pick.json` and returns a
  `PickSummary` (windows picked, skipped, and whose M0 was replaced). It uses the verdicts and
  picking parameters of the run's `quality.json`, so the saved curves are the judged picks; it
  refuses a run not yet assessed, or without a good window.
- `runs/finding.py`: `find_run`, `load_manifest` and `load_image`. Only names shaped like run IDs
  are looked up, so an ID cannot reach outside the output directory; an unknown ID lists the five
  latest runs.
- Tests:
  - synthetic shots with known dispersion curves, through sigpipe's own `phase_shift`;
  - analytic Gaussian ridges, so each measurement has an exact expected value;
  - real runs of both demo profiles;
  - a mutation check (42 deliberate bugs, all caught).
- On the 14 demo windows: active 7 good; passive 6 bad and 1 doubtful (xmid 2.88, whose only flag
  is constant wavelength). No passive window is good.

## Milestone 4: MCP server

`src/paco/server.py` is the only module that imports `mcp` (SDK 2.2.0, MCP specification
2026-07-28). Run it with `uv run python -m paco.server`: Streamable HTTP on
`http://127.0.0.1:8000/mcp`, reachable from this machine only. Settings come from the server's
environment; `.env` (git-ignored) sets `PACO_INPUT_DIR=data/input`.

- **Tools**, in the workflow's order (the order `tools/list` gives):
  - `list_profiles`, `inspect_profile(profile)`;
  - `preset_settings(profile)`: the settings `run_processing` can change for that profile's
    preset, as compact JSON (2,932 characters for active), only when the agent asks;
  - `run_processing(profile, overrides)`: the preset is the profile's kind; reports progress
    after every window;
  - `quality_settings()`: the picking parameters and thresholds, with descriptions;
  - `dispersion_quality(run_id, picking, thresholds)`, `pick(run_id)`.
- **What the model reads before calling:** the 7 cards (name, description, argument schema) come to
  4,025 characters, about 1,000 tokens, on every request; the server's instructions (the workflow,
  474 characters) go into the system prompt.
- **Errors:** PACo's errors are `ValueError`s written for the agent; a decorator turns them into
  `ToolError`, whose message the model reads. Any other exception is a bug: the SDK shows the model
  only "Error executing tool ...", and logs the traceback in the server's terminal.
- **Progress:** `run_processing` reports "n of N windows" to the host (not to the model). The SDK
  runs plain tools in a worker thread, so the report goes back through `anyio.from_thread.run`.
- **Figures:** the server sets matplotlib's Agg backend, since `pick` draws from worker threads.
- **Tests** (`tests/test_server.py`, in-memory `Client`): tool order, a card budget (4,500
  characters), every argument described, instructions budget (600), results equal to the plain
  functions, the whole workflow with progress, errors the model reads, bugs it does not. Six
  deliberate server bugs, all caught.
- Checked over real HTTP with the SDK's client: 5 s for a 4-window active run, progress 0/7 to
  7/7 on a 7-window passive run.

## Milestone 5: background jobs and the inversion

- `inversion/`: a port of PAC's seismic inversion (sigpipe's MCMC, `Invert(method="mcmc")`, PAC's
  fixed Vp/Vs 1.77 and dz 0.01 m). `invert_window` writes the same 14 files as PAC's
  `invert_position` (the five models, a log, forward-modelled curves, three figures), checked
  against PAC on the same window. `InversionParameters` holds PAC's form defaults: 2 layers, Vs
  100 to 1,000 m/s (steps of 20), thicknesses 1 to 10 m (steps of 1), 100,000 iterations with
  10,000 burn-in, 5 chains.
- A job lives in the run's `inversion.json`: `submit_inversion` writes it queued (job ID
  `inv-<UTC time>-<suffix>`), `invert_run` inverts the M0 curve of every picked window in worker
  processes and records each window as it finishes (median model), `summarize_inversion` reports
  it. The file is written whole then renamed, so a reader never sees half of it.
- `jobs/`: `JobManager` runs jobs one at a time in a background thread of the server, and knows
  which are alive; a record a stopped server left queued or running reads as interrupted.
- Tools (10 in all, 5,221 characters of cards):
  - `inversion_settings()`: the parameters, on demand (1,918 characters);
  - `invert(run_id, parameters)`: asks the user to approve the curves through the host (MCP
    elicitation), then returns a job ID at once; refuses, without asking, a run with no picks or
    already being inverted;
  - `job_status(job_id)`: state, windows done, the range of each layer's Vs and each interface's
    depth so far, a few distinct errors.
- Elicitation under the 2026-07-28 specification: a server can no longer send the client a
  request of its own (`ctx.elicit` fails); the tool answers "input required" and the client
  retries with the answer. In SDK v2 a resolver parameter (`Annotated[..., Resolve(ask)]`) does
  it: the resolver runs on every round (it only reads and checks), the tool body once, and the
  model never sees the parameter.
- Timing (12 cores here): about 0.26 ms per iteration and chain on a 103-point curve, so about
  2 minutes per window with PAC's defaults; 7-point curves take about 25 s.
- Checked over real HTTP: approval asked once, `invert` answered in 0.01 s, `job_status` followed
  the job from new connections.
- Tests: `tests/test_inversion.py` (parameters, submitting, running, a window without M0, a job
  that cannot run, finding jobs, summaries, the job manager) and the new tools in
  `tests/test_server.py` (a fake user approves or declines). Nine deliberate bugs, all caught.

## Decisions (2026-09-23)

- **PAC's role (option A):** PACo depends on sigpipe only; PAC is the reference and the source of
  expected test results.
- **Layout:** one module per feature, split into sub-packages (models, logic, summary). Domain
  code never imports `mcp` or `openai`.
- **Records:** any format obspy reads. The demo profiles are SEG-2 `.dat` files.
- **Profile model:** `Profile` and `Record` built on sigpipe types, instead of PAC's parallel
  lists.
- **Presets:**
  - config models, like PAC's, now fully generated from sigpipe's signatures;
  - the agent sees only what PAC uses: no FTAN, beamforming or extra whitening methods;
  - values from PAC's forms (`distance_max` changed to 1000 m); PAC's 5 Hz whitening taper
    replaces sigpipe's 1000 Hz;
  - derived values as in PAC, except the IIR filter `fmax`, set to 0.95 × Nyquist;
  - checks like sigpipe's: root stacking `n >= 1`; a dispersion `fmax` above Nyquist is left to
    sigpipe, which lowers it to Nyquist;
  - a preset's mode must match the profile's type; short field names (`filtering`);
  - the number of workers is a server setting, not part of presets.
- **Agent view of the schema:** raw JSON Schema (MCP's standard), kept lean and under a tested
  budget.
- **Runs:**
  - timestamp run IDs in UTC, like `20260923-142501-a3f9`;
  - PAC-shaped folders, grouped by profile;
  - a process pool whose workers run in the run folder, so sigpipe's `logs/` folder and logger
    reset stay inside the run.
- **Picking (milestone 3, core built in `paco/picking/`):** automatic, adapted from malw-pipe's
  picker (`../malw-pipe/docs/picking.md`).
  - The fundamental mode is the lowest ridge, scanned upwards from the aliasing floor (2 x
    receiver spacing); Viterbi tracking inside a +/-20 % corridor, penalising relative velocity
    change per Hz.
  - M0 only (`max_modes = 1`); the higher-mode search is kept but off.
  - No wavelength limit (`max_wavelength = None`), the user's choice after comparing no limit,
    2 x L and 1 x L on the demo windows: 2 x L removed the unresolved low frequencies on active
    data without losing visible ridge; 1 x L lost 17-33 Hz of clean active ridge. Confirmed after
    seeing the saved curves, mostly beyond 2 x L (see the picking risks): the human cuts them in
    PAC's UI before any inversion. Claude would have limited the search to 2 x L.
  - A mode is judged on its kept points; a mode needs 5 kept points with a median coherence >=
    1.5 x the noise floor 1/sqrt(N). A point is kept if:
    - it is not pinned: neither on its corridor's edge, nor in a column whose ridge a search bound
      cuts (the smoothing can move such a pick off the edge);
    - it is not 0 Hz, which has no wavelength;
    - its coherence reaches the noise floor, and half the mode's median coherence (the user chose
      this relative-coherence rule over a continuity rule or both; Claude would have picked both).
  - The code lives in `paco/picking/` first, and moves to sigpipe once proven.
  - Output: `DispersionCurves_0000.csv` in PAC's layout, for good windows only (the user's choice;
    Claude would have included doubtful windows, flagged, for the approval step).
  - An existing M0 curve is replaced, like PAC's own re-pick, even one corrected by hand (the
    user's choice; Claude would have had `pick` replace only its own unchanged curves). Curves
    from an earlier pick stay in windows that are no longer good.
- **Quality metric (milestone 3, `paco/quality/`):** four measurements on the kept points of M0,
  each with a threshold that raises a flag; good with no flag, doubtful with one, bad with two or
  more, or without a ridge.
  - sharpness: peak width over the window's resolution, flagged below 0.8. A perfect plane wave
    scores 0.98 to 1.08, depending on the array, so the first threshold (1.0) flagged clean
    shots.
  - prominence (below 2), on_data (below 0.6), constant_wavelength (above 0.4).
  - The thresholds were tuned on the 14 demo windows only: recalibrate them on windows judged by
    hand.
  - Advice per flag, for the agent: wording reviewed by the user.
- **AI picker: later.** Discussed 2026-09-23: training a model (image in, M0 velocity per
  frequency out) needs labelled pairs, and none are at hand. Chosen route: keep the rule-based
  picker and the quality metric, and let human-approved picks (milestone 6) accumulate as training
  data (PAC's layout already pairs `DispersionImage_0000.hdf5` with `DispersionCurves_0000.csv`).
  A trained model can later replace the picker behind the same `pick` tool. Another idea (the
  user's, 2026-09-23): a vision-language model could judge the figure `pick` redraws, at about a
  thousand tokens of context per figure; it too must judge the exact curve saved.
- **MCP server (milestone 4):**
  - SDK: the official `mcp` 2.x (`MCPServer`), not v1's `FastMCP` nor the separate `fastmcp`.
  - Transport: Streamable HTTP (the user's choice, and Claude's): the server outlives the host, so
    milestone 5's long inversions survive an agent restart.
  - Settings on demand: `overrides`, `picking` and `thresholds` are plain objects in the cards,
    described by `preset_settings` and `quality_settings` only when asked (the user's choice, and
    Claude's), instead of about 2,500 tokens of schemas on every request.
  - The preset is the profile's kind (the user's choice, and Claude's).
  - The agent may change the picking parameters and the thresholds (the user's choice; Claude
    would have kept the judge fixed). The descriptions ask it to do so only with a reason and to
    tell the user; `quality.json` records the parameters used.
  - Tool results stay typed models (`structuredContent` plus pretty JSON text); the host decides
    what the model reads.
- **Inversion (milestone 5):**
  - PAC's form defaults, one process per window, PAC's files in each window folder.
  - Approval: PACo asks the user through the host with MCP elicitation (the user's choice; Claude
    would have used a command outside MCP, which PACo itself enforces). The model cannot answer
    the question, but the approval is only as trustworthy as the host.
  - Jobs: a background job and `job_status` (the user's choice, and Claude's). MCP's tasks
    extension is not in SDK 2.2.0.
  - The inversion parameters are exposed on demand, like the other settings.
  - One job at a time; a run holds one inversion record, replaced by the next inversion.

## Concepts covered (to confirm with the check questions)

- A tool's input schema is the model's only contract: a few named parameters with known defaults
  work better than a full config.
- Tool outputs stay short (summaries of about 200 to 300 characters), and error messages say how
  to recover.
- Presets instead of free-form pipelines. Inputs are validated before any work starts.
- Parity tests against a reference implementation, reference files that record their provenance,
  and broken copies of the code to prove the tests catch mistakes.
- Run manifests (resolved config plus versions) make runs reproducible, a basic of MLOps.
- Process isolation for libraries with global side effects. Python 3.14's forkserver needs a
  `__main__` guard in scripts.
- One source of truth: schemas generated from the library's own signatures, so drift fails
  loudly instead of silently.
- Error messages are the model's only way to fix its next call: name the field, list what is
  allowed, suggest the closest name.
- Validate against the data before any work: the same rules as the library, checked once instead
  of failing in every window.
- A tool schema travels with every request: measure it, and give it a budget.
- Test an algorithm on inputs with a known answer: synthetic data through the real transform for
  the picker, analytic images with exact expected values for the metric.
- A strict `xfail` test is an executable bug report: it fails once the bug is fixed, as a
  reminder to remove the marker.
- Overlapping rules hide each other from tests: a mutant that removes one survives while another
  rule covers for it, so each rule needs a test that isolates it.
- A tool that builds on another tool's result reads that tool's record (`pick` reads
  `quality.json`), so its output is exactly what was judged; without the record, it refuses and
  names the tool to call first.
- MCP roles: the model only reads text and writes tool calls; the host translates between it and
  the MCP client; the server exposes tools and never knows which model is on the other side.
- `tools/list` gives each tool's card (name, description, argument schema); `tools/call` returns
  content (text for the model), `structuredContent` (JSON for programs) or an error.
- Since the 2026-07-28 specification MCP is stateless: no handshake, no session. State across
  calls is a handle the server mints and the model passes back, like `run_id`.
- stdio vs Streamable HTTP: a child process that lives with its host, or a service at a URL that
  outlives it; with stdio, stdout belongs to the protocol.
- A tool description is a prompt sent with every request: what it returns, when to call it,
  what each argument means. Pydantic's titles and pretty-printing cost tokens for nothing.
- An error the model can fix must reach it (`ToolError`); a bug must not (its message may leak
  internals and cannot help the model).
- Progress notifications go to the host, not the model; a sync tool runs in a worker thread and
  sends them through the event loop.
- Test a server with an in-memory client: the same messages as over HTTP, without a network.
- Human in the loop: elicitation asks the user through the host, and the model never sees the
  question; under the 2026-07-28 specification it is a retried request, so anything that runs
  before the answer runs twice and must not change anything.
- Long work as a job: return a handle at once, keep progress and results in a durable record,
  and let the model poll; a record without a live job means the server stopped.

## Open issues

- **Milestone 0 gaps:** the defaults table (`docs/pac_defaults.md`) and the explain-back were
  skipped.
- **Tutor rules:** they are not saved as `CLAUDE.md`, so they only live in the conversation and in
  Claude's memory. Their data-layout line should say "seismic records, in any format obspy reads
  (SEG-2 in the demo profiles)".
- **PAC README:** the second wording change ("any format ObsPy can read") is uncommitted in
  `../PAC`.
- **Demo data location:** tests read `../PAC/data/input`, `PACo/data/input` holds a copy, and the
  settings default is `/data/input`. One source should be chosen.
- **`pyproject.toml`:**
  - `sigpipe[silex,santiludo]` installs keras, torch and santiludo, which no planned tool needs;
  - `known-first-party = ["sigpipe"]` should be removed or set to `paco`.
- **Stale docstring:** the `MASWParameters` docstring still says "PAC's form defaults", but
  `distance_max` is now 1000 m.
- **Runs:**
  - an interrupted run leaves a folder without `run.json` (milestone 5: write the manifest when
    the run starts, with a status);
  - scripts that call `run_processing` need a `__main__` guard.
- **Rules left to sigpipe** (rare): FK selection on unevenly spaced receivers, segments shorter
  than the 25 samples of the pipeline's padding taper, a dispersion band too narrow to hold a
  frequency step. They still fail inside each window.
- **Generated models:** pyright cannot see their fields; code reaches stages by name
  (`model_dump()["dispersion"]`) and sees presets as `PresetBase`.
- **Jobs and inversion:**
  - the approval is only as good as the host: milestone 6's host must show the question to the
    user and never let the model answer it;
  - jobs die with the server (reported as interrupted), and there is no tool to cancel one;
  - the MCMC is not seeded, so an inversion cannot be reproduced exactly (as in PAC);
  - curves with no wavelength limit are long, which makes inversions slower too;
  - the approval message holds the run folder's absolute path.
- **MCP server:**
  - a `run_processing` call keeps its HTTP request open until the run ends: fine for minutes;
  - the cards budget went from 4,500 to 5,500 characters for the inversion tools;
  - tool results are sent as pretty-printed JSON text: about a third larger than compact JSON;
    the milestone 6 host should give the model the compact `structuredContent`;
  - pydantic's `title`s stay in the tool schemas (about 13 % of a card): not worth working around
    the SDK yet;
  - `anyio` is imported directly but comes with `mcp`; `mcp[cli]` has no upper bound (a `<3`
    was suggested, the user's call).
- **Picking risks:**
  - the lowest-ridge scan can stop on the air wave (about 340 m/s) in active data on stiff soils;
  - with no wavelength limit, M0 keeps unresolved low-frequency points, and resampling over
    wavelength (one point per metre) turns them into most of the saved curve: on the four
    24-receiver active_p1 windows, 64 to 91 % of the points lie beyond 2 x the window length
    (xmid 2.88: 94 of 103 points, 3.7 to 15.1 Hz, up to 765 m/s). With `max_wavelength = 2.0` the
    same windows stay good and keep 5 to 9 points (154 to 272 m/s). Kept on purpose: the approval
    step must cut them;
  - on passive_p1 (24-receiver windows, preset defaults) the images have no clear ridge: M0
    follows the edge of the broad bright region, yet its coherence is about 2 x the floor, so
    coherence alone cannot reject it. The quality metric must. The picker also finds an M0 in pure
    noise on 48 receivers, which the metric judges bad.
  - a coherent wrong branch is still kept: when M0 drops below the aliasing floor and a higher mode
    is present, the pick carries on along M1, labelled M0 (synthetic, 48 receivers 1 m apart:
    above 75 Hz). A continuity rule would catch it.
  - the metric works on medians, so a few wrong points do not change a verdict: the approval step
    must look at the curve itself.
- **PAC's defaults:** its form defaults give 3-receiver windows, which make flat passive images
  with nothing to pick.

## Next

Milestone 6: the agent loop, the host between Qwen (vLLM, OpenAI-compatible API) and PACo's
server: tool cards to OpenAI tools, compact `structuredContent` for the model, the server's
instructions in the system prompt, elicitation questions to the user in the terminal. Ask the user
for the vLLM address and key.
