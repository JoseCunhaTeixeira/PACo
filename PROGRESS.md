# PACo progress

Last updated 2026-09-24. Read this first at the start of each session.

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
| 6. Agent loop | Done: runs with Qwen3 on vLLM, on this machine's AMD GPU; fixed from what Qwen did (see "With Qwen on the GPU") |
| 7. Evaluation | Done: 3 plays per scenario; Qwen3-4B 23 of 24 plays, Qwen3-8B-FP8 21 of 24, on the same code |
| 8. Packaging | Done: commands, Docker image (tried), Compose file with vLLM (tried on AMD with `compose.rocm.yaml`), CI (passes on GitHub after a tag fix), README |
| 9. Preprocessing and phase shift split | Done: records preprocessed once, bit-identical images for 24-receiver windows, padding dropped, default windows of 5; 364 tests |
| 10. Gate framework | Done: verdicts, flags and actions, the QC log, budgets, the configuration, attempts kept in the run, the phase shift done again per window, the report; 385 tests |
| 11. G1 signal QC, G2 image QC, metrics moved to G3 | Done: three gates with measured thresholds, the trigger stage, `judge_run`, docs per gate; 415 tests |
| 12. G3 extensions, G4 curve profile QC | Done: the curve's own rules, the pick saved as judged, the picking done again per window, G4 over the line with its pseudo-section; 433 tests |
| 13. Checks before S4, smooth model, G5, G6 | Done: S2's coherence rules (the band, the ladder), the checks before S4, the smooth median monitored, G5, G6, the inversion done again per window, G3's near field; 492 tests |
| 14. Tools, no approval, role and policy, scenarios, README | Built: stage tools with their gates' retries, redo, the job with G5 and G6, the approval removed, the agent asks only when stuck, the suite on synthetic defects; first evaluation 14 of 33, fixed; 498 tests, the ideal agent 11 of 11; second evaluation to record |
| After 14: windows and picks (2026-09-25) | Built: picks as far as the ridge holds, G3 against a plane wave, the ladder of short lengths proposing and the agent deciding; 502 tests; evaluation to run |

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

## Milestone 6: the agent loop

`src/paco/agent/`, the only package that imports `openai` (3.19, which uses `httpx2`):

- `settings.py`: `AgentSettings`, from the environment or `.env`: `PACO_LLM_BASE_URL` and
  `PACO_LLM_MODEL` (required, no default), `PACO_LLM_API_KEY` (default `EMPTY`, what vLLM accepts
  without `--api-key`), `PACO_MCP_URL` (PACo's server), `PACO_MAX_TOOL_CALLS` (15). The server's
  `Settings` now ignores these keys (`extra="ignore"`): a `PACO_LLM_*` line in `.env` used to
  stop the server.
- `model.py`: the model behind a small interface (`ChatModel`: messages and tools in, a `Reply`
  out). `OpenAIChat` calls an OpenAI-compatible chat API such as vLLM's; Qwen3's thinking
  (`<think>...</think>`) is dropped from the reply, so it does not fill the context.
- `conversion.py`: the server's tool cards become OpenAI function tools; a result becomes compact
  JSON (`structuredContent`), a string (the settings tools), or the error's message.
- `loop.py`: `Agent.answer(question)` sends the conversation and the tools to the model, calls
  the tools it asks for (with progress printed), gives the results back, until the model answers
  in plain text. Bad JSON arguments, an unknown tool, or more than 15 calls in one answer go back
  to the model as text.
- `terminal.py` and `python -m paco.agent`: a chat in the terminal. An approval question from
  `invert` goes to the user (y/N), never to the model.
- Before the first question the model already has about 1,530 tokens: the system prompt (role and
  the server's instructions, 787 characters) and the 10 tools (5,330 characters).
- Tests (`tests/test_agent.py`): a scripted model stands in for Qwen, over PACo's real server in
  memory; the request vLLM would receive is checked through a fake HTTP transport. Seven
  deliberate bugs, all caught. Not run with the real model yet.

## Milestone 7: evaluation

- The agent records each conversation (`paco/agent/record.py`): the messages as the model saw
  them, and every step with its cost: model calls (duration, prompt and completion tokens, when
  the API gives them) and tool calls (arguments, called or refused, error or not, duration, what
  the model read). The chat saves it on exit in `data/output/agent_logs/`.
- `paco/evaluation/`, run with `python -m paco.evaluation [scenario ...]`:
  - 8 scenarios, 2 per kind: look around (list, describe), process and judge (active, passive),
    recover (unknown profile, windows longer than the line), approval (declined, approved);
  - rule checks (`checks.py`): tools called with given arguments (nested objects may hold more),
    succeeded or not, order, number of calls, facts in the answer (numbers as whole numbers),
    the user asked to approve, what the server left on disk (the number of good windows, an
    inversion started or not). Added with Qwen: a tool never called, and every successful call
    of a tool keeping given arguments;
  - `--repeat N` plays each scenario N times, for pass rates;
  - a simulated user answers approval questions as each scenario says;
  - each scenario runs against PACo's real server in this process, with its own output
    directory (`PACO_OUTPUT_DIR`, restored after), and waits for the inversions it started;
  - an optional judge model (`PACO_JUDGE_BASE_URL`, `PACO_JUDGE_MODEL`) grades each
    conversation from 1 to 5 against the scenario's rubric; an unreadable grade is recorded as
    such;
  - the report: a table in the terminal, and `report.json` with one `transcript.json` per
    scenario, in `data/output/evaluations/<eval_id>/`.
- Checked that every scenario can be passed: an ideal scripted agent passes 8 of 8 (and found a
  bug, fixed: a scenario that processed nothing left no folder for its transcript).
- Tests (`tests/test_evaluation.py`): each check on hand-built conversations, the judge's
  parsing, three scenarios played with scripted agents (one right, one inventing, one going up to
  a declined approval), a whole evaluation, the report. Seven deliberate bugs, all caught.

## Milestone 8: packaging

- Commands (`[project.scripts]`): `paco-server`, `paco-agent`, `paco-evaluate`.
- Server address from the settings: `PACO_HOST`, `PACO_PORT`, and `PACO_ALLOWED_HOSTS`, which turns
  the SDK's Host-header check (against DNS rebinding) back on when the server listens beyond
  127.0.0.1; the SDK only turns it on by itself for 127.0.0.1. Checked: a request for
  `evil.test` gets 421, one for `paco-server` or `127.0.0.1` goes through.
- sigpipe without its `silex` and `santiludo` extras: nothing in PACo used them; the environment
  went from 6.0 GB to 765 MB (torch and 3.2 GB of CUDA libraries gone).
- The tests read PACo's own copy of the demo profiles (`data/input`, committed, identical to
  PAC's): no PAC clone is needed.
- `Dockerfile`: PACo's image (1.29 GB), built with uv 0.10.9 from `uv.lock`; `g++` in the build
  stage only, for bayesbay's C++ extensions (no wheel for Python 3.14). Runs as user 1000,
  reads `/data/input`, writes `/data/output`. Tried: the whole workflow over HTTP into the
  server's container (process, judge, pick, approved inversion).
- `compose.yaml`: vLLM v0.30.0 serving Qwen3 (default `Qwen/Qwen3-8B`, 16k context, flags from
  Qwen's and vLLM's documentation: `--enable-auto-tool-choice --tool-call-parser hermes
  --reasoning-parser qwen3`), `paco-server`, and the `agent` and `evaluate` services; ports on
  127.0.0.1 only. Checked with `docker compose config`; not run (no GPU here).
- `.github/workflows/ci.yml`: ruff, format, pyright and pytest on every push. The suite passes
  in a clean copy (no `.env`, no PAC clone, fresh environment), as a runner would see it.
- `README.md`: install, data, settings, commands, tools, safety, vLLM, Docker, development.
- Also: sigpipe is a third-party import for ruff (21 import blocks reordered), and the
  `MASWParameters` docstring mentions the `distance_max` exception.
- CI's first push failed: `astral-sh/setup-uv@v10` does not exist, since setup-uv publishes no
  major tags after v7. Pinned to `v10.2.0` (checked against the repository's tags, and with
  actionlint); `actions/checkout@v7` exists. The next push (f1d30f9) passed.

## With Qwen on the GPU (2026-09-23)

### vLLM on this machine

- The GPU is an AMD Radeon RX 9070 XT (RDNA4, gfx1201, 16 GB). vLLM's ROCm image
  (`vllm/vllm-openai-rocm:v0.30.0`, 49 GB) supports it. `compose.rocm.yaml` goes over
  `compose.yaml`: the ROCm image, AMD's device access (`/dev/kfd`, `/dev/dri`, group `video`,
  `SYS_PTRACE`, `seccomp=unconfined`), and `~/.cache/huggingface` for the models.
- vLLM's 4-bit formats (AWQ, GPTQ) do not run on AMD, and Qwen3-8B in bf16 needs about 16.4 GB
  for its weights. So Qwen3-4B (bf16, 7.56 GiB) first, then Qwen3-8B-FP8 (8.8 GiB).
- The first start failed. CUDA graphs for vLLM's default 512 sequences took the memory of the KV
  cache: 2.25 GiB were needed, 1.01 GiB were left. With `--max-num-seqs=8` (PACo serves one
  user), the graphs take 0.56 GiB and the KV cache holds 34,560 tokens, 2.1 conversations of
  16k. vLLM sees 15.8 of the card's 15.9 GiB free, and takes 92 % of it.
- `.env`: `PACO_LLM_BASE_URL=http://127.0.0.1:8001/v1` and `PACO_LLM_MODEL`.
- First contact: "Which seismic profiles…" took one `list_profiles` call and 2.2 s. The prompt
  is 1,608 tokens before the question.

### What the evaluation showed

The first run seemed to hang on `inversion_approved`, and was stopped. It was in fact
processing 73 windows (75 s), because the model had left out the step. Processing runs in the
evaluation's own process, so no worker process showed.

Baseline (`eval-20260923-175258-e575`): 5 of 8. What Qwen3-4B got wrong:

1. **Batches with made-up IDs.** In one reply it called `run_processing`, then
   `dispersion_quality`, `pick` and `invert` on `run_12345`, and `job_status` on `job_67890`,
   before seeing any result. It even asked for `inversion_settings` in the same batch. The
   server's errors name the latest runs, so it recovered, but at the cost of calls, and once it
   processed the profile again.
2. **Guessed parameter names.** It sent `iterations` and `chains` four times in a row, and never
   read `inversion_settings`. The error said "Extra inputs are not permitted (see
   inversion_settings)".
3. **A copied example.** It copied the one-key example of `overrides`
   (`{"masw": {"length": 24}}`) and dropped the step the user had asked for. That gave 73 windows
   instead of 4, and passive windows that were good where none should be.
4. **`n_succeeded` read as good windows.** It took the processing summary's count for good
   windows, and answered without judging.

Fixes:

1. **One tool call per reply** (`parallel_tool_calls=false`, which vLLM honours: 3 calls per
   reply became 1). The user's choice, and Claude's.
2. **Errors that name the right keys.** The overrides' explainer now also covers `picking`,
   `thresholds` and `parameters`. An unknown name gets the allowed list and the closest one
   ("Did you mean n_iterations?"). List items are written `vs_layers[0]`.
3. **A two-key example:** `{"masw": {"length": 48, "step": 12}}`.
4. **`run_processing`'s description:** a window that succeeded has an image, not yet a good one.

After these (`eval-20260923-180218-472e`): 6 of 8. `inversion_approved` recovered from
`{"inversion": {...}}` in one call, thanks to the new message. The suite ran in 4 minutes
instead of 9. Two problems were left:

5. **It told the user to call `dispersion_quality`:** "you need to run dispersion_quality
   next".
6. **It went beyond the request.** Asked to process with windows of 120 receivers, it processed
   with 96, then judged, picked and asked to invert. Its answer only spoke of the approval.

The role now says: do what the user asks, all of it and nothing more, calling the tools yourself,
since the user cannot call them. A new check, `never_called("invert")`, is on the three
processing scenarios that do not ask for an inversion. The ideal agent still passes 8 of 8.

The next run (`eval-20260923-180727-79a4`) passed 8 of 8, but its transcripts showed a failure
the checks could not see:

7. **It changed the user's windows without a word.** In both approval scenarios it processed
   with `{"masw": {"step": 24}}`: the default 3-receiver windows, none of them good. It then
   followed the quality advice, with 48 receivers (the example's value) and a narrower band,
   and inverted those windows. The approval scenarios did not check the processing.

What changed:

- **A new check.** `only_called("run_processing", ..., overrides=SMALL_WINDOWS)` is on the four
  scenarios that name windows: every run that succeeded must keep them.
- **The earlier runs, re-scored** from their transcripts: the baseline 4 of 8 (not 5), the first
  fixes 6 of 8, the role 6 of 8 (not 8).
- **The role:** keep the settings the user gave, and ask before changing any.
- **The example.** Its values had leaked both ways: one key dropped the step, two keys added
  step 12 and chose 48. It now shows placeholders:
  `{"masw": {"length": <receivers>, "step": <receivers>}}`.
- **Checks read JSON text as the SDK does.** They read an object sent as JSON text the way the
  SDK decodes it; before, such a call counted as a miss.

8. **One play per scenario is too noisy.** The same code kept the user's windows in one run and
   dropped them in the next (Qwen samples at temperature 0.6). `paco-evaluate --repeat N` plays
   each scenario N times, each play in its own folder (`judge_active/1`, ...). The report adds
   pass rates per scenario. The user's choice, and Claude's.

With 3 plays each (`eval-20260923-185238-4694`): 19 of 24 plays. Two weaknesses came back every
time:

9. **`judge_active`, 0 of 3.** One call, then "4 windows, all of which succeeded. Thus, 4 windows
   are good". The word in the result (`n_succeeded`) beat the description. Renamed
   `n_processed`.
10. **`judge_passive`, 1 of 3.** It reprocessed on the advice, with windows of its own. The
    server's instructions said "follow its advice: change settings…", against the role's "keep
    the settings the user gave". The model followed the more specific one. The instructions now
    say: follow the advice except on settings the user chose, and ask before changing those
    (571 of 600 characters).

After these (`eval-20260923-190854-32a0`): 23 of 24. The one miss uncovered a bug in the
inversion, which the model reported honestly:

11. **"2,000 iterations" crashed every window.** The burn-in stayed at PAC's 10,000, so nothing
    was left to sample: `KeyError: 'space.vs1'` in sigpipe. The job was still recorded as
    "succeeded", and the other plays passed because `invert` only has to start the job.
    - bayesbay keeps iteration i when i > burn-in and (i - burn-in) is a multiple of
      `save_every`, which sigpipe sets to 150. Tried on a window: 150 iterations after the
      burn-in keep one model; 149 keep none and crash. With one model the window still
      succeeds, but the posterior-marginals figure is skipped (a density needs two points).
    - PAC has the same bug: its backend only checks `n_burnin_iterations > 0`.
    - Fixed:
      - a burn-in that is not given is a tenth of `n_iterations`, PAC's ratio;
      - a given burn-in must leave at least 150 iterations to sample (`SAMPLE_EVERY`), or the
        parameters are refused with the rule;
      - a job whose every window failed is "failed";
      - a new check, `inversion_succeeded`, reads `inversion.json`.
    - Tests run the real sampler on both sides of the limit.
    - Fixed at the source too, at the user's request (in `../sigpipe` and `../PAC`, not
      committed):
      - sigpipe's `inversion_mcmc` names the constant (`SAVE_EVERY = 150`) and refuses such a
        run before sampling. Its first inversion tests (`tests/algorithms/test_inversion_mcmc.py`)
        run a real 300-iteration chain to show the shortest run keeps one model.
      - PAC's `InversionParameters` checks the same rule, so the form shows "Invalid config —
        parameters: … must exceed n_burnin_iterations … by at least 150" before any job starts
        (`tests/test_inversion_parameters.py`).
      - PAC pins sigpipe 31274d8, PACo 403fea2: both keep their own copy of 150 until they pin a
        sigpipe with `SAVE_EVERY`.

Also:

- JSON strings sent instead of objects for arguments are decoded by the SDK.
- The evaluation's output is now quiet (logging at WARNING; the SDK had set INFO), and the loop
  shows each failure as a `failed:` line, in the chat too.

### Results with 3 plays per scenario (final code, 2026-09-24)

| Scenario | Qwen3-4B | Qwen3-8B-FP8 |
|---|---|---|
| list_profiles | 3/3 | 3/3 |
| describe_profile | 3/3 | 3/3 |
| judge_active | 3/3 | 3/3 |
| judge_passive | 3/3 | 2/3 |
| unknown_profile | 3/3 | 3/3 |
| windows_too_long | 2/3 | 1/3 |
| inversion_declined | 3/3 | 3/3 |
| inversion_approved | 3/3 | 3/3 |
| **plays passed** | **23/24** | **21/24** |
| tool calls, all plays | 92 | 104 |
| time, all plays | 19 min | 19 min |

- Runs `eval-20260923-200656-08bc` (4B) and `eval-20260924-130503-1c59` (8B), the latter with a
  12k context (`VLLM_MAX_MODEL_LEN=12288`: on this 16 GB card the FP8 model leaves 2.17 GiB for the
  KV cache, and 16k needs 2.25). The 8B's three approved inversions were first scored as failed
  because the answer had no job ID: it had followed the job with `job_status` to its end and
  reported the models, a better answer. The check now accepts either (`any_of`); both runs are
  re-scored with it here.
- The 8B's misses: twice it used windows of 96 receivers without saying so (`windows_too_long`),
  once it reprocessed the passive line with windows of its own. The 4B's one miss: it looped ten
  times on an inversion nobody asked for, hit the 15-call budget, and then wrote "Job started":
  the only invented result seen, and the most serious failure of the suite.
- The 8B follows jobs to their end (9–10 calls per approved inversion, against 5–8), and
  runs about as fast in total: FP8 on this card is not slower than the 4B in bf16.
- After milestone 9 (`eval-20260924-135443-ec28`, Qwen3-8B-FP8): 20 of 24. The same two
  weaknesses (the passive windows changed twice, 96 unsaid once), and one play where it read
  the profile, saw 96 receivers and asked which length to use, without processing: what the
  rubric allows, and what the check `called("run_processing")` refused. That check is gone.
  Milestone 10 changed no tool: not run again.

## QC workflow (milestones 9 to 14, from 2026-09-24)

The spec is `docs/qc_workflow.md` (the user's, saved verbatim), with its decisions appended. The
plan: one milestone at a time, ruff, pyright and pytest green before the next, then
`paco-evaluate --repeat 3`; each gate documented with real examples under `docs/gates/`.

### Milestone 9: preprocessing and phase shift split

- Today one sigpipe pipeline per window does everything, from the raw records (the window's
  receivers only) to the stacked image. Both presets share the same first steps: detrend
  (constant, linear), mute, filter; everything after depends on the window (the passive
  cross-correlation uses the window's first receiver as its virtual source).
- S1, per record: PAC's trace editing, mute and filter on the whole record, saved as
  `records/<record>/Stream_0000.hdf5` (sigpipe's stream format, with its figure). S2, per window:
  the window's receivers taken from the preprocessed records, then today's remaining steps, in
  the window folder as today.
- Parity: the pipeline works in float32 from the loader on, and sigpipe's stream files keep
  float32, so the split must give bit-identical images. The test keeps a frozen copy of today's
  single pipeline and compares the two paths on demo windows.
- A record that fails S1 fails the windows that use it, with its name in their error; the other
  windows run.
- Built on 2026-09-24. `paco.pipelines`: `build_preprocessing_pipeline` (S1, one per record),
  `build_image_pipeline` (S2, one per window, per preset mode), `SelectReceivers` (PACo's own
  transformer: the window's receivers of a loaded stream, with the window's acquisition), and
  `record_folder`. `paco.runs.processing`: `preprocess_records` then `process_windows`, each
  with its own worker pool; `run.json` gains `records` (empty in older runs).
- **Padding dropped** (the user's decision, made by editing both image pipelines; recorded in
  the spec's decisions): PAC's `Pad(n=1000, taper=25)` before the phase shift is gone. "Today's
  results" are PAC's steps minus that one.
- **Parity, measured.** For windows of 24 receivers, the split's images are bit-identical to the
  frozen single pipeline's (every dataset of `DispersionImage_0000.hdf5`; for passive, the
  correlation's `Stream_0000.hdf5` too). For PAC's default 3-receiver windows they are not:
  scipy's linear detrend solves all traces of an array at once, and LAPACK rounds a 3-trace batch
  differently from the 96-trace record (2e-4 on values of 1,000, the last float32 bit; batches
  of 4, 8, 12, 16, 24 and 48 traces come out the same, 3, 5 and 6 do not). The same holds for the
  default of 5. The images then agree to 1e-6 except at 0 Hz, a sum of the signs
  of near-zero DC terms, which flips with that bit. Before the split, the last bits of a trace
  thus depended on the window it was in; now a record is preprocessed once for every window.
- With PAC's padding still in, the two paths also differed in the last float32 digits for
  24-receiver windows (seen once, not chased: the padding went).

### Milestone 10: the gate framework

Built on 2026-09-24, `src/paco/qc/`, domain code only (no mcp, no openai):

- `models.py`: one language for every gate (rules 1 and 2). `GateResult` per unit (a record or
  a window): verdict `pass`, `retry` or `reject`; each `Metric` with its threshold and bound;
  `Flag`s with the stage at fault, a message, and an `Action` (`Override` with overrides ready
  to apply, `ExcludeTraces`, `ExcludeRecord`, `Reject`), fixable or not; `Kept` (band,
  wavelength range, points, traces, records: rule 6). `Attempt` is one line of the log (rule 8):
  unit, stage, attempt number, parameters, what triggered it, times, status, the gate's result.
- `log.py`: `qc_log.jsonl` in the run folder, appended one attempt per line: safe against a crash
  and readable during a run; a unit's state (attempts, retries spent) is read back from it and
  kept nowhere else. `ensure_initial_attempts` writes the run's first attempts from `run.json`
  (`paco.runs` never imports `paco.qc`: the import would be circular through `paco.quality`).
- `budgets.py`: retries, not attempts (rule 4): 2 per gate and unit, 2 x xmids per run, shared;
  `can_retry`; `budget_spent` rejects with a `budget_spent` flag ahead of the last attempt's.
- `config.py`: `QCConfig`, every threshold and budget in one place (rule 9): the budgets, and
  `curve`, today's `QualityParameters` (G3); each gate adds its own model at its milestone.
  Loaded from the JSON file `PACO_QC_CONFIG` names, else the defaults; snapshotted into the run
  as `qc_config.json`. Locked during a run: the tools will stop taking thresholds (milestone 14).
- `attempts.py`: `invalidate(window, stage, n)` moves the results of that stage and the later
  ones into `xmid_<x>/attempts/<n>_<stage>/` (rule 3), by stage: the image and the window's
  figures (S2), the curve and `quality.json` (S3), PAC's inversion files (S4). The final result
  stays at the top, in PAC's layout.
- `rerun.py`: `rerun_phase_shift(run_id, units, overrides, settings, triggered_by)`: the
  windows named are invalidated from S2 on, the phase shift runs again with the run's preset
  plus the overrides (`presets.apply_overrides`: a stage given with another method is replaced
  whole, otherwise merged; then resolved against the profile again), and the log gets the
  attempts. `masw` cannot change (the windows would move); unknown windows are refused with
  the run's. Re-running the picking and the inversion per window comes with their gates
  (milestones 11 and 13), when their records become per window.
- `report.py`: `QCReport` per run, from the log: per unit, the latest verdict and flags of each
  gate, the attempts, the final parameters of each stage, the reasons of a reject, and what the
  latest curve kept (rule 8); counts per gate; `qc_report.json`. `summarize_report` for the
  agent: counts, retries, then each flag with the stretches of xmids raising it and the change
  it suggests ("G2 ridge_at_vmax, xmid 12.00-13.00 (3), 14.00 (1): … -> phase_shift
  {"dispersion":{"vmax":1500}}"), never one line per xmid.
- Tests (`tests/test_qc.py`, 21): the models round-trip, the log and its counts, both budgets,
  the configuration from a file and its snapshot, the archive of a stage and the stages after
  it, the report and its summary, and the phase shift done again on a real run (the archived
  image is the first one, the new one has the new velocity range, the other windows are
  untouched, the log holds the initial attempts then the new ones, and again counts on).
- Open: `run.json`'s window statuses are the initial run's; after a re-run the log is the truth
  (`dispersion_quality` still reads `run.json`; it moves to G3 in milestone 11).

### Milestone 11: G1, G2 and G3

Built on 2026-09-24 (`src/paco/qc/g1_signal.py`, `g2_image.py`, `g3_curve.py`, `judging.py`;
documented in `docs/gates/`, with the demo's real outputs and figures):

- **G1, signal QC per record.** Dead, clipped and NaN traces; RMS off the decay with offset
  (Theil–Sen fit in log-log, 3 MADs and a factor 2: a smooth decay has tiny MADs); SNR and the
  usable band from a surface-wave window (arrivals at vg_max to vg_min, padded) against a noise
  window (before the trigger, else after the slowest arrival), the band within 6 dB of the noise
  and 20 dB of the peak (after the wave, the noise window is quieter at every frequency); lateral
  coherence and reversed polarity from neighbours' cross-correlations; the trigger from first
  breaks (a 5 ms envelope above 5 x the noise RMS, on traces whose SNR passes, fitted robustly).
  Passive records get the three trigger-free checks.
- **The trigger decision.** Both demo records are triggered late (+18.8 and +10.4 ms; the first
  break at 0.75 m comes at 19 to 21 ms). The spec's rule (reject) would have rejected the whole
  demo line: the user chose to correct it in S1. The active preset gained a `trigger` stage
  (`t0`, 0 by default; `paco.transformers.ShiftTrigger`, in a module the presets can import),
  G1's flag is a retry with the measured t0 as its override, and breaks that are no line
  (scatter beyond 50 ms) still reject. The active schema grew by 170 characters (budget 3,300).
- **G2, image QC per window.** Coherent columns by a level between the floor 1/sqrt(N) and 1
  (a multiple of the floor is impossible with 5 receivers); ridges on the grid's edges (a vmin
  below 30 m/s is an artifact to start above); the band reaching the image's edges; competing
  ridges (a higher mode -> pick two modes; below the aliasing limit 2 dx f -> cut the band);
  a band narrower than G1's usable one.
- **G3, curve QC per window.** Today's metrics wrapped as a gate: sharpness rejects (not
  fixable at this window length), prominence filters to the band or mutes, on_data narrows the
  band, constant wavelength cuts at 2 window lengths, no ridge loosens the mode rule.
- **`judge_run`:** G1 on the records, G2 on the images, then the picking as an attempt of its
  own with G3; results against the log's attempts (`record_result` appends the attempt again
  with its result; the last line of an attempt is its current one); `qc_config.json` and
  `qc_report.json` written. On active_p1 (24-receiver windows): G1 2 retry, G2 4 retry, G3
  4 pass; the summary names the trigger corrections, the three traces to exclude, and the
  band that reaches PAC's default 100 Hz while the records are usable to 324 Hz.
- **Thresholds**, measured then accepted by the user (`docs/gates/*.md`, "To judge" lists what
  is still open). Findings: with the default windows of 5 receivers, prominence is 1.03 to
  1.35 on all 92 windows and sharpness rejects 33; no threshold separates them. The window
  length must come from the data (the coherence rules).
- Tests: G1 on synthetic causal shots with one defect each (14), G2 on analytic images (8),
  G3 on the quality tests' images with exact picks (6), `judge_run` on a real run.
- Not yet: the coherence rules (S2 parameters from G1's band and the geometry), G1's
  actions applied by the loop (exclude traces, exclude a record), re-running the picking and
  the inversion per window, and the gates' documentation of G4 to G6.

### Milestone 12: G3's curve rules and G4

Built on 2026-09-24 (`src/paco/qc/g3_curve.py` rewritten, `g4_profile.py`, `judging.py`'s
`judge_picking` and `judge_line`, `rerun.py`'s `rerun_picking`, `picks.saving.save_pick`;
documented in `docs/gates/G3.md` and `G4.md` with the dense demo line's pseudo-section):

- **G3's curve rules**, on the curve saved for the inversion (resampled in wavelength):
  wavelengths beyond 2 window lengths are cut (the run's picking starts with `max_wavelength`
  2, in `qc_config.json`'s `picking`), points below 2 dx reported, at least 5 points, a mode
  jump above 30 % between consecutive points (half the corridor), the air wave at 330–345 m/s
  (a mute to 320 m/s), an inverse trend by Spearman's rank correlation (kept, never rejected),
  uncertainties above 50 % of the velocity rejected (the array's geometry: only longer windows
  help). `CurveThresholds` wraps today's `QualityParameters` under `metrics`. `kept.n_points`
  is now the resampled curve's count (5 to 10 on 24-receiver windows), what the inversion gets.
- **The pick saved as judged.** `judge_picking` picks, writes `DispersionCurves_0000.csv` (and
  redraws the figure) through `save_pick`, then judges with G3 and logs the attempt with its
  parameters and result; a pick done again archives the previous curve and inversion under
  `attempts/<n>_picking/`. `rerun_picking(run_id, units, overrides, settings)` starts from each
  window's latest pick (G4's guide differs per window), refuses unknown windows and parameters.
- **G4, the curve profile QC**, `judge_line`: the curves whose latest pick passed G3, read back
  from their files, each against two neighbours a side at the union of their wavelengths. An
  outlier is off neighbours that agree with each other (within 7.5 % of their median) and fits
  no side (15 %): re-picked with the neighbours' median as the picking's `guide`. Off one side
  and fitting the other: a shared change, geology, kept. Sides that disagree among themselves
  or share fewer than 3 wavelengths judge nothing. The line's own result (unit `line`): the
  gaps, the spread of the depths of investigation; rejected only without any curve.
- **Measured on active_p1.** 19 windows of 24 receivers every 4: G3 17 pass, 2 `mode_jump`
  (steps of 53 and 31 %; the good curves step by up to 26 %), inverse trend on 11 (kept),
  uncertainties 8 to 15 %; G4: the 17 curves within 2 to 13 % of their neighbours, no outlier,
  the two jumping windows as gaps. The tests' 4 windows every 24 receivers: one neighbour a
  side is no evidence (`no_neighbours`), where a first version called xmid 8.88, 28 % off both,
  an outlier. Default 5-receiver windows: 67 windows without a ridge within 2 m (frequencies
  above the image's 100 Hz), 25 with one point, 4 with 56 % uncertainties: no curve for the
  line. The window length must come from the data (coherence rules, milestone 13).
- Rejected on the way: the spec's "1 or 2 xmids" as an isolated outlier (two changed windows
  make their neighbours disagree; a k-neighbour median flagged both edges of a step as
  outliers), a side of one curve as evidence, a `_limits` registry keyed by `id()` in a frozen
  dataclass (a field instead).
- Tests: G3's curve rules on curves given with the pick (8 more), G4 on synthetic lines (9),
  `judge_run` with G4 and `rerun_picking` on a real run.
- Evaluation after milestone 11 (Qwen3-8B-FP8, 3 plays): 20 of 24. `judge_passive` 0/3: told
  by `dispersion_quality` that every 24-receiver window is bad, the model changed the window
  length the user chose (48, 64, 96 receivers) instead of asking; `windows_too_long` 2/3 (one
  answer without the number 96). Both checks enforce the "ask before changing" rule that
  milestone 14 replaces (the agent changes and says so).
- Evaluation after milestone 12 (`eval-20260924-153639-5e5b`, Qwen3-8B-FP8, 3 plays): 23 of
  24. The one miss is the same: on `judge_passive` it reprocessed the passive line with windows
  of 48 receivers instead of the user's 24. No tool changed since milestone 11, so 20 and 23
  are the same code: three plays per scenario are that noisy.
- Not yet: the coherence rules (S2 parameters from G1's band and the geometry), G1's actions
  applied by the loop, the inversion re-run per window, `invert` reading G4's verdict.

### Milestone 13: parameters from the data, the inversion's gates

Started on 2026-09-24. Measured first on active_p1, then the design brought as four options
(recorded in `docs/qc_workflow.md`, "Taken during milestone 13"; the user chose Claude's
recommendation each time):

- **Window length.** G3 on active_p1 every 4 receivers, fmax 100 Hz: 0 of 23 windows pass at
  5 and 8 receivers, 0 of 22 at 12, 12 of 21 at 16, 17 of 19 at 24, 15 of 17 at 32, 13 of 13
  at 48; the kept curves reach 7, 11, 15 and 21 m. Chosen: the shortest length at which two
  thirds of a few trial xmids pass G3, found by a ladder before S2.
- **The band.** Opening the image to G1's usable 324 Hz made G3 worse at every length (24
  receivers: 17 pass at 100 Hz, 15 at 150 Hz, 6 at 324 Hz; 16 receivers: 12, 8, 4): above
  100 Hz a second ridge competes and the picker jumps. So G2's `band_at_fmax` retry (1.5 x
  fmax) would have made the loop worse. Chosen: fmax capped by G1 and Nyquist, never widened;
  G2's two band flags kept, not retried.
- **Priors.** Inverting the dense line's 17 good curves with PAC's bounds: chains converge
  (split R-hat at most 1.08, acceptance 22 to 40 %), but the interface sits at 6 to 7.5 m,
  beyond the resolved depth (about half the longest wavelength, 5.5 m), and the half-space's
  Vs is the prior's (posterior spread 60 to 110 % of the prior's at 7 to 10 m). With bounds
  from each curve (Vs 0.8 Vr_min to 1.5 Vr_max, half-space top at most λmax/2): interface at
  1.9 to 3.9 m, chain medians within 2 to 11 %, misfits 0.23 to 1.08 (normalised by the
  uncertainties) against 0.29 to 1.28. Chosen: derived per window.
- **G5's model.** The smooth median misfits 2.3 to 3.2 where its layered median fits within
  1.0 to 1.3, on the three windows with a thin stiff top layer (8.88, 12.88, 13.88): PAC's
  smoothing spreads the boundary over about 3.5 m. Chosen: G5 judges the smooth median by band,
  and keeps with a flag a window where only the smoothing loses the fit.
- **The near-offset rule, measured** (the spec's "nearest source offset >= λmax/2", as
  `masw.distance_min` = 1.5 window lengths): it drops the near shot from the windows at both
  ends of active_p1, and G3 falls from 17 to 11 passes of 19 at 24 receivers (8 mode jumps),
  from 12 to 5 of 21 at 16. Not applied; to bring to the user with the thresholds.
- **Found:** `invert_window` (ported from PAC) forward-models the median model for its figure
  without a guard; on xmid 13.88 with derived bounds, disba found no fundamental mode for a
  274 m/s layer over a 202 m/s half-space (the picked 291 m/s at 97 Hz is faster than any
  normal mode of that model), and the window failed after its inversion had succeeded.
  Guarded: the figure goes without that curve.

Built on 2026-09-24 (`src/paco/qc/coherence.py`, `line.py`, `priors.py`, `g5_model.py`,
`g6_models.py`, `sides.py`, `inverting.py`; `src/paco/inversion/measuring.py`; documented in
`docs/gates/S2_rules.md`, `S4_checks.md`, `G5.md`, `G6.md`, with `G2.md` updated):

- **The coherence rules for S2** (`coherence.py`, `line.py`). `cap_band`: fmax capped at every
  record's usable fmax (G1) and Nyquist, fmin raised to the highest usable fmin, never widened.
  `choose_length`: the ladder 5, 8, 12, 16, 24, 32, 48, 64, 96, 128 receivers (at most half the
  line), from the user's length when given, each tried on 5 windows spread along the line (S2,
  the picking, G3; their folders in `coherence/<length>/`, the trials in `coherence.json`); the
  first where 2/3 pass is kept, else the best tried. `process_line`: S1, G1, these rules, S2 on
  the whole line, then `judge_run` (G2 to G4); the rules' changes are notes of a `line` phase
  shift attempt. `run_processing` was split into steps it shares (`new_run_folder`,
  `write_manifest`, `process_windows` reading another folder's records).
- **G2's band flags kept** (the decision on the band): `band_at_fmax` and `narrower_than_usable`
  are information now, and G2's verdict ignores kept flags (the four demo windows pass G2).
- **The checks before S4** (`priors.py`, `derive_inversion`): Vs bounds 0.8 x to 1.5 x the
  curve's velocities, layers at least λmin/3 thick, the half-space's top at most λmax/2 (shared
  by the layers above), PAC's steps in proportion; values given are kept when they pass (Vs
  bounds reaching the slowest velocity and 1.09 x the fastest, thicknesses within the limits,
  no more layers than the curve resolves) and changed with notes otherwise.
- **The smooth median, monitored** (`paco.inversion.measuring`). An inversion now saves its
  posterior samples (`SeismicInversion_Samples_0000.npz`); `measure_inversion` reads the fit of
  the smooth and layered medians by band (RMS of residuals over the uncertainties, PAC's
  residual), the split R-hat per parameter (chains cut back apart from sigpipe's
  concatenation), the acceptance rates (from the log), the share of samples at each bound,
  the useful depth (posterior against prior spread of Vs by depth, the prior drawn with a seed)
  and the smooth median's Vs at round depths (1 to 5 m on the demo). `job_status` reports those
  depths' Vs ranges, the useful depth and the misfit, instead of the layered model's.
- **G5** (`g5_model.py`): misfit by band (at most 2), the layered median's misfit (the
  smoothing told apart), R-hat 1.1, acceptance 10 %, 100 models a chain, 10 % of samples at a
  bound, the useful depth reported. Actions, cheapest first: sample twice as long, widen a Vs
  bound (x0.8, x1.25), a thickness bound below the curve's reach, one layer fewer when a layer
  piles at its thinnest, one more layer (up to 4) when the model misfits; kept: the smoothing's
  misfit, an interface at the depth limit; rejected: points no fundamental mode reaches.
- **G6** (`g6_models.py`): the smooth medians' Vs at the report depths, down to each useful
  depth, each against two neighbours a side with G4's side logic (moved to `sides.py`, shared);
  a model off agreeing neighbours while its curve fits theirs is non-unique (invert again,
  twice as long); the curves showing the change keep it; the line's gaps and useful depths.
- **S4 the QC way** (`inverting.py`): `judge_inversions` inverts the windows G4 passed (refuses a
  run G4 has not judged, or a line it rejected), each with derived bounds, in workers, logs each
  as an `inversion` attempt with its parameters and notes, then G5 and G6;
  `rerun_inversion` starts again from a window's latest parameters, archives the previous
  attempt, and derives every bound again for another number of layers.
- **The report**: the checks' notes and the failed attempts in the summary, grouped by stretches
  of xmids; stretches follow the line's spacing (xmids named to 2 decimals sat 0.24 and 0.26 m
  apart and broke every run of a 0.25 m line; two windows far apart printed as a stretch).
- **Measured on active_p1** (dense line, 17 curves, PAC's effort, derived bounds): G5 17 pass,
  four `smoothing_misfit` kept (2.2 to 2.8 against layered 1.03 to 1.41); R-hat at most 1.012,
  acceptance 47 to 75 %, at most 8.9 % of samples at a bound (the inverse windows' half-space
  wants to go softer); useful depth 2.3 to 4.3 m or the whole model; G6 16 fit, xmid 13.88 a
  change shared with one side, kept. `process_line` with no settings: the ladder keeps 16
  receivers (4 of 5 trials pass), 50 of the line's 81 windows pass G3 (33 s).
- **Synthetic defects** (a curve from 200 m/s over 350 m/s at 3 m, uncertainties 10 %): derived
  bounds recover 202 over 341 at 3.1 m; 3,000 iterations: `not_converged`; bounds 10 to 50 %
  wrong (Vs2 at most 320, thickness at most 2 or 1.5 m, at least 4 m) still fit within the
  errors, with 4 to 8 % of samples at the bound, where the demo's own inverse windows put 5 to
  9 %: G5 cannot see them on this data, the derived bounds are the protection. A Vs1 minimum
  above the curve's slowest velocity is changed by the checks before S4.
- **Found in sigpipe:** with Vs2 at most 300 m/s, `inversion_mcmc` fails after sampling
  (`operands could not be broadcast together with shapes (600,) (480,)`): bayesbay saves a
  model's predicted curve only once its likelihood has been computed, so a chain stuck on its
  first model keeps none, and sigpipe sums misfits over both. PACo logs the window as a failed
  attempt, and the summary says so. Not fixed at the source (the user's call).
- **Thresholds, brought with the measurements after the build, accepted by the user** (recorded
  in `docs/qc_workflow.md`): the ladder at 9 trial windows and 80 % (as first built, 5 trials at
  two thirds kept 16 receivers on active_p1, where 50 of the line's 81 windows passed G3; now it
  keeps 24, where 66 of 73 pass); the near-offset rule reported, not applied (G3's
  `near_offset` metric and kept `near_field` flag: the ten windows at the dense line's ends,
  38 of 73 on the whole line at 24 receivers); G5's and G6's defaults as proposed.
- Tests: 492 (priors 9, measures 17, G5 14, G6 6, S4 on a real run 5, the S2 rules and a line
  6, G3's near field, the summary's notes, failures and stretches).
- Evaluation after milestone 13 (`eval-20260924-164113-aef5`, Qwen3-8B-FP8, 3 plays, run before
  the ladder's and G3's last changes, which no tool uses yet): 20 of 24. `judge_passive` 1/3
  (the model replaced the user's 24 receivers by 48 and 72 on the passive line, as before);
  `windows_too_long` 2/3 (an answer without 96); `inversion_approved` 2/3, a new miss: the
  model turned 24 receivers into 6 m and sent `{"masw": {"length": 6.0, "step": 6.0}}`,
  accepted as 6 receivers, before processing again with 24; it then sent the same wrong
  `invert` call three times. The schema says nothing of `length` being receivers: for
  milestone 14. The ideal scripted agent passes 8 of 8.
- Not yet: the loop that applies the gates' changes, and the tools over `process_line`,
  `judge_inversions` and the reruns (milestone 14).

### Milestone 14: the tools, the loop, the agent's policy

Started on 2026-09-24. Brought as options before building (recorded in `docs/qc_workflow.md`,
"Taken at the start of milestone 14"; the user chose Claude's recommendation each time): stage
tools with their gates' retries and `redo`; synthetic defects in the data where physical; the
evaluation at 8 workers and PAC's effort; the host refusing a repeat of a failed call. During the
build the user added: the agent may ask, "if it is needed, so I can help him choose" (brought
as options: it asks only when stuck); and asked whether the model should build its own pipelines
from sigpipe's transformers (Claude's answer: not with Qwen3-8B; validated optional stages per
preset later, if wanted).

Built (`src/paco/qc/line.py`, `curves.py`, `inverting.py`, `redo.py`, `loops.py`;
`src/paco/server.py`; `src/paco/agent/loop.py`; `src/paco/evaluation/`; documented in
`docs/gates/loop.md`, the README, `G2.md`):

- **The stage loops.** `process_line` (run_processing): S1, G1 with its fixes (a record
  preprocessed again with its own trigger correction, the traces and records it excludes left
  out of the windows, `Exclusions` in `paco.windows` and `run.json`), the S2 rules, S2, G2
  with its retries in groups of windows sharing a change. `pick_line` (pick): S3 and G3 with
  its re-picks, G4 and its outliers re-picked along the neighbours' guide. The inversion job
  (`submit_inversion`, `run_inversion_job`): S4 on the windows G4 passed, G5's and G6's retries,
  each window recorded as it ends, the gates' summary at the end. `redo_stage` and the job for
  the inversion: back to a stage for windows by xmid or flag, then what follows up to G4.
  `RetryBudget` counts each retry as it is granted; `redo` is refused once the run's budget is
  spent.
- **The summary says what the loops did**: each retry trigger with its windows, the change it
  made ("with phase_shift {"dispersion":{"vmax":375.0}}") and the verdicts now; the checks'
  changes and exclusions as notes, grouped by text; failed attempts.
- **The tools** (9, 5,424 characters of cards): `run_processing`, `pick`, `invert`,
  `job_status` (waits up to 2 minutes for the job), `redo`, and the four unchanged ones;
  `dispersion_quality`, `quality_settings` and the approval are gone (thresholds locked, G4 the
  go or no-go). Each stage tool returns the run, the gates' summary and what to do next, "you
  are stuck, ask the user" included. The server's instructions (534 characters) and the role
  (845): plan, act, observe, adapt; say which settings the gates changed; ask only when stuck,
  one question with 2 or 3 options, its choice first. The host refuses a call identical to one
  that just failed. The terminal no longer relays approvals.
- **Fixes found on the way**: G2 fooled by a grid too narrow (the artifact at 1 m/s read as an
  alias, the band cut to 12 Hz): the edge, competing-ridge and alias checks look above the
  30 m/s floor, and the grid comes first; the ladder's trial windows get G2's grid fix; a band
  wholly below the usable one now goes up instead of failing; G5 asks at once the iterations
  100 models a chain need (2,000 iterations doubled twice left 48); the job records its depths
  and the final verdicts; batches of retries overshot the run's budget.
- **The evaluation** (`paco.evaluation`): 11 scenarios, the approval ones gone. Built per
  evaluation from the demo: `active_dead` (trace 40 of the first record zeroed, as MiniSEED).
  Scenarios: looking around (3), the loop (`pick_active`: G1's trigger, G3's mode jump;
  `dead_trace`; `narrow_velocities`: vmax 250 m/s; `few_iterations`: 2,000; `tight_bounds`:
  Vs 100-180 m/s; `zero_settings`: the whole line to Vs models with no setting), stuck
  (`no_curve`: the passive line; `windows_too_long`). New checks: the agent asked nothing / asked,
  the thresholds stayed the configuration's, the loop retried for a trigger, a trace excluded,
  no settings invented, the value a gate set. The ideal scripted agent passes 11 of 11
  (zero_settings: 12 minutes on 8 workers).
- **Defects that could not be built** from the demo's records: an air wave the picker follows
  (the demo's soil is slower than the air wave; scaling the geometry or compressing the time
  axis broke G1's windows and band), and an isolated G4 outlier (overlapping windows spread any
  receiver defect over several windows: geology to G4). Both stay covered by the unit tests.

**First evaluation** (`eval-20260924-184935-a154`, Qwen3-8B-FP8, 3 plays, 8 workers): 14 of 33
plays. Looking around 9 of 9, `no_curve` 3 of 3, `windows_too_long` 2 of 3 (once it answered
without asking); every loop scenario 0 of 3. What failed, read from the transcripts:

- **Questions at the end** (15 plays): "Would you like me to invert them?", or asking what to do
  about a rejected window. The role said "ask only when stuck" but not what is not stuck.
- **The gates' changes not said** (vmax 375, 17,000 iterations, the bounds past 180 m/s, trace
  40): they were in the summary, as JSON inside a retry line, and the model did not pick them
  out.
- **The next step pushed too far**: `pick`'s hint said "call invert", and the agent inverted
  when the user asked for curves only (2 plays).
- **Setting names guessed**: `velocity_limit` or `velocity` at the top of the overrides, or
  vmax left out and given to `invert` as `max_velocity` (`narrow_velocities`; once it read
  `preset_settings` after the refusal and stopped). In `tight_bounds`, up to six calls to
  `invert` before one passed: `layers.bounds`, `sampler.iterations`, then one Vs range for two
  layers, refused.
- **sigpipe's sampler failed** on 1 window of 66 (`zero_settings`: "operands could not be
  broadcast together with shapes (3000,) (2975,)", a chain that kept no predicted curve for some
  models), so the check "every window a model" failed.

Fixed after it:

- The tools return `changed`, the settings the gates and the checks changed in words: "dispersion
  vmax 250 -> 375 at xmid 3.88-15.88 (3), by G2:ridge_at_vmax", the notes of the checks
  ("line: masw length 32 for the whole line: ..."). The job's status has it too, once ended.
- The role (1,051 characters) says what stuck is (no image or curve left, the budget spent
  before the request is done, a request the data do not allow), that rejected windows are gaps
  to report, that the answer ends without an offer or a question, and to report every item of
  `changed` and the windows left without a result.
- The hints are conditional: "pick comes next for run_id ..., if the user asked for curves or
  models"; "3 curves passed G3 and G4. invert can run on run_id ..., if the user asked for
  models; otherwise answer."
- The settings' descriptions show their shape with placeholders (`{"dispersion": {"vmax":
  <m/s>}}`, `{"vs_layers": [{"vs_min": <m/s>, "vs_max": <m/s>}]}`), the window length says
  "receivers, not metres", and one Vs or thickness range stands for every layer.
- A failed inversion is tried once more with the same parameters (trigger `S4:failed`, from the
  gate budget "S4"): the sampler is not seeded, and the failure does not come back every time.
  The fix at the source belongs in sigpipe (open issue).

Checked after the fixes: 498 tests pass, ruff and pyright clean; the ideal scripted agent passes
11 of 11 (`zero_settings` in 530 s). The second evaluation (`eval-20260924-211019-73d5`, Qwen3-8B-FP8,
3 plays), started on 2026-09-24 at 23:10, was stopped unfinished at the user's request (the
computer going to sleep): run it again, after the window change below.

**CI failed on the commit of milestones 9 to 14** (`05cc650`, GitHub's Tests step; lint,
format and types passed). Reproduced on 2026-09-25 in a fresh clone of the commit, without
`.env`, on 4 cores like GitHub's runners (all passed here, on 12 cores with `.env`); three
tests, fixed:

- `test_the_ladder_keeps_the_shortest_length_that_passes` built `Settings` without
  `input_dir`: `.env` supplied it here, CI fell back to the container's `/data/input`.
- `test_the_split_runs_and_gives_todays_results`, passive: bit for bit on 12 cores, but on 4
  the correlations round some values differently in the last float32 bit (at most 1.3e-7 of
  the largest value), as LAPACK's batches do for the default windows. Floats are now compared
  within a millionth of the largest value, the rest exactly.
- `test_g5s_retries_are_in_the_jobs_summary` (seen in one clean run of three): the unseeded
  sampler sometimes needs a second G5 retry for a window, and the line then says "(each its
  own)"; the test now checks only the part that does not depend on the draws.

The clean clone then passes 498 of 498 on 4 cores.

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
- **Packaging (milestone 8):** Docker Compose with vLLM (the user's choice; Claude would have
  stopped at the Docker image, which can be tried here); sigpipe's extras dropped and CI on
  PACo's demo data (the user's choices, and Claude's).
- **Evaluation (milestone 7):** rule checks and an optional judge model (the user's choice;
  Claude would have started with rule checks alone, since a judge needs a second model, and a
  model grading itself is biased); four kinds of scenarios, two each.
- **Agent loop (milestone 6):** our own loop with the `openai` client and the MCP client (the
  user's choice, and Claude's), instead of an agent framework: every step is visible.
- **With Qwen:**
  - One tool call per reply (`parallel_tool_calls=false`; the user's choice, and Claude's). In a
    batch, Qwen3-4B made up the IDs that earlier calls had not returned yet. Rejected: a rule in
    the role (this model ignored "see inversion_settings"), and leaving the errors to correct it
    (calls lost, a profile processed twice).
  - Repeated plays for the evaluation (`--repeat N`; the user's choice, and Claude's), since
    the model samples. Rejected: greedy decoding (it measures a mode the chat does not use, and
    Qwen warns it makes the thinking loop), and single runs read as rough signs.
  - The burn-in: a tenth of `n_iterations` when not given, and at least 150 iterations left to
    sample (the user's choice, and Claude's). Rejected: the check alone (the model would have to
    choose a burn-in the user never mentioned), and the scaling alone (a bad explicit value
    would still crash every window).
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
- The agent loop: the host sends the conversation and the tools, the model answers with text or
  tool calls (each with an ID its result must quote), the host runs them and sends the results
  back, until the model answers in text. The model never talks to the server.
- Test an agent without a model: a scripted model gives fixed replies and records what it was
  sent; the server and the loop are the real ones.
- Evaluate an agent on scenarios: check what it did (tool calls and arguments), what it left
  behind (files on disk), and what it said (facts in the answer); a judge model can grade the
  rest, with its own errors. Make sure a perfect agent can pass the suite, or the suite is
  wrong.
- A small model copies the values of the examples in tool descriptions: a placeholder shows the
  shape without a value to copy.
- Parallel tool calls invite made-up IDs: a call that needs another call's result must come
  after it.
- An error that names the allowed keys, and the closest one, lets even a 4B model fix its call
  in one step; "see inversion_settings" did not.
- Checks only see what they look for: a passing table hid a model changing the user's
  settings. Read the transcripts. And a model that samples needs repeated plays before two
  scores can be compared.

## Open issues

- **Milestone 0 gaps:** the defaults table (`docs/pac_defaults.md`) and the explain-back were
  skipped.
- **Tutor rules:** they are not saved as `CLAUDE.md`, so they only live in the conversation and in
  Claude's memory. Their data-layout line should say "seismic records, in any format obspy reads
  (SEG-2 in the demo profiles)".
- **PAC README:** the second wording change ("any format ObsPy can read") is uncommitted in
  `../PAC`.
- **Packaging (milestone 8):**
  - only vLLM's service has run, on AMD with `compose.rocm.yaml`. Not tried yet: the NVIDIA
    path, and the `paco-server`, `agent` and `evaluate` containers next to vLLM (the agent's
    TTY);
  - CI passes on GitHub since the tag fix (commit f1d30f9);
  - the image is 1.29 GB (scipy, obspy, matplotlib, h5py, bayesbay);
  - `mcp[cli]` still has no upper bound.
- **Runs:**
  - an interrupted run leaves a folder without `run.json` (milestone 5: write the manifest when
    the run starts, with a status);
  - scripts that call `run_processing` need a `__main__` guard.
- **Rules left to sigpipe** (rare): FK selection on unevenly spaced receivers, segments shorter
  than the 25 samples of the pipeline's padding taper, a dispersion band too narrow to hold a
  frequency step. They still fail inside each window.
- **Generated models:** pyright cannot see their fields; code reaches stages by name
  (`model_dump()["dispersion"]`) and sees presets as `PresetBase`.
- **Agent (milestone 6):**
  - Qwen3 thinks before each reply: up to 1,800 tokens and 28 s for a single step with Qwen3-4B.
    Its thinking is not kept in the conversation. Turning it off (`enable_thinking: false`)
    would be faster; the evaluation can say what it costs in quality;
  - about 1,600 tokens are taken before the first question, and every result stays in the
    conversation: long sessions will need trimming in an 8k context;
  - conversations are saved by the chat (milestone 7).
- **Evaluation (milestone 7):**
  - Qwen samples (temperature 0.6 in its `generation_config.json`, which vLLM applies): compare
    models or prompts with `--repeat`, and even 3 plays only give coarse pass rates;
  - the checks only see what they look for: an 8 of 8 hid a model changing the user's windows,
    until `only_called` was added. Read transcripts, not only the table;
  - no judge model has been run: on this 16 GB card, vLLM serves one model at a time;
  - the fact checks are strict about wording: "0,25 m" or "25 cm" miss "0.25";
  - scenarios run the server in the evaluation's own process, not over HTTP, and point it at
    their folder through an environment variable;
  - a judge using the same model as the agent grades its own answers: better another model.
- **Jobs and inversion:**
  - the burn-in fixes in `../sigpipe` and `../PAC` are not committed yet. Once sigpipe's is
    pushed, PACo and PAC can import `SAVE_EVERY` instead of keeping 150 themselves;
  - PAC's `src` has 141 pyright errors in strict mode, before and after its burn-in fix;
  - an `inversion.json` written before the burn-in rule, with fewer than 150 iterations after
    the burn-in, no longer loads; none exists outside the evaluation folders here;
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
  with nothing to pick. PACo's default is 5 receivers since 2026-09-24 (the user's choice).

## After milestone 14: the windows and the picks (2026-09-25)

Asked by the user after the pause: "the windows should be smaller 5 7 9 11, you are using too
bigger windows"; "that rule is not good [2 window lengths], I like the Nyquist limit ... we
should pick where we can at low frequencies and stop when it becomes weird ... around 10 Hz
maybe"; "maybe it is better to let the agent think and find out the best window size, depending
on the profile length". Each point brought as options with measurements on active_p1 (the
user's choice, and Claude's, each time), recorded in `docs/qc_workflow.md` ("Taken on
2026-09-25") and in `docs/gates/G3.md` and `S2_rules.md`:

- **The pick goes as far as its ridge holds, at both ends** (`paco.picking.modes`): the longest
  continuous run of kept points, a step steeper than |d ln v / d ln f| = 2 or over 2 Hz of
  weak columns ending it; and no point where a perfect plane wave for the window stands less
  than 1 % above its column's median (the window resolves no velocity there;
  `paco.picking.plane_waves`). The cut at 2 window lengths is gone from the picking's defaults
  and from G3 (`long_wavelengths`; `points_within` became `curve_points`); G3's
  `constant_wavelength` now cuts where its stretch starts.
- **G3 judges sharpness and prominence against a perfect plane wave for the same window**
  (`paco.quality.measuring`): on the demo both equal a plane wave's at every length, where the
  fixed limits failed every window of 5 to 11 receivers (a 5-receiver plane wave's prominence
  is 1.06). Prominence counted up to 4, so that long arrays keep the old limit (2).
- **The ladder: 5, 7, 9, 11, then 16, 24, 32 ...; it proposes, the agent decides.**
  `run_processing` returns `lengths`, every length tried (trial windows passing G3, the
  wavelengths their curves reach, windows on the line), one past the proposed one included;
  its `next` and the role say the length is the agent's to change for depth or lateral detail,
  saying why. A length given, by the user or the agent, is kept (the ladder climbed past it
  before: the agent's 7 receivers became 16 again).
- **G4's guide** for an outlier is thinned to 12 points: longer curves made it about 200, past
  the token budget of a tool result.
- **The evaluation**: `deeper` and `detail`, the agent choosing a longer or a shorter length
  than proposed; `zero_settings` lets it choose the length; `pick_active` checks G1's trace
  exclusion instead of the mode jump, gone with the picks above 50 Hz.

On the demo: the four 24-receiver windows every 24 all pass G3 and G4 (the mode jump of xmid
14.88 sat above the 50 Hz line, where the pick now stops); with no settings the ladder proposes
16 receivers (8/9 trials; 5/9 at 5, 6/9 at 7 and 9, 7/9 at 11, 9/9 at 24), and 79 of the
line's 81 curves pass G3 and G4. The report depths go deeper with the longer curves (2 to 10 m
where they were 1 to 4 m).

Also on 2026-09-25: **CI failed on `05cc650`** (see milestone 14) and **on `24e80b2`**,
committed while this work was half done (the new rules in, the demo's facts in the tests not
yet); the working tree was then checked as CI builds it (a fresh clone, 4 cores, no `.env`).

## Next

1. Run the evaluation again, with the 13 scenarios (the second of milestone 14 was stopped),
   record it and close the milestone; fix what it shows first.
2. The review of the "To judge" lists (the user's choice for after milestone 14), prepared on
   2026-09-24 and not yet brought: one round of four questions, below, then apply and prune the
   lists. Settled since, by the decisions of 2026-09-25: the tracked-point jumps, a length the
   user gave climbing (it no longer climbs), the mode jump of xmid 14.88 (gone); the inverse
   trend is now 1 of 4 windows at 24 receivers, 10 of 81 at 16, all below 50 Hz.
3. The decision to keep fixed pipelines (optional validated stages per preset, if wanted).

### Prepared for the To-judge review (2026-09-24)

Measured for it, on the dense line (active_p1, 24 receivers every 4, 19 windows):

- **Mode jumps on the tracked points** (G3's question): between consecutive kept columns
  (every 0.5 Hz) the good windows step by up to 47 % (at low frequencies and across gaps), the
  bad ones by 35 and 69 %: no separation. On the resampled curve G3 uses, 26 % at most against
  31 and 53 %. Settled: keep the resampled curve.
- **The 50 Hz mains line and the inverse trend** (G3's and G4's question; figure
  `docs/gates/figures/active_p1_dense_picks.png`). Every image has a dark column at 50 Hz: the
  hum is in phase on every geophone, so its peak sits at the grid's top velocity, and the pick
  breaks there. A notch would not change the image (the phase shift normalizes each trace's
  spectrum, so a filter common to all traces cancels). Below 48 Hz the curves are nearly flat
  (±7 %, within their 8-15 % uncertainties) and 5 windows stay inverse (3.88, 8.88-10.88,
  20.88) where 11 were; the points above 52 Hz sit 5 to 45 % faster, on branches bending
  upward, and make the models' thin stiff top (each of G5's four `smoothing_misfit` windows,
  2.88, 8.88, 12.88, 13.88, has one).
- **Why the ladder took 32 receivers for the user's 24** with vmax 250 m/s: its 24-receiver
  trials failed on prominence (a narrow grid raises each column's median, which lowers G3's
  prominence) and mode jumps; 1 of 9 on a length flag (sharpness). 32 passed 9 of 9.
- **Already decided in milestone 13**, to prune from the lists: G2's band (capped, reported),
  G5's misfit 2.0 and at-bound 10 %, G6's action for `non_unique` (longer sampling) and its
  15 %, S2's band, 9 trials at 80 %, and the near field (reported). G3's sharpness and
  uncertainty per window: the ladder now picks the length for the line.

The questions, Claude's picks marked:

1. **Loop changes** (several): G1's retries against a budget per record, not the windows' run
   budget (picked); a length the user gave climbs only when the trials fail on length flags
   (picked); a mode jump's first move is the band cut where the jump starts, the corridor
   halved second (picked: halving twice never fixed xmid 14.88, whose jump sits at 62-70 Hz).
2. **Gate changes** (several): G1 leaves the near field (under half the longest wavelength)
   out of its decay fit (picked: 2 of record 2's 3 excluded traces are the nearest); G4 says
   an inverse trend over the line once, kept (picked); G5 reports PAC's residual in % of the
   velocity, not judged (picked); G5's `no_mode` suggests a phase-shift fmax below the points
   faster than any mode, for the agent's `redo` (picked).
3. **The stiff top**: unsure, keep the flag and record that the trend rests on the points above
   50 Hz, re-judge on another line (picked); real, a stiff crust; not real, a G3 rule keeping
   the part of a pick below a gap when it resumes more than 15 % off.
4. **The rest kept as they are** (G1's 10 and 50 ms and no lower band edge, G2's 70 %, G3's
   prominence filter and 5 points, G4's 15 % and its budget-spent rejection, G6 without
   interface depths, S4's 0.8x, λmax/2 and one set of bounds) and the decided items pruned
   (picked); or each gate's kept items as options.

Still open from before: the judge model; the NVIDIA path of `compose.yaml`; committing the
burn-in fixes in `../sigpipe` and `../PAC`; sigpipe's `inversion_mcmc` failing when a chain
keeps no predicted curve (milestone 13; PACo now tries such a window once more). The guard
against repeating a failed call is built (milestone 14).
