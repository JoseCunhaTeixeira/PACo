# PACo progress

Last updated 2026-09-23. Read this first at the start of each session.

## Working agreement

- Tutor mode, from the rules pasted in the first session (not saved as `CLAUDE.md` yet).
- Amended 2026-09-23: Claude writes code only when explicitly asked, and only what is asked.
  Design decisions still go through options first.
- Repos: sigpipe in `../sigpipe` (GitHub HEAD 403fea2), PAC in `../PAC` (372357c, plus one
  uncommitted README change). PACo installs sigpipe from git; `uv.lock` pins 403fea2.

## Status

| Milestone | Status |
|---|---|
| 0. Orientation | Done, with gaps (see open issues) |
| 1. Plain functions | Done: `inspect_profile`, `run_processing` |
| 2. Schemas | Done: generated presets, messages for the agent, checks before any work, schema budget; 173 tests |
| 3. Quality metric and picking | Next. Picking is designed (see decisions) |
| 4 to 8 | Not started |

Check questions: milestone 1 asked, answer pending. Milestone 2 (what happens when sigpipe renames
a parameter): answered with hints on 2026-09-23, up to the plain-words rung. Worth revisiting:
why generated models fail at import while hand-written ones fail in every window.

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
    data without losing visible ridge; 1 x L lost 17-33 Hz of clean active ridge.
  - A mode is judged on its kept points; a point is kept if not pinned and at or above the noise
    floor 1/sqrt(N); a mode needs 5 kept points with a median coherence >= 1.5 x the floor.
  - The code lives in `paco/picking/` first, and moves to sigpipe once proven.
  - Output: `DispersionCurves_0000.csv` in PAC's layout (still to build).
- **AI picker: later.** Discussed 2026-09-23: training a model (image in, M0 velocity per
  frequency out) needs labelled pairs, and none are at hand. Chosen route: keep the rule-based
  picker and the quality metric, and let human-approved picks (milestone 6) accumulate as training
  data (PAC's layout already pairs `DispersionImage_0000.hdf5` with `DispersionCurves_0000.csv`).
  A trained model can later replace the picker behind the same `pick` tool.
- **Inversion (milestone 5):** an inversion preset built from PAC's form defaults (2 layers, Vs
  100 to 1000 m/s, thicknesses 1 to 10 m, 100k iterations, 10k burn-in, 5 chains), run as a
  background job, one process per window.

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
- **For milestone 4:** the passive override schema (about 1,800 tokens) is a large share of an 8k
  context; the MCP tools should send one preset's schema at a time, not both.
- **Picking risks:**
  - the lowest-ridge scan can stop on the air wave (about 340 m/s) in active data on stiff soils;
  - with no wavelength limit, M0 keeps unresolved low-frequency points (active: below about
    13 Hz, climbing to 700 m/s), and resampling over wavelength turns them into long curves;
  - on passive_p1 (24-receiver windows, preset defaults) the images have no clear ridge: M0
    follows the edge of the broad bright region, yet its coherence is about 2 x the floor, so
    coherence alone cannot reject it. The quality metric must.
- **PAC's defaults:** its form defaults give 3-receiver windows, which make flat passive images
  with nothing to pick.

## Next

Milestone 3: automatic picking (designed above), then the quality metric that the picker's
diagnostics feed (`dispersion_quality`), designed with the user as the domain expert.
