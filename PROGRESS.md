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
| 1. Plain functions | Done: `inspect_profile`, `run_processing`, 134 tests |
| 2. Schemas | Next |
| 3. Quality metric and picking | Picking designed (see decisions) |
| 4 to 8 | Not started |

Milestone 1 check question: asked 2026-09-23, answer pending.

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
- `presets/`:
  - `ActivePreset` and `PassivePreset` are PAC's config models, with PAC's form defaults.
  - `make_preset(name, overrides)` applies the agent's overrides.
  - `resolve_preset(preset, profile)` fills in the derived values.
- `pipelines/`: a port of PAC's adapters. `build_pipeline(preset, window, output_folder)`
  refuses presets that were not resolved.
- `runs/`: `run_processing(profile, preset, overrides, settings, on_progress)` returns a
  `RunSummary` and writes `<output_dir>/<profile>/<run_id>/`: PAC's `xmid_<x>/` folders, plus a
  `run.json` manifest.

Checks against PAC:

- **Windows:** identical in 12 cases. The reference is `tests/data/pac_windows.json`.
- **Presets:** resolved presets validate as PAC configs.
- **Pipelines:** identical steps, and bit-identical outputs on demo windows, even though PAC's
  lockfile pins sigpipe 31274d8 and PACo's pins 403fea2.
- **Tests:** 134 tests in about 12 s. Each feature was also checked with deliberately broken
  copies of the code.

## Decisions (2026-09-23)

- **PAC's role (option A):** PACo depends on sigpipe only; PAC is the reference and the source of
  expected test results.
- **Layout:** one module per feature, split into sub-packages (models, logic, summary). Domain
  code never imports `mcp` or `openai`.
- **Records:** any format obspy reads. The demo profiles are SEG-2 `.dat` files.
- **Profile model:** `Profile` and `Record` built on sigpipe types, instead of PAC's parallel
  lists.
- **Presets:**
  - config models, like PAC's;
  - values from PAC's forms (`distance_max` changed to 1000 m);
  - derived values as in PAC, except the IIR filter `fmax`, set to 0.95 × Nyquist;
  - a preset's mode must match the profile's type;
  - short field names (`filtering`, not `filtering_params`);
  - the number of workers is a server setting, not part of presets.
- **Runs:**
  - timestamp run IDs in UTC, like `20260923-142501-a3f9`;
  - PAC-shaped folders, grouped by profile;
  - a process pool whose workers run in the run folder, so sigpipe's `logs/` folder and logger
    reset stay inside the run.
- **Picking (to build at the start of milestone 3):** automatic, adapted from malw-pipe's picker
  (`../malw-pipe/docs/picking.md`).
  - The fundamental mode is the lowest ridge, scanned upwards from the aliasing floor.
  - Viterbi tracking runs inside a corridor, with the `on_edge` and coherence-to-floor
    diagnostics.
  - It picks M0, then the higher modes.
  - The code lives in `paco/picking/` first, and moves to sigpipe once proven.
  - Output: `DispersionCurves_0000.csv` in PAC's layout.
- **Inversion (milestone 5):** an inversion preset built from PAC's form defaults (2 layers, Vs
  100 to 1000 m/s, thicknesses 1 to 10 m, 100k iterations, 10k burn-in, 5 chains), run as a
  background job, one process per window.

## Concepts covered (to confirm with the check question)

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
- **For milestone 2:**
  - the passive preset's JSON schema is about 8,200 characters (about 2,000 tokens), so a compact
    override schema is needed;
  - pydantic's error messages need rewording for the model;
  - sigpipe has whitening methods PAC doesn't expose (`savgol`, `stft_*`).
- **Picking risks:**
  - the lowest-ridge scan can stop on the air wave (about 340 m/s) in active data on stiff soils;
  - Viterbi costs O(n_f × n_v²), about 1 to 2 s per window.
- **PAC's defaults:** its form defaults give 3-receiver windows, which make flat passive images
  with nothing to pick.

## Next

Milestone 2, schemas: Pydantic models for presets and overrides, generated from sigpipe's
registries, with rejection messages written for the model. Picking starts milestone 3.
