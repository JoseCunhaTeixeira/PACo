# PACo

PACo lets a language model process MASW seismic profiles the way
[PAC](https://github.com/JoseCunhaTeixeira/PAC) does: from raw records to dispersion curves and
shear-wave velocity models, asked in plain words. The science is
[sigpipe](https://github.com/JoseCunhaTeixeira/sigpipe)'s, shared with PAC. Results are written in
PAC's layout, so PAC opens them. A quality gate checks each stage and fixes what it can.

PACo works in one of two modes, with the same model:

- **workflow** (the default): cheaper, but its results are less good. PACo leads the model step
  by step through its tools and chooses the settings by fixed rules. The model never sees the
  data, only the gates' short summaries. A small model on your own GPU is enough (Qwen3-14B).
- **autonomous**: better results, but costly. PACo tries every combination of the line's
  settings, the model looks at all the images and keeps the cleanest, then decides each window
  and inverts. It keeps a journal of what it tried and why. It needs a frontier model (Claude,
  GPT, Gemini) with vision and a long context, and uses many more tokens. See
  [The autonomous mode](#the-autonomous-mode).

`PACO_AGENT_MODE` chooses the mode (or `--mode` on the commands).

```
you  <-->  paco-agent  <-- the model's API -->  the model
               |
               | MCP
               v
          paco-server  -->  sigpipe  -->  PACO_OUTPUT_DIR (PAC's layout)
```

- **paco-server**: an [MCP](https://modelcontextprotocol.io) server with the mode's tools.
- **paco-agent**: the chat, in your terminal.
- **paco-evaluate**, **paco-ab**: test the agent on scenarios (see [Other commands](#other-commands)).

In PAC, the AI assistant page runs the same agent on PAC's folders, and its settings popup chooses
the mode and the model. [PAC's
README](https://github.com/JoseCunhaTeixeira/PAC#the-assistant-optional) says how to install it.

## Install

PACo needs Python 3.14, [uv](https://docs.astral.sh/uv/) and a model (see [The model](#the-model)).

```sh
git clone https://github.com/JoseCunhaTeixeira/PACo.git
cd PACo
uv sync                  # or: uv sync --extra petro, for the petrophysical inversion
```

## Data

PACo's repository holds no data. It reads the profiles in PAC's input folder (`PACO_INPUT_DIR`)
and writes its results in PAC's output folder (`PACO_OUTPUT_DIR`), so PAC opens them. A profile is
a folder:

```
PAC/data/input/
  active_test_profile/
    101.dat, 102.dat, ...        # the records, in any format ObsPy reads (SEG-2 here)
    receiver_positions.yaml
    source_positions.yaml        # active profiles only
```

Each run goes to `PACO_OUTPUT_DIR/<profile>/<run_id>/`: PAC's `xmid_<x>/` window folders (images,
curves, models), `run.json` (what was processed, with which settings and files), and
`qc_log.jsonl` (every attempt of every stage, with the gates' verdicts). Nothing is deleted:
results replaced by a stage done again go to the window's `replaced/` folder.

## Configure

PACo reads its settings from the environment, or from a `.env` file in the folder you start it
from. For example, with your own GPU server:

```sh
PACO_INPUT_DIR=../PAC/data/input
PACO_OUTPUT_DIR=../PAC/data/output
PACO_AGENT_MODE=workflow
PACO_LLM_PROVIDER=server
PACO_LLM_BASE_URL=http://127.0.0.1:8001/v1
PACO_LLM_MODEL=Qwen/Qwen3-14B-FP8
PACO_LLM_CONTEXT=16384
```

| Setting | Default | What it sets |
|---|---|---|
| `PACO_INPUT_DIR` | `/data/input` | Where the profiles are |
| `PACO_OUTPUT_DIR` | `/data/output` | Where the results go |
| `PACO_AGENT_MODE` | `workflow` | `workflow` or `autonomous` |
| `PACO_LLM_PROVIDER` | `server` | Who serves the model (see [The model](#the-model)) |
| `PACO_LLM_BASE_URL` | the provider's | The model's address. Needed for your own server |
| `PACO_LLM_MODEL` | required | The model's name |
| `PACO_LLM_API_KEY` | none | Your API key, if the provider asks for one |
| `PACO_LLM_CONTEXT` | `16384` on your server, `131072` on a provider's API | The model's context, in tokens. With vLLM, the same number as its `--max-model-len` |
| `PACO_WORKERS` | `1` | Processes used to process and invert |

Rarely needed: `PACO_HOST` and `PACO_PORT` (where `paco-server` listens, `127.0.0.1:8000`),
`PACO_MCP_URL` (where the chat finds it), `PACO_ALLOWED_HOSTS` (host names the server accepts
when it listens beyond this machine), `PACO_QC_CONFIG` (a JSON file of the gates' thresholds) and
`PACO_CACHE_GB` (the size of the images' cache, 2 GB; 0 for none).

## Run

Start the server, then chat in another terminal:

```sh
uv run paco-server
uv run paco-agent
```

Both use `PACO_AGENT_MODE`. To use the other mode once, add `--mode autonomous` (or
`--mode workflow`) to both commands.

A session looks like this (an illustration: the answer's wording depends on the model):

```
you> Process active_test_profile with windows of 24 receivers, every 24 receivers, and pick the curves.
-> run_processing({"profile": "active_test_profile", "overrides": {"masw": {"length": 24, "step": 24}}})
   run_processing: 4 of 4 windows
-> pick({"run_id": "20261018-144613-9de0"})

paco> 3 of the 4 windows have a curve that passed G3 and G4. At xmid 89.25 m, competing ridges
left no curve to pick. G1 left out 3 shots, and a mute made the images cleaner.

Left out:
- xmid 89.25 (1): G3 nothing_to_try, on_data

Parameters used:
- mode active: the profile's kind
- MASW windows of 24 receivers (34.5 m), 4 windows, xmid 17.25 to 125.25 m: given (in your request, or chosen by the agent); its trial windows: 21/27 at 24 receivers (picks 7%)
- muting 80 to 1500 m/s: the mute trial (the gathers' surface waves 125.3 to 332.2 m/s, median 202.5): the standard mute 80 to 1500 m/s, 14 of 15 trial windows of 24 receivers (34.5 m) passing G3, against 11 without a mute
- ...

Settings the gates changed:
- 101.dat, 129.dat, 133.dat: left out of every window
- ...
```

PACo processes the three kinds of lines PAC does: active (shots), passive-active
(interferometry on the shots) and passive (ambient noise). It takes the profile's kind; ask for
another in plain words. Given no window length, it tries a few and keeps the one whose curves are
best. As in PAC, a run has one set of settings for the whole line, and each window's curve is
picked and inverted on its own.

### Other commands

- `uv run paco-evaluate` plays the workflow's test scenarios, 5 times each, and gives each one's
  pass rate. Its inversions are quick (`--full-inversion` for PAC's). `--history` compares past
  evaluations.
- `uv run paco-ab` plays both modes on the same scenarios and compares them. `--list` names the
  scenarios; `--smoke` checks the model in a few calls.
- `uv run paco-replay RUN_ID` makes a run again from its inputs and its log, and checks that the
  results are the same.
- `uv run paco-call TOOL '{...}'` calls one tool without the model. `--list` names the tools.
- `uv run paco-trace show FILE` prints a saved conversation; `replay FILE` plays it again through
  the current code.

The evaluations use the model of `.env`, which also judges the answers.

## The workflow mode

PACo chooses the settings by fixed rules: trials of the window length, the mute and a passive
line's segments, then the gates judge each stage and fix what they can. It is fast and cheap, but
fixed rules miss settings a full search finds: on PAC's test lines its images hold their ridge to
about 50 Hz, where the best settings hold it to 80 to 100 Hz.

PACo's server leads the model through these tools:

| Tool | What it does |
|---|---|
| `inspect` | Reads what exists, changing nothing: the profiles, the runs, a run's windows, one window |
| `preset_settings` | The processing settings the model may change, for a profile and mode |
| `run_processing` | Processes a profile into one dispersion image per window: records checked and fixed (G1), window length chosen from trials unless given, a mute kept where it helps, images judged (G2) |
| `compare` | Compares 2 to 4 sets of settings on a sample of windows (depth, band, curve length, windows passing), changing no run |
| `pick` | Picks each window's fundamental mode (G3) and judges the curves along the line (G4) |
| `judge` | Judges curves already there (G3, G4), picking nothing |
| `inversion_settings` | The inversion's parameters |
| `invert` | Inverts the curves that passed into Vs models, in the background (G5, G6), and draws the line's velocity section |
| `job_status` | Where an inversion stands |
| `petro_models` | The petrophysical models, and how many of the run's curves each covers (only when you ask for soils) |
| `invert_petro` | Inverts the covered curves into soils, N values and the water table (G7, G8); needs the `petro` extra |
| `redo` | Redoes a stage for some windows, with a gate's suggested change, and what follows |

## The autonomous mode

How a line goes:

1. **Survey.** The line's geometry, its records checked by G1, the shots' reach, the usable band,
   the surface waves' velocities.
2. **The full grid, in code.** Every combination of the values the physics sets for each setting
   of PAC's page: the window length, the shots, the mute, a passive line's segments, FK
   selection, normalization and stacking. Each variant is processed on 12 windows along the line
   and checked by the gates. The values are in `src/paco/autonomous/grid.yaml`, and `paco-grid`
   prints the grid's size.
3. **The model's eye.** Only the really bad variants are dropped: no image, or no curve along the
   line. The model looks at all the others, in sheets of 12, each variant at the same 4 places
   along the line, with no score: no score tells a clean image as the eye does. It keeps the
   cleanest of each sheet: one ridge, sharp and unbroken at every place, the picks on it, over a
   wide band.
   The kept meet again until a last sheet, whose kept are compared at full size. Each sheet's
   judgment is saved in the study.
4. **The line.** The model refines its choice, applies it to the whole line, picks and decides
   each window, freezes the curves, inverts them and reports.

It costs more than the workflow mode, for better results: about 25 to 45 minutes and $3 to $7 a
line with Claude Opus and 11 workers, before the inversion (estimates from PAC's test lines). The
grid takes longer with fewer workers.

The tools work on a profile's **study**: a folder in `<output>/<profile>/studies/`
that keeps everything (the survey, each experiment and its trials, the line committed as a run PAC
opens, the journal, the decisions, the frozen curves and the inversions). If a conversation stops,
the next one on the profile resumes the study. A message names its profile; without one, PACo
lists the profiles.

| Tool | What it does |
|---|---|
| `survey` | The line before any setting: its geometry, its records as G1 measures them, the reach, the usable band, the surface waves' cone, the modes it takes |
| `search` | Runs the full grid of the line's settings in code, drops the really bad variants, and shows the others in sheets of images to judge by eye |
| `judge` | Keeps the cleanest variants of each sheet, round after round; the last sheet's are tried side by side at full size |
| `try_settings` | Refines around the choice: variants of the line's settings on chosen windows, judged by the gates. Waits, like `commit`, until every sheet is judged |
| `commit` | Applies settings to the whole line, with their reasons: a run that PAC opens |
| `observe`, `look` | Reads the checks of the line committed; looks at a window's records, image, curve or model |
| `advise` | PACo's own choice for a lever (the window length, the mute, a passive line's segments, the picking), not applied |
| `pick`, `decide`, `freeze` | Picks windows again, accepts or rejects each one with its reason, freezes the curves for the inversion |
| `invert`, `job_status` | Inverts the frozen curves, in the background |
| `petro_models`, `invert_petro` | Only when you ask for soils: the petrophysical model that covers the most frozen curves, then their soils, N values and water table (G7, G8); needs the `petro` extra |
| `journal` | Reads the journal; writes a hypothesis, a conclusion or a decision |

Code keeps the work honest. The settings you type are never changed. The phases go in order:
survey, curves, inversion, report. A window is rejected only after real tries. Budgets stop a run
that spends too much; judging the sheets does not count against them. The model sees the
figures: the autonomous mode judges every image by eye, so it needs a model that takes images.
The design is in `docs/autonomous_plan.md`.

## How PACo works

- **The scope of your message.** Before any tool runs, the model reads your message into a
  form: the stages asked, the profile or run, the positions, the windows, the processing mode,
  the workers ("use 8 workers"). Code refuses calls outside it, and the answer starts with that
  line, for you to check what PACo read.
- **The run.** Code settles it before the model acts: the run you name (or says it does not
  exist, with the runs there); asked to process a profile that has runs, whether to make a new
  run or which run to work on; asked only later stages, the run the conversation is on, else
  which run when the profile has several. The model then gets one plan on that run.
- **Work already there.** PACo goes on from a run's work, and asks before redoing it (the
  windows without a curve, everything again); the options wait until you choose, and your
  choice goes on with what you asked.
- **Your work in PAC.** Curves, runs and models you make in PAC's pages are yours: no gate judges
  or changes them, and PACo asks before replacing one (the old one kept in `by_hand/`).
- **Your settings.** Settings you give are kept for the run: no gate changes them; a window that
  would need it is left out, saying what its gate asks.
- **Quality gates.** G1 to G8 judge each stage (records, images, curves, models, soils, each one
  and along the line), and fix what they can within budgets; every attempt is in the run's
  `qc_log.jsonl`.
- **Answers.** Code writes the fixed parts around the model's text: the scope, what was done, the
  windows left out and why, the stages your message asked that were not done, the parameters
  used and the settings the gates changed.

The details are in `docs/agent.md`, the gates' specification in `docs/qc_workflow.md`, and each
gate in `docs/gates/`.

## Safety

- `paco-server` listens on 127.0.0.1 by default: only this machine can reach it. Anyone who
  reaches it can run PACo's tools, which write files and start long computations.
- PACo runs only the stages your message asks, and asks before redoing work already there or
  changing work you made by hand.
- Nothing is deleted: replaced results go to `replaced/`, replaced hand work to `by_hand/`.
- Tool results and your files' names reach the model as data, never as instructions; profiles
  are found by name and runs by ID, never by a path the model gives.
- The gates' thresholds change only between runs (`PACO_QC_CONFIG`); every run records its own.

## The model

Both modes use the same model, set by `PACO_LLM_*`. There are two ways to reach it:

| Where the model is | `PACO_LLM_PROVIDER` | What to set |
|---|---|---|
| A provider's API | `anthropic` (Claude), `openai`, `gemini` or `mistral` | `PACO_LLM_MODEL`, `PACO_LLM_API_KEY` |
| Your server, or any server speaking OpenAI's API: vLLM, Ollama, LM Studio, OpenRouter, a gateway | `server` (the default) | `PACO_LLM_BASE_URL`, `PACO_LLM_MODEL`, `PACO_LLM_CONTEXT`, and `PACO_LLM_API_KEY` if it asks for one |

The workflow mode runs well with `Qwen/Qwen3-14B-FP8`, on a GPU with 24 GB of memory. The
autonomous mode needs a frontier model with vision and 128k tokens of context.

**Your GPU, with vLLM.** Serve Qwen3 with its tool-call and reasoning parsers:

```sh
vllm serve Qwen/Qwen3-14B-FP8 --port 8001 --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
```

Then set `PACO_LLM_BASE_URL=http://127.0.0.1:8001/v1`, `PACO_LLM_MODEL=Qwen/Qwen3-14B-FP8` and
`PACO_LLM_CONTEXT=16384`, the same number as `--max-model-len` (port 8000 is `paco-server`'s).
32768 tokens leave room for longer conversations.

**A GPU on another machine.** Run the same command there, and reach it through SSH:
`ssh -N -L 8001:127.0.0.1:8001 user@gpu-machine`. The model stays off the network. If the tunnel
drops, PACo tries again for up to 30 s.

**What a provider receives.** Your messages, the tools' descriptions and what the tools return:
the gates' summaries in the workflow mode, the line's measures and tables in the autonomous mode.
In the autonomous mode, also the figures it looks at: the mode judges every image by eye.

**Other models.** The workflow mode works with any model that calls tools and fills a JSON schema.
With vLLM, a model of another family needs its own `--tool-call-parser`, and no
`--reasoning-parser qwen3`. PACo's 15 test scenarios, played 5 times each (`uv run paco-evaluate`,
on PAC's former demo line when measured):

| Model | GPU memory | PACo's tests (75 plays) |
|---|---|---|
| `Qwen/Qwen3-14B-FP8` (the one PACo runs) | 24 GB | 75 passed |
| `Qwen/Qwen3-8B-FP8` | 16 GB, with a 12,288-token context | 68 passed |
| `Qwen/Qwen3.8-27B` | 40 to 48 GB (FP8) | 69 passed |
| `Qwen/Qwen3-30B-A3B-FP8` | 48 GB, or two 24 GB GPUs | not tested |
| `Qwen/Qwen3-32B-FP8` | 48 GB, or two 24 GB GPUs | not tested |

## Docker

On a GPU machine with Docker and NVIDIA's container toolkit, `compose.yaml` runs vLLM,
`paco-server` and the chat. PAC's data folder (`../PAC/data`, or `PACO_DATA_DIR`) is mounted as
`/data`.

```sh
docker compose up -d vllm paco-server   # the first start downloads the model
docker compose run --rm agent           # chat in this terminal
docker compose run --rm evaluate        # the evaluation suite
```

Set `PACO_LLM_MODEL=Qwen/Qwen3-14B-FP8` in `.env`. `PACO_LLM_CONTEXT` (16384 by default) sets the
context of both vLLM and PACo. `PACO_WORKERS` and `HF_TOKEN` can go there too. The ports (8000 for
the server, 8001 for vLLM) are open to this machine only.

On an AMD GPU (ROCm: Radeon RX 9000 or 7900 series, Instinct), add `compose.rocm.yaml`. It uses
vLLM's ROCm image and the models already in `~/.cache/huggingface`:

```sh
docker compose -f compose.yaml -f compose.rocm.yaml up -d vllm paco-server
```

The 4-bit formats (AWQ, GPTQ) do not run on AMD GPUs: the model runs in FP8, on a 24 GB card.

With vLLM in Docker and PACo run with uv: `PACO_LLM_BASE_URL=http://127.0.0.1:8001/v1`.

## Develop

```sh
uv run ruff check && uv run ruff format --check   # lint and format
uv run pyright src tests                          # types
uv run pytest                                     # the fast tests, in under a minute
uv run pytest --slow                              # all the tests, the slow ones too (about 50 min)
uv run pre-commit install                         # lint, format, types and fast tests on commit
```

The slow tests (marked `slow`) run PACo's processing, picking and inversion of the test profiles:
sigpipe's work, which sigpipe tests itself. `uv run pytest` leaves them out; run them with
`--slow` before a release. The tests read PAC's test profiles, `active_test_profile` and
`passive_test_profile` (in PAC's `data/input`, beside this folder, or the folder
`PACO_TEST_INPUT_DIR` names); without them, the tests that read them are skipped.

The tools' schemas are kept in `tests/snapshots/tool_schemas.json`: a change fails
`tests/test_server.py` until it is reviewed and written again with
`PACO_SNAPSHOT_UPDATE=1 uv run pytest tests/test_server.py -k snapshot`.

## License

CC BY 4.0 (see `LICENSE`).
