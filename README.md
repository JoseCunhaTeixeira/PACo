# PACo

PACo lets a language model process MASW seismic profiles the way
[PAC](https://github.com/JoseCunhaTeixeira/PAC) does, from raw records to dispersion curves and
shear-wave velocity models, at your request in plain words. The science is
[sigpipe](https://github.com/JoseCunhaTeixeira/sigpipe)'s, shared with PAC, and the results are
written in PAC's layout, so PAC opens them. A quality gate checks each stage and fixes what it can.

PACo has two agents, on one model setup (`PACO_LLM_*`, any provider):

- **The guided workflow** (`guided`, the default): PACo's server leads the model step by step
  through its tools, and the model never sees the data, only the gates' short summaries. Made for
  a small model on your own GPU (Qwen3-14B served by vLLM).
- **The autonomous geophysicist** (`autonomous`, `paco.geo`): the model surveys the line, chooses
  the windows, the shots and the images, runs its own experiments and judges them with the gates,
  decides every window, freezes the curves and inverts, keeping a journal of its hypotheses,
  experiments and decisions. Made for a frontier model with a long context (128k tokens), at many
  more tokens a line. See [The autonomous geophysicist](#the-autonomous-geophysicist).

```
you  <-->  paco-agent  <-- the model's API -->  the model (vLLM, or any provider)
               |
               | MCP
               v
          paco-server  -->  sigpipe  -->  PACO_OUTPUT_DIR (PAC's layout)
```

- **paco-server**: an [MCP](https://modelcontextprotocol.io) server exposing the agent's tools
  (`--mode autonomous`: the geophysicist's).
- **paco-agent**: the chat in your terminal (`--mode autonomous`: with the geophysicist).
- **paco-evaluate**: plays a suite of scenarios with the guided agent and scores it; **paco-ab**
  plays both agents on the same scenarios and compares them.
- **paco-replay**: makes a run again from its inputs and its log, and compares the two.
- **paco-call**: calls one tool without the model.

In PAC, the AI assistant page runs the same agents on PAC's folders, its settings page choosing
the agent, the model and vision: [PAC's
README](https://github.com/JoseCunhaTeixeira/PAC#the-assistant-optional) says how to install it.

## Install

PACo needs Python 3.14 and [uv](https://docs.astral.sh/uv/), and a model: your own, behind an
OpenAI-compatible API such as vLLM's, or any provider's (see [The model](#the-model)).

```sh
git clone https://github.com/JoseCunhaTeixeira/PACo.git
cd PACo
uv sync                  # or: uv sync --extra petro, for the petrophysical inversion
                         # --extra vertex, --extra bedrock: Vertex AI's, Bedrock's credentials
```

## Data

PACo's repository holds no data. Each profile is a folder of `PACO_INPUT_DIR`, PAC's own input
folder for one (its demo profiles `active_p1` and `passive_p1` among them), and the results go to
`PACO_OUTPUT_DIR`, PAC's output folder for PAC to open them:

```
PAC/data/input/
  active_p2/
    101.dat, 102.dat, ...        # the records, in any format ObsPy reads (SEG-2 here)
    receiver_positions.yaml
    source_positions.yaml        # active profiles only
```

Each run goes to `PACO_OUTPUT_DIR/<profile>/<run_id>/`: PAC's `xmid_<x>/` window folders (images,
curves, models), `run.json` (what was processed, with which settings and files), and
`qc_log.jsonl`, every attempt of every stage with the gates' verdicts. Nothing is deleted: results
a stage done again replaces go to the window's `replaced/` folder.

## Configure

Settings come from environment variables or a `.env` file in the folder you start PACo from:

```sh
PACO_INPUT_DIR=../PAC/data/input
PACO_OUTPUT_DIR=../PAC/data/output
PACO_LLM_BASE_URL=http://gpu-host:8001/v1
PACO_LLM_MODEL=Qwen/Qwen3-14B-FP8
PACO_LLM_CONTEXT=16384
```

| Setting | Default | What it sets |
|---|---|---|
| `PACO_INPUT_DIR`, `PACO_OUTPUT_DIR` | `/data/input`, `/data/output` | Where the profiles are, and where results go |
| `PACO_AGENT_MODE` | `guided` | The agent `paco-agent`, `paco-server` and PAC's chat run: `guided` or `autonomous` |
| `PACO_LLM_PROVIDER` | `vllm` | Who serves the model (see [The model](#the-model)) |
| `PACO_LLM_MODEL` | required | The model's name, as its provider serves it |
| `PACO_LLM_BASE_URL` | the provider's | The model's API (e.g. `http://gpu-host:8001/v1`); required for `vllm`, `openai-compatible`, `azure-openai` and `anthropic-foundry` |
| `PACO_LLM_API_KEY` | `EMPTY` | Your key at the provider, or your server's if it asks for one |
| `PACO_LLM_REGION`, `PACO_LLM_PROJECT` | none | The cloud region (Vertex AI, Bedrock) and Google Cloud project (Vertex AI) |
| `PACO_LLM_CONTEXT` | `12288` on a server of yours, `131072` on a provider's API | The model's context, as its server serves it (vLLM's `--max-model-len`) |
| `PACO_LLM_VISION` | `false` | Whether figures may reach the model (the geophysicist's `look`) |
| `PACO_LLM_THINKING`, `PACO_LLM_EFFORT` | on, the model's | Whether the model thinks before its replies, and how hard |
| `PACO_LLM_TEMPERATURE`, `PACO_LLM_SEED` | the server's | The conversation's sampling, on a server of yours |
| `PACO_WORKERS` | `1` | Worker processes for processing and inversion |
| `PACO_HOST`, `PACO_PORT` | `127.0.0.1`, `8000` | Where `paco-server` listens |
| `PACO_ALLOWED_HOSTS` | none | Host names the server accepts, when it listens beyond 127.0.0.1 |
| `PACO_MCP_URL` | `http://127.0.0.1:8000/mcp` | Where the agent finds `paco-server` |
| `PACO_MAX_TOOL_CALLS` | `15` | Tool calls the model may make for one answer |
| `PACO_MAX_TURN_S`, `PACO_MAX_TURN_TOKENS` | `7200`, `40000` | Caps on one answer: its time, and the tokens the model writes |
| `PACO_TOOL_TIMEOUT_S` | `3600` | One tool call's timeout |
| `PACO_QC_CONFIG` | PACo's defaults | A JSON file of the gates' thresholds and retry budgets |
| `PACO_CACHE_DIR`, `PACO_CACHE_GB` | `<output>/.cache`, `2` | The cache of the windows' images, and its size (`0`: none) |
| `PACO_REPLAY_TOLERANCE` | `1e-5` | The difference `paco-replay` allows, relative to the run's largest value |
| `PACO_LOG_DIR` | `<output>/agent_logs` | Where the chat saves its conversations |
| `PACO_EVALUATION_DIR` | `<output>/evaluations` | Where evaluations write their reports |
| `PACO_JUDGE_MODEL`, `PACO_JUDGE_PROVIDER`, `PACO_JUDGE_BASE_URL`, `PACO_JUDGE_API_KEY`... | none | An optional judge model for the evaluations, any provider as the agents' model |
| `PACO_GEO_HOST`, `PACO_GEO_PORT`, `PACO_GEO_MCP_URL` | `127.0.0.1`, `8010` | Where the geophysicist's server listens, and where its chat finds it |

## Run

Start the server, then chat in another terminal:

```sh
uv run paco-server
uv run paco-agent
```

With the autonomous geophysicist, on a frontier model: `uv run paco-server --mode autonomous`,
then `uv run paco-agent --mode autonomous` (or `PACO_AGENT_MODE=autonomous` in `.env` for both).

A session looks like this (an illustration: the answer's wording depends on the model):

```
you> Process active_p1 with windows of 24 receivers, every 24 receivers, and pick the curves.
-> run_processing({"profile": "active_p1", "overrides": {"masw": {"length": 24, "step": 24}}})
   run_processing: 4 of 4 windows
-> pick({"run_id": "20260924-175047-f0a0"})

paco> The 4 curves passed G3 and G4. G1 left out traces 1, 89 and 90 of 2.dat.

Parameters used:
- mode active: the profile's kind
- MASW windows of 24 receivers (5.75 m), 4 windows, xmid 2.875 to 20.875 m: given (in your request, or chosen by the agent); its trial windows: 26/27 at 24 receivers
- shots 0 to 24.34 m from a window's middle: beyond it from the shot, the traces' median SNR falls under 2 dB (G1), so the windows stack no farther shot
- ...

Settings the gates changed:
- 2.dat: traces [1, 89, 90] left out of the windows
- ...
```

PACo processes a profile in PAC's three modes: active (shots), passive-active (interferometry on
the shots) and passive (ambient noise); ask for another mode in plain words. Given no window
length, it chooses the one whose curves are best, from trial windows. Like a run made in PAC's
pages, a run has one set of settings for its records and its images, the same for every window,
which PACo changes for the whole line when more windows then give a curve that passes; each
window's curve is picked on its own, and each curve inverted with settings of its own.

Other commands:
- `uv run paco-ab` plays both agents on the same scenarios, on the model of `.env`, and compares
  them (`--list` names the scenarios, `--smoke` tests the model's client in a few calls).
- `uv run paco-evaluate` plays the scenarios (5 times each) and reports each one's pass rate;
  its inversions are quick (the agent is measured, not the models), `--full-inversion` for
  PAC's effort; `--history` compares the evaluations kept, by model and prompts' version.
- `uv run paco-replay RUN_ID` makes a run again from its inputs and its log, and checks that the
  records, images and curves come out the same.
- `uv run paco-call TOOL '{...}'` calls one tool without the model; `--list` names the tools.
- `uv run paco-trace show FILE` prints a saved conversation; `replay FILE` plays it again through
  the current code.

## Tools

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

## The autonomous geophysicist

The geophysicist's tools (`paco-server --mode autonomous`) work on a profile's **study**, a folder
of `<output>/<profile>/studies/` that keeps everything: the survey, each experiment and its
trials, the line committed (a run in PAC's layout), the journal, the decisions, the frozen curves
and the inversions. Stopped, the next conversation on the profile resumes it.

| Tool | What it does |
|---|---|
| `survey` | The line before any setting: its geometry, its records as G1 measures them, the reach, the usable band, the surface waves' cone, the modes it takes |
| `try_settings` | Tries variants of the line's settings on chosen windows, judged by the gates, against the current settings |
| `commit` | Applies settings to the whole line, with their reasons: a run that PAC opens |
| `observe`, `look` | Reads the checks of the line committed; looks at a window's records, image, curve or model (a figure with vision on, tables and measures otherwise) |
| `advise` | PACo's own choice for a lever (the window length, the mute, a passive line's segments, the picking), not applied |
| `pick`, `decide`, `freeze` | Picks windows again, accepts or rejects each one with its reason, freezes the curves for the inversion |
| `invert`, `job_status` | Inverts the frozen curves, in the background |
| `journal` | Reads the journal; writes a hypothesis, a conclusion or a decision |

Code keeps the work honest: settings you type are kept and never changed, the phases go in order
(the survey, the curves, the inversion, the report), a window is rejected only once tried hard,
and the budgets stop a run that spends too much. The figures reach the model only with
`PACO_LLM_VISION=true`. Its plan is `docs/geo_plan.md`.

## How PACo works

- **The scope of your message.** Before any tool runs, the model reads your message into a
  form: the stages asked, the profile or run, the positions, the windows, the mode, the
  workers ("use 8 workers"). Code refuses calls outside it, and the answer starts with that
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

Both agents talk to the model `PACO_LLM_*` names, through its provider's own API:

| `PACO_LLM_PROVIDER` | The model | Credentials |
|---|---|---|
| `vllm` (default), `openai-compatible` | Your own server, at `PACO_LLM_BASE_URL` | Its key, if it asks for one |
| `anthropic` | Claude, on Anthropic's API | `PACO_LLM_API_KEY` |
| `anthropic-vertex`, `gemini-vertex` | Claude, Gemini, on Google Cloud's Vertex AI | The machine's Google credentials (`gcloud auth application-default login`; `--extra vertex`), `PACO_LLM_REGION`, `PACO_LLM_PROJECT` |
| `anthropic-bedrock` | Claude, on Amazon Bedrock | A Bedrock API key, or AWS's credentials (`--extra bedrock`); `PACO_LLM_REGION` |
| `anthropic-foundry`, `azure-openai` | Claude, OpenAI's models, on Microsoft's clouds | `PACO_LLM_API_KEY`; the resource's endpoint as `PACO_LLM_BASE_URL` |
| `openai`, `gemini`, `mistral` | OpenAI's, Google's, Mistral's models, on their APIs | `PACO_LLM_API_KEY` |

The guided workflow runs with **`Qwen/Qwen3-14B-FP8`**, which needs a GPU with 24 GB of memory;
the autonomous geophysicist needs a frontier model (Claude, GPT, Gemini) with 128k tokens of
context.

**On a GPU, with vLLM.** Qwen3 calls tools through vLLM's Hermes parser, and its thinking is kept
out of the answers by the Qwen3 reasoning parser:

```sh
vllm serve Qwen/Qwen3-14B-FP8 --port 8001 --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
```

Then `PACO_LLM_BASE_URL=http://127.0.0.1:8001/v1` (`paco-server` takes 8000),
`PACO_LLM_MODEL=Qwen/Qwen3-14B-FP8` and `PACO_LLM_CONTEXT=16384`. 32,768 tokens
(`--max-model-len 32768`) leave room for long conversations.

**On another machine's GPU** (a lab workstation, a GPU server rented in the cloud): the same
command there, its port reached through SSH from the machine PACo runs on
(`ssh -N -L 8001:127.0.0.1:8001 user@gpu-machine`), then
`PACO_LLM_BASE_URL=http://127.0.0.1:8001/v1`. The model stays invisible from the network. A
tunnel that restarts (SSH reconnecting, or `gcloud run services proxy` taking a fresh token every
55 minutes) is waited for: a connection refused or dropped is tried again for up to 30 s before
the answer fails.

**From an online service** serving Qwen3-14B behind an OpenAI-compatible API: its address,
`PACO_LLM_MODEL` as the service names the model, and `PACO_LLM_API_KEY`. The service then
receives your messages, the tools' descriptions and the gates' summaries, never the records,
images or models.

**Another model.** The guided workflow works with any model with tool calling and JSON-schema
output, on a provider above or behind an OpenAI-compatible chat API: on your own server, set
`PACO_LLM_BASE_URL`, `PACO_LLM_MODEL` and `PACO_LLM_CONTEXT`. In vLLM, a model of another family needs its own tool-call parser in place
of `hermes`, and no Qwen3 reasoning parser. PACo's tests are 15 scenarios of the assistant's
rules, played 5 times each (`uv run paco-evaluate`), on active_p1 when these were measured; the
suite now plays on active_p2:

| Model | GPU memory | PACo's tests (75 plays) |
|---|---|---|
| `Qwen/Qwen3-14B-FP8` (the one PACo runs) | 24 GB | 75 passed |
| `Qwen/Qwen3-8B-FP8` | 16 GB, with a 12,288-token context | 68 passed |
| `Qwen/Qwen3.8-27B` | 40 to 48 GB (FP8) | 69 passed |
| `Qwen/Qwen3-30B-A3B-FP8` | 48 GB, or two 24 GB GPUs | not tested |
| `Qwen/Qwen3-32B-FP8` | 48 GB, or two 24 GB GPUs | not tested |

## Docker

On a GPU machine with Docker and NVIDIA's container toolkit, `compose.yaml` runs the whole stack:
vLLM, `paco-server`, and the agent. `PACO_DATA_DIR` (PAC's data folder by default, `../PAC/data`)
is mounted as `/data`.

```sh
docker compose up -d vllm paco-server   # the first start downloads the model
docker compose run --rm agent           # chat in this terminal
docker compose run --rm evaluate        # the evaluation suite
```

`PACO_LLM_MODEL=Qwen/Qwen3-14B-FP8` is required in `.env`; `VLLM_MAX_MODEL_LEN` (default 16384),
`PACO_WORKERS` and `HF_TOKEN` can be set there too. The server and vLLM publish their ports
(8000 and 8001) on 127.0.0.1 only.

On an AMD GPU (ROCm: Radeon RX 9000 or 7900 series, Instinct), add `compose.rocm.yaml`, which
swaps in vLLM's ROCm image and reuses the models in `~/.cache/huggingface`:

```sh
docker compose -f compose.yaml -f compose.rocm.yaml up -d vllm paco-server
```

vLLM's 4-bit formats (AWQ, GPTQ) do not run on AMD GPUs: the model runs in FP8, on a 24 GB card.

With vLLM in Docker and PACo run with uv, point the agent at the container:
`PACO_LLM_BASE_URL=http://127.0.0.1:8001/v1`.

## Develop

```sh
uv run ruff check && uv run ruff format --check   # lint and format
uv run pyright src tests                          # types
uv run pytest                                     # tests, on PAC's local active_p2 and passive_p2
uv run pre-commit install                         # lint, format, types and fast tests on commit
```

The tests read PAC's local lines `active_p2` and `passive_p2` (`../PAC/data/input`, or the
folder `PACO_TEST_INPUT_DIR` names), kept in no repository; without them, the tests that read
them are skipped.

The tools' schemas are kept in `tests/snapshots/tool_schemas.json`: a change fails
`tests/test_server.py` until it is reviewed and written again with
`PACO_SNAPSHOT_UPDATE=1 uv run pytest tests/test_server.py -k snapshot`.

## License

CC BY 4.0 (see `LICENSE`).
