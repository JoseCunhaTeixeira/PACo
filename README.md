# PACo

PACo lets a language model process MASW seismic profiles the way
[PAC](https://github.com/JoseCunhaTeixeira/PAC) does: from raw records to dispersion curves and
layered shear-wave velocity models. The model (Qwen3, served by vLLM) chooses the steps; the
science is done by [sigpipe](https://github.com/JoseCunhaTeixeira/sigpipe) and its MASW layer,
`sigpipe.masw`, which PAC uses too; the results are written in PAC's layout, so PAC's UI can
open them. Each stage is checked by a quality gate that fixes what it can (G1 to G8, see
[Quality control](#quality-control)); the model never sees the data, only the gates' short
summaries, and asks you only when it is stuck.

```
you  <-->  paco-agent  <-- OpenAI API -->  vLLM (Qwen3)
               |
               | MCP (Streamable HTTP)
               v
          paco-server  -->  sigpipe  -->  data/output (PAC's layout)
```

- **paco-server** is an [MCP](https://modelcontextprotocol.io) server: it exposes PACo's tools.
- **paco-agent** is the chat in your terminal: it passes the tools to the model and runs the
  calls the model asks for. When the data cannot decide (no curve left, a request the line does
  not allow), the model asks you, with a few options, in its answer.
- **paco-evaluate** plays a suite of scenarios with the model, and scores it.

PACo holds the agent, the MCP server, the quality gates (their thresholds, retries and log) and
the evaluation. Profiles, windows, the processing settings, runs, picks, the inversion and the
quality measures are sigpipe's (`sigpipe.masw`), shared with PAC.

## In PAC

PAC, the web application, has an Assistant page that runs PACo's agent: installed with PAC's
`agent` extra, the agent calls PACo's tools inside PAC, on PAC's folders, and its runs open in
PAC's pages. [PAC's README](https://github.com/JoseCunhaTeixeira/PAC#the-assistant-optional)
says how to install it, with the model on your own GPU or on another machine's. The rest of this
README is PACo on its own: its server, its chat in a terminal, and its evaluation.

## Install

PACo needs Python 3.14 and [uv](https://docs.astral.sh/uv/). The agent and the evaluation also
need a Qwen3 model behind an OpenAI-compatible API, such as vLLM (see [vLLM](#vllm)).

```sh
git clone https://github.com/JoseCunhaTeixeira/PACo.git
cd PACo
uv sync                  # or: uv sync --extra petro, for the petrophysical inversion
```

## Data

Each profile is a folder of `PACO_INPUT_DIR`:

```
data/input/
  active_p1/
    1.dat, 2.dat                 # the records, in any format ObsPy reads (SEG-2 here)
    receiver_positions.yaml
    source_positions.yaml        # active profiles only
  passive_p1/
    ...
```

The two demo profiles, `active_p1` and `passive_p1`, come with the repository. Results go to
`PACO_OUTPUT_DIR/<profile>/<run_id>/`: PAC's `xmid_<x>/` folders (dispersion images, picked
curves, inversion models; earlier attempts under `attempts/`), `records/` (the preprocessed
records), and the run's records: `run.json` (what was processed, with which settings),
`qc_log.jsonl` (every attempt of every stage, with the gates' verdicts), `qc_report.json`,
`qc_config.json` (the thresholds used), `coherence.json` (how the window length was chosen) and
`inversion.json` (the inversion job).

## Configure

Settings come from environment variables or from a `.env` file in the folder you start PACo
from. For the demo, with vLLM on a GPU machine called `gpu-host`:

```sh
PACO_INPUT_DIR=data/input
PACO_LLM_BASE_URL=http://gpu-host:8001/v1
PACO_LLM_MODEL=Qwen/Qwen3-8B
```

| Setting | Default | What it sets |
|---|---|---|
| `PACO_INPUT_DIR` | `/data/input` | Where the profiles are |
| `PACO_OUTPUT_DIR` | `data/output` | Where results go |
| `PACO_WORKERS` | `1` | Windows processed, or inverted, in parallel (the evaluation uses 8) |
| `PACO_QC_CONFIG` | PACo's defaults | A JSON file of the gates' thresholds and retry budgets (`paco.qc.QCConfig`), changed between runs only |
| `PACO_HOST`, `PACO_PORT` | `127.0.0.1`, `8000` | Where `paco-server` listens |
| `PACO_ALLOWED_HOSTS` | none | Host names the server accepts, e.g. `["paco-server:*"]`; needed when it listens beyond 127.0.0.1 |
| `PACO_LLM_BASE_URL` | required | The model's OpenAI-compatible API, e.g. `http://gpu-host:8001/v1` |
| `PACO_LLM_MODEL` | required | The model's name, as vLLM serves it |
| `PACO_LLM_API_KEY` | `EMPTY` | Only if vLLM was started with `--api-key` |
| `PACO_MCP_URL` | `http://127.0.0.1:8000/mcp` | Where the agent finds `paco-server` |
| `PACO_MAX_TOOL_CALLS` | `15` | Tool calls the model may make for one answer |
| `PACO_LOG_DIR` | `data/output/agent_logs` | Where the chat saves its conversations |
| `PACO_JUDGE_BASE_URL`, `PACO_JUDGE_MODEL`, `PACO_JUDGE_API_KEY` | none | An optional judge model for the evaluation |
| `PACO_EVALUATION_DIR` | `data/output/evaluations` | Where evaluations write their reports |

## Run

In one terminal, start the server:

```sh
uv run paco-server
```

In another, chat with the agent:

```sh
uv run paco-agent
```

A session looks like this (an illustration: the answer's wording depends on the model):

```
you> Process active_p1 with windows of 24 receivers, every 24 receivers, and pick the curves.
-> run_processing({"profile": "active_p1", "overrides": {"masw": {"length": 24, "step": 24}}})
   run_processing: 4 of 4 windows
-> pick({"run_id": "20260924-175047-f0a0"})

paco> The 4 curves passed G3 and G4. G1 corrected the records' 19 and 10 ms trigger delays and
left out traces 1, 89 and 90 of 2.dat.

Settings the gates changed:
- trigger t0 the default -> 0.0188 at 1.dat, 2.dat (each window its own), by G1:shifted_trigger
- 2.dat: traces [1, 89, 90] left out of the windows
- ...
```

Whatever the model writes, PACo's host (`src/paco/agent/host.py`) lists the settings the gates
changed after the answer, as the tools gave them; once the work has started, asks the model
once more for an answer without a question or an offer, unless a tool said it is stuck (a
question before any work, such as an impossible request, reaches you); and refuses `invert` (and a
`redo` of the inversion) unless your message asks for models (invert, inversion, model, Vs,
shear), and `invert_petro` unless it asks for soils or the water table (soil, sol, water table,
nappe, N value, SPT, clay, sand, silt, loam, petro...).

PACo processes a profile in PAC's three modes: an active profile as shots (`active`, the
default) or by interferometry on its shots (`passive-active`: each shot's surface waves
cross-correlated with the receiver nearest the shot, the correlations stacked), a passive
profile as ambient noise (`passive`). Ask for another mode in plain words ("process active_p1
in passive-active mode"); `inspect_profile` lists a profile's modes. Two defaults differ from
PAC's forms, measured on real lines: passive records are cut into 2 s segments, whitened and
normalized one-bit (PAC: 0.1 s, neither; no curve on a real noise line), and passive-active
correlates each shot's surface-wave window only (`correlation_window`, which can be switched
off).

Given no window length, `run_processing` proposes one: the shortest at which most trial windows
along the line give a curve G3 passes, with a table of every length it tried (trial windows
passed, the wavelengths their curves reach, windows on the line). The model keeps it, or runs
again with another length when the request asks for more depth (longer windows) or lateral
detail (shorter), and says why; a length you or the model give is kept as it is.

The conversation is saved when you leave (`exit`). To evaluate the model, run
`uv run paco-evaluate`, or name scenarios: `uv run paco-evaluate list_profiles pick_active`.
The model samples its answers, so one play of a scenario is a noisy measure: `--repeat 3` plays
each scenario three times and reports pass rates.

## Tools

| Tool | What it does |
|---|---|
| `list_profiles` | The profiles you can process |
| `inspect_profile` | One profile: active or passive, the modes it can be processed in, receivers, spacing, sampling |
| `preset_settings` | The processing settings the model may change, for one profile and mode |
| `run_processing` | Preprocesses the records (G1, its fixes applied), proposes a window length from trial windows unless given (the lengths tried listed for the model to choose from) and caps the band to the data, makes one dispersion image per window (G2, retried when it can be fixed); returns a `run_id` and the gates' summary |
| `pick` | Picks each window's fundamental mode (G3, picked again when it can be fixed), judges the curves over the line (G4, outliers picked again), saves them in PAC's layout |
| `inversion_settings` | The inversion's parameters; left out, each window's bounds come from its own curve |
| `invert` | Inverts the curves G4 passed, in the background (G5 on each model, G6 over the line, each retrying what it can), then writes the line's velocity section and the picked curves against the predicted ones, as PAC does (`SeismicInversion_VelocitySection_0000.png` and `.hdf5`, `SeismicInversion_PseudoSectionComparison_0000_M0.png`, in the run folder, over the models G5 passed); returns a `job_id` |
| `job_status` | Where an inversion stands, after waiting up to 2 minutes: the smooth median models' Vs at a few depths, the depth their curves inform them down to (half the longest wavelength) and their misfit; the gates' summary once it ends |
| `petro_models` | Only when you ask for soils or the water table: the Silex models of the petrophysical inversion, what each was trained on, and how many of the run's curves it covers |
| `invert_petro` | Inverts the curves G4 passed that the chosen model covers into soils, N values and the water table (G7 on each, G8 over the line), then writes PAC's petrophysical sections over the models both passed; needs the `petro` extra |
| `redo` | Goes back to a stage (preprocessing, phase shift, picking, inversion) for some windows, with the changes a gate suggested, and redoes what follows |

## Quality control

Eight gates judge the stages (`docs/qc_workflow.md`, the spec, with its decisions): G1 each
record, G2 each dispersion image, G3 each curve, G4 the curves over the line, G5 each model, G6
the models over the line, G7 each petrophysical model, G8 those over the line. Each gives a verdict (pass, retry, reject), each metric with its threshold, and for
each flag the stage at fault and a change that can be applied as it is. The stage tools apply
their own gate's changes, within budgets (2 retries per gate and window, 2 per window over the
run); a change of an earlier stage is the model's to make, with `redo`. The band (the records'
median usable band), the farthest shot a window stacks (where the traces' median SNR falls
under 2 dB) and the inversion's bounds come from the data, the window length is proposed from
it, for the model to choose (`docs/gates/S2_rules.md`, `docs/gates/S4_checks.md`). Each pick
goes as far as its ridge holds, at both ends, and G3 judges sharpness and prominence against a
perfect plane wave for the same window, so that short windows are judged fairly
(`docs/gates/G3.md`). An inversion has at least 3 layers and starts at 4; G5 adds layers up to
what each curve resolves. `docs/gates/` documents every gate with its thresholds, the demo's
real outputs, and what is still to judge.

The petrophysical inversion runs only when you ask for soils or the water table: the host
refuses it otherwise, as it refuses an inversion you did not ask for. A Silex model predicts
each window's soil column from its curve; it applies only to curves within the band and
velocities it was trained on (the bundled one: 15 to 50 Hz, so curves reaching 43 Hz), and the
others are left out with the reason. G7 judges how well each column gives its curve back (G5's
limit), and G8 compares the columns' Vs and water tables with their neighbours
(`docs/gates/G7.md`, `G8.md`).

## Safety

- `paco-server` listens on 127.0.0.1 by default: only this machine can reach it. Anyone who can
  reach it can run PACo's tools, which write files and start long computations.
- No tool asks you anything: when your message asks for models, an inversion starts for the
  curves G4 passed, the go or no-go before it; when it asks for soils or the water table, the
  petrophysical inversion does. Every attempt is in the run's `qc_log.jsonl` and
  `qc_report.json`, with the curve each model came from: review the curves afterwards
  (`DispersionImage_0000.png` in each window folder, or PAC's UI), correct them there if
  needed, and invert again.
- The gates' thresholds are locked during a run: only you change them, between runs
  (`PACO_QC_CONFIG`); every run records the ones it used.
- Profiles are found by name and runs by ID: a tool never takes a path from the model.

## vLLM

Qwen3 calls tools through vLLM's Hermes parser, and its thinking is kept out of the answers by
the Qwen3 reasoning parser:

```sh
vllm serve Qwen/Qwen3-8B --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
```

## Docker

On a GPU machine with Docker and NVIDIA's container toolkit, `compose.yaml` runs the whole stack:
vLLM, `paco-server`, and the agent. `./data` is mounted as `/data`.

```sh
docker compose up -d vllm paco-server   # the first start downloads the model
docker compose run --rm agent           # chat in this terminal
docker compose run --rm evaluate        # the evaluation suite
```

`PACO_LLM_MODEL`, `VLLM_MAX_MODEL_LEN` (default 16384), `PACO_WORKERS` and `HF_TOKEN` can be set
in `.env`. The server and vLLM publish their ports (8000 and 8001) on 127.0.0.1 only.

On an AMD GPU (ROCm: Radeon RX 9000 or 7900 series, Instinct), add `compose.rocm.yaml`, which
swaps in vLLM's ROCm image and reuses the models in `~/.cache/huggingface`:

```sh
docker compose -f compose.yaml -f compose.rocm.yaml up -d vllm paco-server
```

vLLM's 4-bit formats (AWQ, GPTQ) do not run on AMD GPUs, and Qwen3-8B needs about 16.4 GB for
its weights. On a 16 GB card, both of these run (tried on a Radeon RX 9070 XT):

- `PACO_LLM_MODEL=Qwen/Qwen3-4B` (bf16, 7.6 GiB of weights);
- `PACO_LLM_MODEL=Qwen/Qwen3-8B-FP8` with `VLLM_MAX_MODEL_LEN=12288`: the weights take 8.8 GiB
  and leave 2.2 GiB for the KV cache, enough for a 12k context, not for 16k.

With vLLM in Docker and PACo run with uv, point the agent at the container:
`PACO_LLM_BASE_URL=http://127.0.0.1:8001/v1`.

## Develop

```sh
uv run ruff check && uv run ruff format --check   # lint and format
uv run pyright src tests                          # types
uv run pytest                                     # tests, on the demo profiles
```

GitHub Actions runs the same checks on every push. `PROGRESS.md` records how PACo was built,
the decisions taken, and the open issues.

## License

CC BY 4.0 (see `LICENSE`).
