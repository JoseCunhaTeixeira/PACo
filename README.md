# PACo

PACo lets a language model process MASW seismic profiles the way
[PAC](https://github.com/JoseCunhaTeixeira/PAC) does: from raw records to dispersion curves and
layered shear-wave velocity models. The model (Qwen3, served by vLLM) chooses the steps; the
science is done by [sigpipe](https://github.com/JoseCunhaTeixeira/sigpipe), and the results are
written in PAC's layout, so PAC's UI can open them. The model never sees the data, only short
summaries, and an inversion starts only after you approve the curves.

```
you  <-->  paco-agent  <-- OpenAI API -->  vLLM (Qwen3)
               |
               | MCP (Streamable HTTP)
               v
          paco-server  -->  sigpipe  -->  data/output (PAC's layout)
```

- **paco-server** is an [MCP](https://modelcontextprotocol.io) server: it exposes PACo's tools.
- **paco-agent** is the chat in your terminal: it passes the tools to the model, runs the calls
  the model asks for, and asks you when a tool needs your approval.
- **paco-evaluate** plays a suite of scenarios with the model, and scores it.

## Install

PACo needs Python 3.14 and [uv](https://docs.astral.sh/uv/). The agent and the evaluation also
need a Qwen3 model behind an OpenAI-compatible API, such as vLLM (see [vLLM](#vllm)).

```sh
git clone https://github.com/JoseCunhaTeixeira/PACo.git
cd PACo
uv sync
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
curves, inversion models), plus `run.json`, `quality.json`, `pick.json` and `inversion.json`,
which record what was done and with which settings.

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
| `PACO_WORKERS` | `1` | Windows processed, or inverted, in parallel |
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
you> Process active_p1 with windows of 24 receivers, every 24 receivers, and tell me how many are good.
-> run_processing({"profile": "active_p1", "overrides": {"masw": {"length": 24, "step": 24}}})
   run_processing: 4 of 4 windows
-> dispersion_quality({"run_id": "20260923-142918-38a0"})

paco> All 4 windows are good.
```

The conversation is saved when you leave (`exit`). To evaluate the model, run
`uv run paco-evaluate`, or name scenarios: `uv run paco-evaluate list_profiles judge_active`.

## Tools

| Tool | What it does |
|---|---|
| `list_profiles` | The profiles you can process |
| `inspect_profile` | One profile: active or passive, receivers, spacing, sampling |
| `preset_settings` | The processing settings the model may change, for one profile |
| `run_processing` | Processes a profile into one dispersion image per MASW window; returns a `run_id` |
| `quality_settings` | The picking parameters and quality thresholds the model may change |
| `dispersion_quality` | Picks each window's fundamental mode and judges it: good, doubtful or bad, with advice |
| `pick` | Saves the curves of the good windows, in PAC's layout, for you to review |
| `inversion_settings` | The inversion's parameters (PAC's defaults: 2 layers, 100,000 iterations, 5 chains) |
| `invert` | Asks you to approve the curves, then inverts them in the background; returns a `job_id` |
| `job_status` | Where an inversion stands: windows done, velocities and depths found so far |

## Safety

- `paco-server` listens on 127.0.0.1 by default: only this machine can reach it. Anyone who can
  reach it can run PACo's tools, which write files and start long computations.
- The model never approves an inversion: the question goes to your terminal. Review the curves
  first (`DispersionImage_0000.png` in each window folder, or PAC's UI), and correct them in
  PAC's UI if needed.
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
its weights: on a 16 GB card, set `PACO_LLM_MODEL=Qwen/Qwen3-4B`, or try an FP8 model such as
`Qwen/Qwen3-8B-FP8`. With vLLM in Docker and PACo run with uv, point the agent at the container:
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
