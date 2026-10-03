# PACo

PACo lets a language model process MASW seismic profiles the way
[PAC](https://github.com/JoseCunhaTeixeira/PAC) does: from raw records to dispersion curves and
layered shear-wave velocity models. The model (Qwen3, served by vLLM) chooses the steps; the
science is done by [sigpipe](https://github.com/JoseCunhaTeixeira/sigpipe) and its MASW layer,
`sigpipe.masw`, which PAC uses too; the results are written in PAC's layout, so PAC's UI can
open them. Each stage is checked by a quality gate that fixes what it can (G1 to G8, see
[Quality control](#quality-control)); the model never sees the data, only the gates' short
summaries, and asks you when your request leaves a choice open or when it is stuck.

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
- **paco-replay** makes a run again from its inputs and its log, and compares the two.
- **paco-call** calls one of the tools without the model.

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
need a model behind an OpenAI-compatible API: Qwen3-14B, served by vLLM on a GPU or by an
online service (see [The model](#the-model)).

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
curves, inversion models; earlier attempts under `attempts/`, results a stage done afresh
replaced under `replaced/`), `records/` (the preprocessed records), and the run's records:
`run.json` (what was processed, with which settings, and the profile's files it read, by name,
size and SHA-256), `qc_log.jsonl` (every attempt of every stage, with the gates' verdicts and
what made each; the settings you gave), `qc_report.json`, `qc_config.json` (the thresholds
used), `coherence.json` (how the window length was chosen) and `inversion.json` (the inversion
job).

The run's state is kept so that nothing is lost and one writer works on it at a time:
- The QC log only grows. Each line has its version, its kind (a stage run, a gate's verdict,
  notes, the settings given, a reset) and who logged it (the agent, a gate, the user). A stage
  done afresh (asked again, or redone in PAC) appends a reset: the attempts before it stay in
  the log, and their results are moved to the window's `replaced/<time>_<stage>/`, never
  deleted.
- Each attempt says what made it: PACo's and sigpipe's versions, and for an agent's call its
  model, the prompts' version, the conversation and the turn.
- The state (each window's attempts and verdicts, whose each result is) is rebuilt from the log
  and the window folders whenever it is read (`qc.rebuild_state`): a curve picked in PAC shows
  at once. `qc_report.json` is a snapshot of it.
- JSON files are written whole or not at all (a temporary file, renamed). PACo holds a run
  while a tool or an inversion job writes it, and PAC's pages hold it while they write
  (`.run.lock`): PAC is refused a run PACo writes (its message says so), and PACo a run PAC
  writes (the model is told to try again when it ends).
- A window's image is kept in the images' cache under a key made of the window, the settings,
  the records it reads (their content) and the code's version; a call that makes the same image
  takes it from there (`cached` in `run.json`'s windows). `paco-replay` never does.
- `uv run paco-replay RUN_ID` makes the run again from its inputs and its QC log, in
  `PACO_OUTPUT_DIR/replays/` (`--keep` keeps it), and compares it with the run: its records,
  images and curves, the same within `PACO_REPLAY_TOLERANCE` of the largest value. It refuses
  a run whose inputs changed. What a person made in PAC and the inversions (a random search
  without a seed) are not replayed, and it says so.

## Configure

Settings come from environment variables or from a `.env` file in the folder you start PACo
from. For the demo, with vLLM on a GPU machine called `gpu-host`:

```sh
PACO_INPUT_DIR=data/input
PACO_LLM_BASE_URL=http://gpu-host:8001/v1
PACO_LLM_MODEL=Qwen/Qwen3-14B-FP8
PACO_LLM_CONTEXT=16384
```

| Setting | Default | What it sets |
|---|---|---|
| `PACO_INPUT_DIR` | `/data/input` | Where the profiles are |
| `PACO_OUTPUT_DIR` | `data/output` | Where results go |
| `PACO_WORKERS` | `1` | Worker processes: records preprocessed, windows imaged, picked and inverted in parallel; when fewer windows than workers are inverted (the gates' retries), each window's chains share the idle cores (the evaluation uses 8, PAC's assistant half the cores) |
| `PACO_CACHE_DIR`, `PACO_CACHE_GB` | `<output dir>/.cache`, `2` | Where the windows' images are kept to be taken again (made by the same code from the same records with the same settings: a profile processed again, the trials of a line), and how many GB at most, those used longest ago removed first; `0` keeps none |
| `PACO_REPLAY_TOLERANCE` | `1e-5` | The largest difference `paco-replay` allows between a run and its replay, relative to the run's largest value (records and images are float32: 1.2e-7 apart at best) |
| `PACO_QC_CONFIG` | PACo's defaults | A JSON file of the gates' thresholds and retry budgets (`paco.qc.QCConfig`), changed between runs only |
| `PACO_HOST`, `PACO_PORT` | `127.0.0.1`, `8000` | Where `paco-server` listens |
| `PACO_ALLOWED_HOSTS` | none | Host names the server accepts, e.g. `["paco-server:*"]`; needed when it listens beyond 127.0.0.1 |
| `PACO_LLM_BASE_URL` | required | The model's OpenAI-compatible API, e.g. `http://gpu-host:8001/v1` |
| `PACO_LLM_MODEL` | required | The model's name, as vLLM serves it |
| `PACO_LLM_API_KEY` | `EMPTY` | Only if vLLM was started with `--api-key` |
| `PACO_LLM_TEMPERATURE`, `PACO_LLM_SEED` | the server's | The conversation's sampling (Qwen3's own is 0.6: it repeats itself colder when it thinks); the scope's form is filled at 0 |
| `PACO_MCP_URL` | `http://127.0.0.1:8000/mcp` | Where the agent finds `paco-server` |
| `PACO_MAX_TOOL_CALLS` | `15` | Tool calls the model may make for one answer |
| `PACO_MAX_TURN_S`, `PACO_MAX_TURN_TOKENS` | `7200`, `40000` | The caps on one answer: its time, and the tokens the model writes (its thinking included); an answer that reaches one ends with what was done and the cap named |
| `PACO_TOOL_TIMEOUT_S` | `3600` | One tool call's timeout (a whole line inverted at PAC's effort takes about 40 min on 8 cores) |
| `PACO_LLM_CONTEXT` | `12288` | The model's context, as its server serves it (vLLM's `--max-model-len`): the loop warns past 85% |
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

Parameters used:
- mode active: the profile's kind
- MASW windows of 24 receivers (5.75 m), 4 windows, xmid 2.875 to 20.875 m: given (in your request, or chosen by the agent); its trial windows: 26/27 at 24 receivers
- shots 0 to 24.34 m from a window's middle: beyond it from the shot, the traces' median SNR falls under 2 dB (G1), so the windows stack no farther shot
- ...

Settings the gates changed:
- trigger t0 the default -> 0.0188 at 1.dat, 2.dat (each window its own), by G1:shifted_trigger
- 2.dat: traces [1, 89, 90] left out of the windows
- ...
```

Whatever the model writes, PACo's host (`src/paco/agent/host.py`) lists the parameters each
stage ran with and why (given, a rule on the data, a gate, or a default: the window length,
the phase shift's band, the picker's settings, the inversions' layers, bounds and effort) and
the settings the gates changed after the answer, as the tools gave them, and follows an
inversion the model starts until it ends, its progress shown, before the model reads on. The
host watches no words, neither yours nor the model's: what to do, and when to ask, the model
decides from your request, and the tools apply the rules below.

PACo processes a profile in PAC's three modes: an active profile as shots (`active`, the
default) or by interferometry on its shots (`passive-active`: each shot's surface waves
cross-correlated with the receiver nearest the shot, the correlations stacked), a passive
profile as ambient noise (`passive`). Ask for another mode in plain words ("process active_p1
in passive-active mode"); `inspect` lists a profile's modes. Passive records are cut into
1 s segments end to end, FK-selected (0.2, always on), whitened and normalized one-bit, and their
correlations stacked phase-weighted (power 2), as PAC's forms do. A passive-active shot
is correlated whole; when its image peaks at the grid's top velocity, G2 asks for the
surface-wave mute (80 to 1,500 m/s) before a wider grid, the records' time origin moved by the
trigger their files state.

Given no window length, `run_processing` chooses one: the shortest at which most trial windows
along the line give a curve G3 passes, or a longer one while it makes the picks more precise,
with a table of every length it tried (trial windows passed, the wavelengths their curves
reach, the picks' precision, windows on the line). The best curves come first: the model keeps
that length whatever depth or lateral detail the request asks, and says down to which depth
its curves reach; a length you give is kept as it is.

The conversation is saved when you leave (`exit`). To evaluate the model, run
`uv run paco-evaluate`, or name scenarios: `uv run paco-evaluate list_profiles pick_active`.
The model samples its answers, so one play of a scenario is a noisy measure: it plays each
scenario five times (`--repeat N` to change it) and reports each scenario's pass rate against
its threshold (3 plays of 5 by default). `uv run paco-evaluate --history` tabulates the pass
rates of every evaluation kept, by model and prompts' version, the latest version against the
one before (`--model` for one model), and over the same plays the tool calls that failed, the
calls refused as outside the message's scope, the answers a cap ended and each gate's retries a
play. An evaluation's plays share one images' cache, removed at its end. A play the model's
server cuts short (out of reach, failing) is lost, not counted, and listed under the report; a
request it refuses (a JSON schema it does not take, the context exceeded) fails the play, with
the server's message.

`uv run paco-call TOOL '{...}'` calls one tool without the model, as the chat would (its run's
`agent_calls.jsonl` names the conversation `paco-call`); `uv run paco-call --list` names the
tools, each with an example call.

## Tools

| Tool | What it does |
|---|---|
| `inspect` | Reads what exists, changing nothing: the profiles, one profile (active or passive, the modes it can be processed in, receivers, spacing, sampling), the runs (who made each, what it holds), one run (its settings, then its windows, those alike grouped: who made each image, curve and model, the gates' verdicts, the models' misfits and depths informed), one window in detail |
| `preset_settings` | The processing settings the model may change, for one profile and mode |
| `run_processing` | Preprocesses the records (G1, its fixes applied; each record's gather and spectra drawn beside it, `Stream_0000.png` and `Spectrum_0000.png`), proposes a window length from trial windows unless given (the lengths tried listed for the model to choose from), tries mutes around the gathers' surface waves at that length and keeps one only where the images gain (unless you gave a muting), caps the band to the data, makes one dispersion image per window (G2, retried when it can be fixed); returns a `run_id` and the gates' summary |
| `compare` | Compares 2 to 4 sets of processing settings on a sample of the line's windows, on the depth reached, the band, the curves' length or the windows passing G3, changing no run: a table and the best |
| `pick` | Picks each window's fundamental mode (G3, picked again when it can be fixed), judges the curves over the line (G4, outliers picked again), saves them in PAC's layout; for some windows given by their positions (m), with picking settings you gave; curves already there, or picked by hand, ask you first |
| `judge` | Judges curves as they are, picking nothing: G3 on the automatic curves no gate judged (PAC's Auto-pick), or on some windows, then G4 over the line |
| `inversion_settings` | The inversion's parameters; left out, each window's bounds come from its own curve |
| `invert` | Inverts the curves G3 and G4 passed and those picked by hand (taken as they are), every mode picked in each window, in the background (automatic curves no gate judged are judged first; given positions, those windows only, again if the assistant inverted them; G5 on each model, G6 over the line, each retrying what it can), then writes the line's velocity section and the picked curves against the predicted ones, as PAC does and shows them (`SeismicInversion_VelocitySection_0000.png`: Vs, its uncertainty and the interfaces, the depth informed veiled; `_lateralsmooth` smoothed along the line; `.hdf5`; `SeismicInversion_PseudoSectionComparison_0000_M0.png` by frequency, `_wavelength` by wavelength; in the run folder, over the models G5 passed); returns the job's status, which PACo's host follows to the end |
| `job_status` | Where an inversion stands, after waiting up to 2 minutes and reporting the windows done: the smooth median models' Vs at a few depths, the depth the data inform them down to (G5's depth informed) and their misfit; the gates' summary once it ends |
| `petro_models` | Only when you ask for soils or the water table: the Silex models of the petrophysical inversion, what each was trained on, and how many of the run's curves it covers |
| `invert_petro` | Inverts the curves G4 passed that the chosen model covers into soils, N values and the water table (G7 on each, G8 over the line), then writes PAC's petrophysical sections over the models both passed, as PAC shows them (soils and N, the rock physics, each also smoothed along the line; the picked curves against the models', by frequency and by wavelength); needs the `petro` extra |
| `redo` | Goes back to a stage (preprocessing, phase shift, picking, inversion) for some windows, with the changes a gate suggested, and redoes what follows |

## The scope of each message

Before any tool runs, the model reads your message into a form, its scope: the stages it asks
(process, pick, invert, soils; none for a question about what exists, or a request no tool
makes), the profile or run, the positions in metres, the windows' length and step in the unit
you give them (receivers or metres), the window lengths a comparison names, whether to do again
work already there, what it says of your hand work,
and which option it chooses among those a tool offered last. The form's JSON schema constrains
the model's output, thinking off; a form that does not parse goes back once with its error,
and a second failure ends the answer asking you to say it again. Code then checks what it can
(a run id PACo never gives, an option never offered, a negative position) and keeps the turn
within the scope (`src/paco/agent/scope.py`):
- a call outside it is refused, unmade, with the reason for the model; an earlier stage stays
  within a scope that asks a later one, the server deciding whether the run needs it;
- each call carries the scope, and the tools apply the rules below with it; the positions a
  tool works at are those your message named (none: the whole line), and the windows and the
  lengths compared those it gave (metres converted by PACo), whatever the model wrote;
- the answer starts with the scope's line (`Scope: pick, invert · active_p1 · at 9 m.`), for
  you to check what PACo read;
- the options a tool offered are kept, so that "the second one" is the call it offered; once a
  tool offered options, the answer runs nothing more but reading (inspect, the settings), and
  ends with them: the choice is yours.

Every answer is written by code around the model's text (`src/paco/agent/answer.py`):
- the scope's line;
- what the model says: its draft put into an answer form, constrained so that the text holds
  no question;
- "Done:" and "Left out:", from the tools' results (what each stage did, the windows a gate
  rejected or a stage failed, with why);
- the question and the options when you must choose (a tool offered options, a tool cannot go
  on without you, or your request for work left it unclear), else "Next:", what you can ask
  next and where PAC shows the results; a message asking for no work (a look, something no
  tool makes) gets an answer, not a question back;
- "Parameters used", "Settings the gates changed" and "Settings kept as you gave them", last.

A number in the model's text that no result of the turn holds is flagged after it.

The loop keeps itself in check (`src/paco/agent/loop.py`):
- With each message naming a profile, the host tells the model the profile's latest run, read
  from the server's resource `paco://profiles/{profile}/latest-run`: the model makes no run id
  up.
- A call made already in the answer is refused; a third time, the answer ends.
- Tool results reach the model in a delimited data block (`<data from="pick">...</data>`), which
  its role says is never an instruction: a profile's or a file's name that reads as one is data.
- Tool results of earlier messages are kept short in the conversation (what each did, the
  windows left out, the options); the transcript keeps them whole.
- Every tool is declared read-only or changing (MCP's `readOnlyHint`, `destructiveHint`).
- A call that changes a run leaves a line in the run's `agent_calls.jsonl`: the conversation,
  the message's turn and scope, the prompts' version, the arguments, how it ended and how long
  it took. The conversation's transcript, in `PACO_LOG_DIR`, names the same conversation.
- `paco-trace show FILE` prints a saved conversation turn by turn; `paco-trace replay FILE`
  plays it again through the current code with its recorded replies and results (no tool
  runs), and says where the answers differ.
- Log lines are JSON on stderr, each with the conversation, turn and run it belongs to.

Tools return their status (`ok`, `partial`, `refused`, `stuck`), what they did in one line and
the windows they left out; their errors are tagged with their kind (`[bad argument]`,
`[precondition]`, `[stuck]`, `[retry]`).

The settings you give are locked for the run (events of the run's QC log, `src/paco/qc/given.py`):
no gate and no check changes them. A window whose gate asks to change one is left out, its line
saying the change asked (`G2 locked, asks dispersion vmax 900 (given: 250)`), for you to choose;
an inversion bound you gave that the curve does not fit stays as given, its note saying what the
check asks (`kept as given (the check sets 450 m/s)`).

The prompts are files (`src/paco/prompts/`: the role, the scope's and the answer's forms with
their examples, the server's instructions); their version is logged with each turn and in the
evaluation's report, and
`tests/test_prompts.py` pins it, so that a change to a prompt is reviewed with the scenarios run
on it.

## Work already there, and work made in PAC's pages

A run goes records, images, curves, models (seismic, then soil columns). PACo goes on from
what is there and asks before redoing it: with nothing, it does everything; asked to pick and
invert a profile whose run has images, it picks and inverts them; asked to process it, it says
the run is there and asks whether to process again (a new run, the old one kept) or go on from
it; asked to invert a run with curves, it inverts them; asked to pick it again, or to process
it, it asks whether to redo the picking, complete the windows without a curve, invert the
curves as they are, or work on some windows; models and soil columns the same way. The tools
make these checks themselves: one that meets such work does nothing, says what is there and
gives the options, each with the call it makes (`again`, `windows="all"` or `"missing"`,
positions, which narrow `windows`). With the scope of your message the rules are applied in
code: a stage your message does not ask goes on from the run's work without a question (asked
to invert, `run_processing` and `pick` point to the run's curves); one it asks whose work is
there gives the options, unless your message asks to do it again; `again` and
`windows="all"` hold only when your message asked for them, in its words or by the option it
chose. What PACo did earlier in the same conversation it goes on with, without asking (the
host sends the conversation's id, turn and scope with each call).

What you make in PAC's pages is yours, verified by you: no gate judges it, nothing automatic
changes it (the gates' retries, the fixes of earlier stages), and the inversion takes it as it
is. A curve you pick or change in Dispersion picking (any mode; each mode apart: adding M1
leaves PACo's M0 and its checks as they were), a run you process in Active, Passive or
Passive-active (its images, which PACo may pick), a model you make in Seismic inversion, a soil
column you make in Petrophysical inversion. A step you ask for that would change it asks first
whether to keep it (`hand="keep"`) or replace it (`hand="replace"`: yours set aside in the
window's `by_hand/` folder). It is replaced only when your message asks it, in its words ("my
hand-picked curve too") or by choosing the replace option the question offered, and kept
without the question only when you chose the keep option: keep or replace is yours to say, not
the model's. A client that sends no scope gets the same question, and a replace only once the
question was shown in an earlier turn of the conversation, over those windows, each answer
serving one step. PAC's own Auto-pick is automatic: PACo judges it (`judge`, or `invert`
first) before inverting it. G4 compares PACo's curves with yours, never yours with
PACo's; a line picked by hand has no G4. The rule for who made a curve is sigpipe's
(`masw.runs.origin`), which PAC reads too.

PACo picks M0 alone: a higher mode is picked by hand, in Dispersion picking, and PACo inverts
it with the rest. In PAC's chat, an answer that worked on a run ends with the run, the time it
took and links to it (Pick the curves, Review, See the models).

## Quality control

Eight gates judge the stages (`docs/qc_workflow.md`, the spec, with its design decisions): G1
each record, G2 each dispersion image, G3 each curve, G4 the curves over the line, G5 each
model, G6 the models over the line, G7 each petrophysical model, G8 those over the line. Each
gives a verdict (pass, retry, reject), each metric with its threshold, and for each flag the
stage at fault and a change that can be applied as it is. The stage tools apply their own
gate's changes, within budgets (2 retries per gate and window, 2 per window over the run, and 6
inversion retries per window of its own); `pick` makes the change of an earlier stage G2 or G3
asks for, once, and any other change of an earlier stage is the model's to make, with `redo`.
The band (the records' median usable band), the farthest shot a window stacks (where the
traces' median SNR falls under 2 dB), the nearest (half the longest wavelength its trial curves
reach) and the inversion's bounds come from the data; the window length is proposed from it,
for the model to choose (`docs/gates/S2_rules.md`, `docs/gates/S4_checks.md`). Each pick
goes as far as its ridge holds, at both ends, and G3 judges sharpness and prominence against a
perfect plane wave for the same window, so that short windows are judged fairly
(`docs/gates/G3.md`). An inversion lets the data choose the number of layers (up to 8, Vs 100
to 2,000 m/s widened where the curve needs it, interfaces from a third of its shortest
wavelength to half its longest), the picks' noise level sampled with the model; G5 judges the
chains on the models' Vs at the depths the curve resolves, and adapts each window: sampling
longer, a Vs bound widened, more layers allowed (with the layers given: a layer added or two
alike merged, the depth shrunk to what the data inform) (`docs/gates/G5.md`). Each window's
parameters are in its `SeismicInversion_Parameters_0000.json`. `docs/gates/` documents every
gate with its thresholds, the demo's real outputs, and what is still to judge.

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
- PACo runs only the stages your message asks (its scope), and asks you before it redoes work
  already there or changes work you made by hand: when your message asks for models, an
  inversion starts for the curves G4 passed; when it asks for soils or the water table, the
  petrophysical inversion does. Every attempt is in the run's `qc_log.jsonl` (append-only, with
  what made it) and `qc_report.json`, with the curve each model came from: review the curves
  afterwards (`DispersionImage_0000.png` in each window folder, or PAC's UI), correct them
  there if needed, and invert again.
- Nothing is deleted: results a step done again replaces are kept in the window's `replaced/`
  folder, and work you made by hand that you choose to replace in its `by_hand/` folder.
- What the tools return, and the names and texts of your files, reach the model as data, never
  as instructions.
- The gates' thresholds are locked during a run: only you change them, between runs
  (`PACO_QC_CONFIG`); every run records the ones it used.
- Profiles are found by name and runs by ID: a tool never takes a path from the model.

## The model

PACo runs with **`Qwen/Qwen3-14B-FP8`**, which needs a GPU with 24 GB of memory.

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
`PACO_LLM_BASE_URL=http://127.0.0.1:8001/v1`. The model stays invisible from the network.

**From an online service** serving Qwen3-14B behind an OpenAI-compatible API: its address,
`PACO_LLM_MODEL` as the service names the model, and `PACO_LLM_API_KEY`. The service then
receives your messages, the tools' descriptions and the gates' summaries, never the records,
images or models.

## Docker

On a GPU machine with Docker and NVIDIA's container toolkit, `compose.yaml` runs the whole stack:
vLLM, `paco-server`, and the agent. `./data` is mounted as `/data`.

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
uv run pytest                                     # tests, on the demo profiles
```

Each tool has an example call and result in `src/paco/examples.py`, checked against the tool's
arguments and what it returns (`tests/test_examples.py`); they stay out of the model's prompts,
whose values a model copies. `pre-commit install` adds the hooks: lint, format, the type check
and the fast tests (those processing no demo data).

The tools' schemas (their descriptions, arguments, annotations and results, what the model and
PAC read) are kept in `tests/data/tool_schemas.json` with their version: a change fails
`tests/test_server.py` until it is reviewed and the snapshot written again with
`PACO_SNAPSHOT_UPDATE=1 uv run pytest tests/test_server.py -k snapshot`.

## License

CC BY 4.0 (see `LICENSE`).
