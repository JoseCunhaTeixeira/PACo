# How PACo's agent works

What the README sums up, in full: the run's state, the scope of each message, how answers are
written, the loop's checks, work already there and work made in PAC's pages, and the quality
gates. The gates' specification is `docs/qc_workflow.md`, each gate in `docs/gates/`.

## The run's state

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

## The host and the modes

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

## The scope of each message

Before any tool runs, the model reads your message into a form, its scope: the stages it asks
(process, pick, invert, soils; none for a question about what exists, or a request no tool
makes), the profile or run, the positions in metres, the windows' length and step in the unit
you give them (receivers or metres), the window lengths a comparison names, whether to do again
work already there, what it says of your hand work, which option it chooses among those
offered, the processing mode it names and the workers (CPU cores) it asks the work to use. The
form's JSON schema constrains
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
  ends with them: the choice is yours. They wait until a message chooses one, a tool offers
  others, or a stage's work is done: a message in between (a typo, a question about the run)
  leaves them offered, its answer giving them again;
- a message that chooses an option asks, with it, what the message that got the options asked:
  its stages from the option's on and its positions, windows and workers (choosing "a new run"
  after "process and invert" processes, picks and inverts; choosing a run to work on picks and
  inverts it);
- the workers your message asks run every stage of its work, at most the machine's cores, said
  in the scope's line and the parameters used;
- a run your message names is the one its calls change: a call on another run is refused (one
  the turn made aside, or the one an option you chose names), so that a run that does not
  exist is said so, with the runs there, never replaced by one the model picked.

## The run and the plan, settled by code

Before the model acts, the host settles the run the message's work is on
(`src/paco/agent/runs.py`), from the scope and the server's lists of runs (the resources
`paco://profiles/{profile}/runs`, `paco://runs/{run_id}` and `paco://runs`):
- a run the message names is checked: one that does not exist is said so, with the profile's
  runs (else the newest of every profile) to choose from, and nothing runs;
- processing asked of a profile that has runs: you choose a new run or the run to work on (the
  newest three, each with its mode, windows and how far it went); asked again, or with a run
  this conversation made, a new run without a question; with no run, a new run;
- only later stages asked (pick, invert, soils): the run the conversation is on goes on;
  otherwise the profile's one run, or you choose among several;
- code writes these questions, the model not called, and the options wait for your choice.

The model then reads one plan after your message: the run (or a new run to make first) and
the stages to do on it, in their order, with those the run lacks (its curves before its
models). A call on another run is refused. After each result the host says what is next; an
answer before the plan is done, nothing in its way, is sent back once. An option you chose
among a tool's is made by the host itself, then the plan goes on.

Every answer is written by code around the model's text (`src/paco/agent/answer.py`):
- the scope's line;
- what the model says: its draft put into an answer form, constrained so that the text holds
  no question;
- "Done:" and "Left out:", from the tools' results (what each stage did, the windows a gate
  rejected or a stage failed, with why);
- "Asked but not done:", the stages the message asked that no tool did (none when the answer
  asks the user to choose: the question says why);
- the question and the options when you must choose (a tool offered options, a tool cannot go
  on without you, or your request for work left it unclear), else "Next:", what you can ask
  next and where PAC shows the results; a message asking for no work (a look, something no
  tool makes) gets an answer, not a question back;
- "Parameters used", "Settings the gates changed" and "Settings kept as you gave them", last.

A number in the model's text that neither a result of the turn, your messages, the tools'
results kept from earlier turns nor the scope holds is flagged after it ("Numbers not found in
the tools' results: ..."); a run's or a job's id is a name, not numbers.

The loop keeps itself in check (`src/paco/agent/loop.py`):
- With each message naming a profile, the host tells the model the profile's latest run, read
  from the server's resource `paco://profiles/{profile}/latest-run`: the model makes no run id
  up.
- A call made already in the answer is refused; a third time, the answer ends. A call the host
  refused (outside the scope, or waiting for your choice) made again ends it at once: it would
  be refused again.
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
invert a profile whose run has images, it picks and inverts them; asked to process a profile
that has runs, it asks whether to make a new run or to work on one of them (above), then does
what you asked on the run you chose; asked to invert a run with curves, it inverts them; asked to pick it again, or to process
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
stage at fault and a change that can be applied as it is. The records and the images have one
set of settings for the whole line, the same for every record and window as in PAC's pages: the
changes G1, G2 and G3 ask of them are tried by the line loop inside `run_processing`, and kept
for the whole line when more windows then give a curve G3 passes (at most 4,
`docs/gates/loop.md`). The picking and the inversion are each window's: `pick` and `invert`
apply their gates' changes, within budgets (2 retries per gate and window, 2 per window over the
run, and 6 inversion retries per window of its own); any other change is the model's to make,
with `redo` (the records or the images for the whole line).
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

## The evaluation

The conversation is saved when you leave (`exit`). To evaluate the model, run
`uv run paco-evaluate`, or name scenarios: `uv run paco-evaluate list_profiles pick_active`.
The model samples its answers, so one play of a scenario is a noisy measure: it plays each
scenario five times (`--repeat N` to change it) and reports each scenario's pass rate against
its threshold (3 plays of 5 by default). `uv run paco-evaluate --history` tabulates the pass
rates of every evaluation kept, by model and prompts' version, the latest version against the
one before (`--model` for one model), and over the same plays the tool calls that failed, the
calls refused as outside the message's scope, the answers a cap ended and each gate's retries a
play. An evaluation's plays share one images' cache, removed at its end, and their inversions
are quick (two chains of 3,000 iterations where the message gave no sampler, G5 taking them as
converged): the plays measure the agent's calls and answers, not the models; the scenarios that
judge the inversion's gates on your settings invert at PAC's effort, as every play does with
`--full-inversion`. A play the model's server cuts short (out of reach, failing, or a 404 page
instead of its JSON: a proxy's while the server behind it is down) is lost, not counted, and
listed under the report; a request it refuses (a JSON schema it does not take, the context
exceeded, a model name it does not serve) fails the play, with the server's message.
