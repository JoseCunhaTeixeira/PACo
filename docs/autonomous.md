# The autonomous mode

In the autonomous mode the agent processes a line as a geophysicist does in PAC. It looks at the
records. It compares every combination of the line's settings on their dispersion images and
chooses the cleanest. It processes the whole line with it, reviews the picked curves and decides
each window. It inverts the accepted curves and reports.

The agent chooses; PACo does the processing and enforces the rules. Every rule is checked in
code, never left to the agent alone.

Run it with `paco-server --mode autonomous` and `paco-agent --mode autonomous`, or from PAC's AI
agent page with the mode set to Autonomous. It needs a model with vision and a long context
(128k tokens): every choice of settings is made by looking at dispersion images.

## Words used here

| Word | Meaning |
|---|---|
| Line | The profile's receivers and records |
| Window | A group of neighbouring receivers; each gives one dispersion image, one curve, one model |
| Settings | Everything PAC's processing page sets: window length, shots used, mute, filter, dispersion image, stacking, picking |
| Variant | One combination of settings |
| Study | One conversation's work on one profile, kept in its own folder |
| Journal | The study's record: the request, what was tried, what was chosen and why |

## The checks

PACo checks every step. A check passes, flags or rejects; the agent then accepts or rejects. A
flagged unit goes on to the next step; only a rejected one stops.

| Check | What it judges |
|---|---|
| Signal check (G1) | Each record: dead or clipped traces, SNR, usable band, trigger; along the line, bad receivers |
| Dispersion image check (G2) | Each window's image: coherent band, peaks at the velocity range's edges, second ridges |
| Curve check (G3) | Each window's picked curve: points on the ridge, aliasing, near field, jumps, wavelength range |
| Check against the neighbouring curves (G4) | Each curve against its neighbours; along the line, gaps and spread |
| Model check (G5) | Each Vs model: fit to its curve, chains, depth informed |
| Check against the neighbouring models (G6) | Each model against its neighbours; along the line, gaps and spread |
| Soil column check (G7) | Each soil column's fit to its curve |
| Check against the neighbouring soil columns (G8) | Each soil column against its neighbours |

## What happens, step by step

### 1. The request

The first message of a conversation makes a study. PACo reads the message into a form: the
profile, the objective (the depth sought, what matters most, the effort) and the settings the user
typed.

The settings the user typed are locked: every later step uses them, and a change of one is refused.
A later message goes on with the same study; a setting it plainly asks for replaces the one given
before. A message about another profile asks for a new conversation.

### 2. The survey

`survey` measures the line before any setting is chosen: its geometry, each record as the signal
check measures it, how far the waves reach from the shots, the usable band, the surface waves'
cone. The agent looks at gathers and spectra (`look`) and writes its conclusion in the journal.

### 3. The search

`search` compares every combination of the line's settings. The values tried are in
`src/paco/autonomous/grid.yaml` (`paco-grid` prints how many combinations). It runs once a study.

1. PACo makes every variant on the same 12 windows along the line.
2. It drops the variants with no dispersion image, or with fewer than a quarter of their windows
   passing the checks.
3. The others are compared by eye, in sheets of 12 variants. A sheet shows a variant a row: its
   dispersion images at up to 12 places along the line, the same places in every row. The images
   are shown alone, without picks, so a pick cannot sway the choice.
4. The first sheets group the variants by window length.
5. Short calls of their own judge the sheets, three sheets a call. Each keeps the 3 cleanest
   variants of each sheet. These calls do not think aloud, and the agent's conversation never
   carries the sheets.
6. The variants kept are mixed into new sheets, round after round, until 3 are left.
7. The agent looks at those 3 itself and chooses the cleanest (`choose`), giving its reason.

Each sheet's judgment is saved. No score ranks the variants: the scores only drop the clearly bad
ones in step 2.

After the search, nothing more is tried: every combination was seen. `try_settings` is offered
only when the search chose none.

### 4. The commit

`commit` processes the whole line with the chosen settings, in PAC's layout:

- the signal check judges every record;
- every window, one receiver apart, gets its dispersion image, judged by the dispersion image
  check;
- PACo picks the curve of every window whose image was not rejected, judged by the curve check;
- the check against the neighbouring curves judges the line.

The reason for each setting is kept with the run; PAC's run card shows it.

### 5. The review of the curves

The agent looks at the picks and the checks' results (`observe`, `look`). Where a pick is poor, it
picks again (`pick`): with the tracker, guided by the neighbours' curves, with a lasso on the image,
by deleting points, or by applying the change the curve check suggested. The checks judge each new
pick.

Then the agent decides each window (`decide`): accepted, or rejected with its reason. PACo refuses:

- accepting a curve the curve check or the check against the neighbouring curves rejected: a
  flagged curve can be accepted, its flags given a reason;
- rejecting a window before it was tried hard: the line's pick, a pick after a flag that can be
  fixed, a pick guided by its neighbours, and a lasso.

### 6. The freeze

`freeze` keeps the accepted curves for the inversion. Nothing is processed or picked after it.

### 7. The inversion

`invert` inverts the frozen curves in the background, in groups of windows that share their
parameters. Each window's bounds come from its own curve; the settings the user typed stay. The
model check judges each model, then the check against the neighbouring models judges the line.
Nothing is retried automatically: the agent reads the checks' results and inverts a group again,
within each window's budget of retries. It then decides each model: PACo refuses accepting a
model the model check rejected; a flagged model can be accepted, its flags given a reason.

### 8. The soil columns

Only when the request asks for them: `petro_models` lists the soil models, and `invert_petro` makes
each window's soil column, N values and water table.

### 9. The report

PACo writes the report around the agent's own text.

## Tools

Each step offers only its tools. `inspect`, `preset_settings`, `inversion_settings` and
`journal` are offered at every step.

| Tool | What it does |
|---|---|
| `survey` | Measures the line before any setting is chosen |
| `search` | Compares every combination of settings by eye until 3 are left |
| `choose` | The agent's choice of the cleanest of the search's last 3 |
| `try_settings` | Only when the search chose none: variants of the settings on a few windows |
| `commit` | Processes the whole line with the chosen settings |
| `observe`, `look` | The checks' results as text; figures of records, images, picks and models |
| `advise` | PACo's own choice for one setting, not applied |
| `pick` | Picks windows again, each its own way |
| `decide` | Accepts or rejects each window, with its reason |
| `freeze` | Keeps the accepted curves for the inversion |
| `invert`, `job_status` | Inverts the frozen curves; follows the job |
| `petro_models`, `invert_petro` | Soil columns, when asked |
| `journal` | The agent's hypotheses, conclusions and decisions, each naming what it rests on |

## Rules PACo enforces

- Only a rejected unit stops: a flagged record, image, curve or model goes on.
- The settings the user typed stay as given.
- A tool of another step is refused.
- A window is rejected only once tried hard.
- Nothing is processed or picked after the freeze.
- A budget spent stops the tools that use it.

## Budgets

The effort in the request sets the budgets. At 80 % of a budget the agent is told. A tool call past
a budget is refused, saying what to do instead. Once the agent's calls or its tokens are spent,
the study stops and says where the work stands. A new conversation starts a new study, with fresh
budgets.

| Effort | Experiments | Commits | Processing time | Inversion retries a window | Agent calls | Tokens |
|---|---|---|---|---|---|---|
| quick | 8 | 2 | 6 × the first processing | 2 | 60 | 2 M |
| normal | 30 | 6 | 60 × | 6 | 200 | 5 M |
| thorough | 60 | 8 | 120 × | 6 | 400 | 10 M |

The tokens count everything the agent reads and writes, the sheets' judging included. The judging
calls do not count as agent calls, and the search's time does not count as processing time.

## What the agent reads

Before each call PACo rebuilds what the agent reads: its role and the step's instructions, the
conversation, and a state card last (the step, the objective, the budgets, the line now, its open
issues). A tool's result shrinks to its journal line once read, and figures leave the conversation
once seen. Past 60 % of the context, the oldest calls drop out. At the freeze the conversation
starts again from the request, the objective and the frozen curves.

## The study's folder

`<output>/<profile>/studies/<id>/` holds the study:

| File | What it holds |
|---|---|
| `journal.jsonl` | The request, the experiments, the commits, the agent's hypotheses, conclusions and decisions |
| `given.json` | The settings the user typed |
| `survey.json` | The survey |
| `searches/` | The search: its variants, each sheet's judgment, the choice, the last 3's dispersion images |
| `experiments/` | Each experiment's variants and their dispersion images |
| `line.json` | The committed line: its run, its settings and their reasons |
| `picks.jsonl`, `decisions.jsonl` | The agent's picks and decisions |
| `curves.json` | The frozen curves |
| `usage.jsonl` | Every call, with its tokens and time |
| `transcripts/` | The conversations |

## After a session

Only the results stay. At the freeze, and when a session ends, PACo removes the study's working
files: the trials' records and windows, the search's dispersion images but the last 3's, and once
the curves are frozen, the survey's figures. The images' cache is emptied when a session ends, once
no other conversation answers.

The line's run, the frozen curves, the journal, the decisions, the experiments' and the search's
results with their dispersion images, and the transcripts stay. A study opened again after its
report makes again what it needs.
