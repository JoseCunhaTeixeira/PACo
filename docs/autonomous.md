# The autonomous mode

In the autonomous mode the model works as a geophysicist does with PAC: it surveys the line, runs
a full grid of the line's settings, judges the images by eye, refines the best, commits one set of
settings for the whole line, decides each window, freezes the curves, inverts them and reports.
PACo gives it the tools and the rules; code checks every rule, never the model alone.

Run it with `paco-server --mode autonomous` and `paco-agent --mode autonomous`, or from PAC's
assistant with the mode set to Autonomous. It needs a frontier model with vision and a long
context (128k tokens): every choice of settings is made by eye.

## Phases

A conversation is one study of one profile: its first message makes the study, each later
message goes on with it, and a new conversation makes a new study. Every message is read into the
objective's form: the settings it types are kept over those given before, and a message about
another profile asks for a new conversation. The study's phase is read from its records before
every call, so a stopped answer goes on, at the next message, where the study stands.

| Phase | What the model does | Ends when |
|---|---|---|
| P0, objective | Code reads the message into a form: the profile, the objective (target depth, priority, effort), the settings the user typed | The study exists |
| P1, survey | `survey`: the geometry, the records as G1 measures them, the reach, the usable band, the surface waves' cone; `look` at gathers and spectra | The model writes its conclusion in the journal |
| P2, curves | `search`, then `try_settings` to refine, `commit`, `pick`, `decide` each window | `freeze` |
| P3, freeze | The accepted curves frozen in `curves.json`; nothing is processed or picked after it | At once |
| P4, inversion | `invert` the frozen curves, `decide` each model; soils only when asked (`petro_models`, `invert_petro`) | Every model decided |
| P5, report | Code writes the report around the model's text | Done |

Each phase offers only its tools. PACo's read-only tools (`inspect`, `preset_settings`,
`inversion_settings`) and `journal` are offered in every phase.

## Tools

| Tool | What it does |
|---|---|
| `survey` | The line before any setting is chosen: geometry, records, reach, usable band, cone, modes |
| `search` | The full grid of the line's settings, scored in code; the variants judged by eye in sheets; the finalists tried side by side |
| `try_settings` | Variants of the line's settings on chosen windows, judged by G1 to G3, against the current settings; changes no run |
| `commit` | The line made with the settings, every window one receiver apart, judged by G1 to G4; the run in PAC's layout |
| `observe`, `look` | The gates' results as text; figures of records, images, picks, experiments and models |
| `advise` | PACo's own choice for a lever (window length, mute, segments, picking), not applied |
| `pick` | Windows picked again, each way with its windows: the tracker, a guide, a lasso, a deletion, or `fixes` (each window's own change G3 suggested); then G4, once |
| `decide` | Each window accepted or rejected with its reason; a rejection only once the window was tried hard |
| `freeze` | The decided curves frozen for the inversion |
| `invert`, `job_status` | The frozen curves inverted in the background, followed by the host |
| `petro_models`, `invert_petro` | Soils, N values and the water table, when asked |
| `journal` | Hypotheses, conclusions and decisions, each resting on the entries it names |

`try_settings` takes the hypothesis it tests, by its id or in words, and `commit` the conclusion it
rests on: each is written in the journal with them.

## The search and the judging

`search` runs once a study. Every setting of PAC's page for the line's method takes the values in
`src/paco/autonomous/grid.yaml` (`paco-grid` prints the grid's size), and every combination is
tried on the same windows along the line. The variants with no image, or with fewer than a quarter
of their windows passing, are dropped. Every other variant is judged by eye, with no score:

- The variants come in sheets of 12, by window length, each drawn at the same four places along
  the line as PAC draws images.
- The host judges the sheets in short calls of their own, three sheets a call: the judging
  prompt, the atlas's dispersion images, the line's facts and the sheets, a form without thinking.
  The conversation itself never carries the sheets.
- Each sheet's cleanest are kept, 2 at most; the kept meet again round after round until one sheet
  is left, whose kept (3 at most) are tried side by side at full size.
- Each sheet's judgment is saved in `searches/S1_judged.jsonl`.

`try_settings` and `commit` wait until every sheet is judged.

## The study's folder

`<output>/<profile>/studies/<id>/` holds the work:

| File | What it holds |
|---|---|
| `journal.jsonl` | The objective, the experiments, the commits, the model's hypotheses, conclusions and decisions |
| `given.json` | The settings the user typed, locked for every trial, commit and inversion |
| `survey.json` | The survey |
| `searches/` | The grid's variants and scores, the knockout, each sheet's judgment |
| `experiments/` | Each experiment's variants and scorecards |
| `line.json` | The committed line: its run and its commits with their reasons |
| `picks.jsonl`, `decisions.jsonl` | The model's own picks and decisions |
| `curves.json` | The frozen curves |
| `usage.jsonl` | Every model call, form and tool call, with its tokens and time |
| `transcripts/` | The conversations |

## The model's context

Before each call the host rebuilds what the model reads: the role and the phase's manual, the
conversation, and the state card last (the phase, the objective, the budgets, the line now, its
open issues). A tool's result shrinks to its journal line once read, and figures leave the
conversation once seen. Past 60 % of the model's context, the oldest calls drop out. At the freeze
the conversation starts again from the request, the objective and the frozen curves.

## Rules code enforces

- The settings the user typed stay as they gave them; a change of one is refused.
- A tool of another phase is refused.
- A window is rejected only once tried hard: the line's pick, a pick after a fixable flag, a guide
  from its neighbours, a lasso when the model sees the images.
- Nothing is processed or picked after the freeze.

## Budgets

The objective's effort sets the budgets. At 80 % of a budget the state card says so; a tool call
past a budget is refused, saying what to do instead; once the model calls or the tokens are spent,
the run stops and says where the work stands. The budgets are the study's, so the conversation's:
a new conversation starts a new study, its budgets fresh.

| Effort | Experiments | Commits | CPU in P2 | Inversion retries a window | Model calls | Tokens |
|---|---|---|---|---|---|---|
| quick | 8 | 2 | 6 × the first processing | 2 | 60 | 2 M |
| normal | 30 | 6 | 60 × | 6 | 200 | 5 M |
| thorough | 60 | 8 | 120 × | 6 | 400 | 10 M |

The tokens count everything the model reads and writes, its forms (the judging's) too. The
search's judging is not counted as model calls, and the search's time is not counted as CPU.

## After a session

Only the results stay. At the freeze, and when a session ends, the study's working files are
removed: the records each base preprocessed, the trials' windows, G1's settled records, the
search's thumbnails, and once the curves are frozen the survey's figures. The images' cache is
emptied when a session ends, once no other conversation answers. The line's run, the frozen
curves, the journal, the decisions, the experiments' and the search's scores and the transcripts
stay. A study opened again after its report makes again what it needs.
