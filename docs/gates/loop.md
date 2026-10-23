# The loop: stage tools, retries, going back, asking

> The lines named below are those the measurements were made on: `active_p1` and `passive_p1`,
> the demo lines, and `active_p2` and `passive_p2`, the real lines of which PAC's test
> profiles, `active_test_profile` and `passive_test_profile`, are light copies.

How the gates act (`docs/qc_workflow.md`): the records and the images are made
with one set of settings for the whole line, every record and window alike, as a person sets
them in PAC's pages; the gates judge each record and window, and the changes they ask of the
records or the images are the line's, tried by the line loop and kept when more windows get a
curve G3 passes. The picking is each window's, with G3's retries; the inversion each window's,
with G5's. Each stage tool returns one summary; going back across stages is the agent's, with
`redo`; the agent asks the user when the request leaves a choice open or when it is stuck.
Code: `src/paco/qc/line.py` (`run_processing`), `line_loop.py` (the line loop), `curves.py`
(`pick`), `inverting.py` (`invert`, a job), `redo.py` (`redo_stage`), `loops.py` (what the
loops share), `budgets.py` (the budgets, and why a retry is refused); tools:
`src/paco/server.py`; tests: `tests/test_server.py` (the tools end to end),
`tests/test_inversion.py` (the job), `tests/test_qc_line_loop.py`, `tests/test_qc_*.py`.

## What each tool runs

| Tool | Stages | Its gate's retries | Left to the agent |
|---|---|---|---|
| `run_processing` | S1 per record, G1, the S2 rules (band, window length, mute), S2 per window, G2, the line loop | G1: the traces and records it excludes left out of the windows. The line loop: the changes G1, G2 and G3 (on a trial pick) ask of the records or the images (a mute, the velocity range, the records' trigger, a passive line's fk threshold), each tried on trial windows and, kept, made on the whole line | none |
| `pick` | S3 and G3 per window, G4 over the line | G3: the window's picking again with its change (corridor, coherence rule, longest wavelength, mode rule, resampling, a band: the coherent one). G4: an outlier picked again along its neighbours' median curve, then G3 on it | a change of the line's settings (`redo`) |
| `invert` (a job) | S4 per window (bounds from its curve), G5, G6 | G5: longer sampling (the iterations 100 models a chain need, or twice), a Vs bound widened, a layer more or fewer. G6: none (a non-unique model is kept: its posterior converged, and sampling longer gives it again) | G5's `no_mode` band: the window's picking, an earlier stage |
| `redo` | the preprocessing or the phase shift for the whole line (the line's settings changed, some windows only refused), the picking or the inversion for the windows given (by xmid or by flag), then what follows up to G4; the inversion as a job | the gates' own, as above | |

Every retry is an attempt in the run's QC log, triggered by `<gate>:<flag>` (a gate's retry)
or `backtrack` (the agent's `redo`); its results move to `attempts/<n>_<stage>/`. A record or
an image made again because the line's settings changed (a change the line loop kept, a
`redo` of the line) is an attempt triggered by `line change` (by `mute trial`, for the mute
trial's), its parameters the line's change it was made for, the same on every record and
window it remade (PAC's attempts say it as what changed, and credit it to the line); the change
is logged on the line's attempt too, with the loop's notes, and each change the loop tried in
`line_loop.json`, with its trial windows' scores (PAC's run card lists them). The budgets
: 2 retries per gate and window, 2 per window over the run, counted as each is granted
(a batch of windows cannot overshoot); a window's inversion draws on its own 6; the line loop
keeps 4 changes at most (`QCConfig.line`), outside them. A unit asking for a retry it cannot
have stays flagged and goes on, as every flagged unit (only a rejected unit stops), a note first
saying why: its budget spent (`budget_spent`), the retry would run with the parameters of the
attempt before (`nothing_to_try`), or it asks only to change settings the user gave (`locked`, its
note naming the change asked: `locked, asks dispersion vmax 375 (given: 250)`). A flag whose change
touches a setting the user gave is held back whole, its parts going together (G5's longer
sampling doubles the iterations and the burn-in); the unit's other flags retry. A limit the
user gave (the picking's band, `fmin` and `fmax`) takes the change within it instead (G3's
coherent band 8 to 40 Hz in a band given from 10 to 50 Hz: 10 to 40 Hz); a change only past it
leaves the attempt's band: `locked, asks fmax 60 (given: 50)`. A change of the
line's settings the line loop did not keep leaves its units as they are, the flag kept as a
note.
`redo` is refused once the run's budget is spent. The settings a call gives that its tool does
not have (an override stage or setting, a picking or an inversion setting, an argument the tool
does not declare) are ignored and said, the call running without them (`paco.ignoring`); a
wrong value of a setting the tool has, or a name misspelling one, refuses the call.

## What the agent reads

The tools return the run, the gates' summary, and what to do next (`next`, e.g. "Call pick with
run_id ..."). The summary gives, one line each: the counts per gate and verdict, the retries
spent against the run's budget (and the inversion's, up to 6 a window), each retry with the
change it made and the verdicts it led to, the changes the rules and the gates made to the
settings, the stages that failed, then each flag once, with the windows or records raising it
as stretches of xmids, its message and the change it suggests:

```
<gate>: <n> <verdict>, ...
Retries: <n> of <the run's budget>.
Inversion retries: <n>, up to 6 a window.
Retried <gate>:<flag> or backtrack, <where> with <stage> <change> (each its own): now <n> <verdict>.
Changes at <stage>, <where>: <note> (each its own values)
Failed <stage>, <where>: <error>
<gate> <flag>, <where>: <message> -> <change>, or keep: <why>, or reject: <why>
```

For example, on `active_p1` with windows of 24 receivers every 24 (`G2.md`):

```
G2: 4 pass
G2 band_at_fmax, xmid 2.88-20.88 (4): The coherent band reaches the image's highest frequency:
  the band may go on beyond it. -> keep: fmax stays: a wider band let other ridges compete
```

The records of the demo start 20 ms before the shot (their SEG-2 headers say `DELAY -0.020`):
the muting moves each record's time origin by that delay (S1), and G1 checks the first breaks
against it (`G1.md`).

## When the agent asks

When the request leaves the user's choice open: work already there that the request asks to
make again without saying how, or work made by hand that a step would change (the tools'
options, `paco.choices`). And when stuck: no image or no curve left for the line, or a request
the data do not allow (windows longer than the
line). Then one short question with 2 to 4 concrete options, its choice first. The tools say when: `pick` answers "G4
rejected the line: no curve to invert, you are stuck. Ask the user which to try, with options:
..."; `redo` refuses a spent budget the same way. Everywhere else the agent decides from the
summaries, and says which settings the gates changed.
