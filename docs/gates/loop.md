# The loop: stage tools, retries, going back, asking

How the gates act (option B of `docs/qc_workflow.md`): the records and the images are made
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
| `pick` | S3 and G3 per window, G4 over the line | G3: the window's picking again with its change (corridor, coherence rule, longest wavelength, mode rule, resampling, a band: the coherent one, or below an alias G2 found). G4: an outlier picked again along its neighbours' median curve, then G3 on it | a change of the line's settings (`redo`) |
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
(rule 4): 2 retries per gate and window, 2 per window over the run, counted as each is granted
(a batch of windows cannot overshoot); a window's inversion draws on its own 6; the line loop
keeps 4 changes at most (`QCConfig.line`), outside them. A
unit asking for a retry it cannot have is rejected with its last flags: its budget spent
(`budget_spent`), the retry would run with the parameters of the attempt before
(`nothing_to_try`), or it asks only to change settings the user gave (`locked`, its reason
naming the change asked: `locked, asks dispersion vmax 375 (given: 250)`). A flag whose change
touches a setting the user gave is held back whole, its parts going together (G5's longer
sampling doubles the iterations and the burn-in); the unit's other flags retry. A change of the
line's settings the line loop did not keep leaves its units as they are, the flag kept as a
note.
`redo` is refused once the run's budget is spent.

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
options, `paco.choices`). And when stuck: no image or no curve left for the line, windows
rejected once the budget is spent, or a request the data do not allow (windows longer than the
line). Then one short question with 2 to 4 concrete options, its choice first. The tools say when: `pick` answers "G4
rejected the line: no curve to invert, you are stuck. Ask the user which to try, with options:
..."; `redo` refuses a spent budget the same way. Everywhere else the agent decides from the
summaries, and says which settings the gates changed.

## Design notes

The loop retries only where a retry can change the result, and leaves no window out unsaid.

- **A retry with the parameters of the attempt before is refused** (`next_try`, `unchanged`):
  it would give its result again. The unit is rejected, `nothing_to_try`.
- **No retry that could not change what it judges**: G1 asks for no filter or mute (it
  measures in the band the images use, the noise before the muting: `G1.md`), G2 for no second
  mode, G3 for no filter or air-wave mute (`G2.md`, `G3.md`), and G6 does not sample a
  converged model longer (`G6.md`); a mute is asked only of records not muted yet.
- **One set of settings for the line, the line loop changing it** (`line_loop.py`): the
  records and the images made as a person makes them in PAC's pages, the same settings for every
  record and window, so that the images compare along the line and a window's fix never makes
  the images of the windows sharing its records over. The gates still judge each unit; the
  changes they ask of the records or the images go together by what they change (the records'
  trigger delays as their median), the one the most failing windows ask first: tried on 15 trial
  windows, those asking it and the others spread along the line, and kept with 2 more passing
  G3 without losing a fifth of the curves' longest wavelengths (the mute trial's rule); the
  whole line is then made again with it, judged again, and the asks read again. A mute keeps
  the line's median pulse as its width, and the mute trial's taper. Measured on the four real
  lines, fixes made per record and window against the line loop (curves the inversion takes;
  processing and picking):
  `active_p1` 69 and 69, 51 s and 46 s; `active_p2` 82 and 90 (the standard mute, which 3
  noisy windows asked: 15 of 15 trial windows against 11), 544 s and 192 s; `passive_p1` 54
  and 55 (the fk threshold halved: 8 of 15 against 3), 76 s and 42 s; `passive_p2` 86 and 82,
  200 s and 80 s (the fk threshold halved helped the 24 windows asking it, and costs the
  others: 9 of 15 against 12, not kept).
- **An alias, or competing ridges, limit the window's picking**: G2's alias and G3's
  `on_data` ask the picking's band of the window (`picking {"fmax": ...}`, `fmin`), the image
  staying the line's.
- **A mode jump's first fix cuts the band** where the largest step between points consecutive
  in frequency sits, the side with fewer points going (`picking {"fmax": ...}` or `fmin`); once
  the band is cut, the corridor is halved. Halving the corridor twice does not fix the jump of
  the demo's xmid 14.88 when its picks are cut at 2 window lengths.
- **A length given, by the user or the agent, is kept**: the ladder does not climb past it
  (`S2_rules.md`).
- **The traces a window leaves out are its records'**: each shot's image is made without the
  traces G1 excluded from that shot, and a trace excluded in at least half of the window's
  records is left out of all of them. Leaving out every trace any record excluded would empty
  every window of `active_p2` (66 shots a window). A record left with fewer than 3 of the
  window's receivers leaves the window. Passive windows keep the union (their records are
  stacked as one). Passive-active windows keep one set of receivers, since their correlation
  gathers are stacked sample by sample: the receivers half the shots excluded leave every shot,
  and a shot that excluded another of the window's receivers leaves the window.
- **What the host guarantees** (`paco.agent.host`): the parameters used and the settings the
  gates changed are listed after every answer, as the tools gave them, and an inversion the
  model starts is followed to its end. It watches no words, neither the user's nor the
  model's: whether to ask, and what the request covers, the model reads.
- **A step back says what came of it**: the summary's "Retried backtrack, ... now ..." gives
  the verdict of the gate that judges the stage redone (G3 for the picking).
- **A grid too narrow can fool G2.** With vmax at 150 m/s on a line whose ground reaches
  290 m/s, only 13 % of the columns peak on the grid's edge (the ridge lies wholly outside),
  and the correlation's artifact at 1 m/s reads as an alias, which would cut the band to 12 Hz.
  So G2 looks above its 30 m/s floor for the edge, the competing ridges and the aliases, and
  judges no second ridge while the grid is too narrow. With vmax at 250 m/s, G2 asks 375 m/s
  of the windows that need it: the line loop widens the line's range when it gains windows; a
  range the user gave stays, and those windows are rejected, `locked`.
- **The ladder's trial windows get G2's velocity-range fix before G3 judges them**: a grid too
  narrow for the ground is no fault of the window's length.
- **A band wholly below the usable one** (a record usable only from 122 Hz, a preset fmax of
  100 Hz) would leave the capped band empty: fmax then goes up to the usable band's top, the
  one case it is widened.
- **Synthetic defects that do not work:** an air wave the picker follows needs M0 faster than
  340 m/s; neither scaling the geometry (G1's time windows then miss the waves) nor
  compressing the records' time axis (G1 then finds them usable only from 122 Hz) gives one on
  the demo's records. An isolated G4 outlier cannot come from a data defect with overlapping
  windows: a defect on a few receivers changes every window holding them, a run G4 rightly
  calls geology. Both stay covered by the gates' tests on analytic curves and synthetic lines.
