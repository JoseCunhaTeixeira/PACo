# The loop: stage tools, retries, going back, asking

How the gates act (milestone 14, option B of `docs/qc_workflow.md`): each stage tool runs its
own gate's retries and returns one summary; going back across stages is the agent's, with
`redo`; the agent asks the user only when it is stuck. Code: `src/paco/qc/line.py`
(`run_processing`), `curves.py` (`pick`), `inverting.py` (`invert`, a job), `redo.py`,
`loops.py` (what the loops share); tools: `src/paco/server.py`; tests: `tests/test_server.py`
(the tools end to end), `tests/test_inversion.py` (the job), `tests/test_qc_*.py`.

## What each tool runs

| Tool | Stages | Its gate's retries | Left to the agent |
|---|---|---|---|
| `run_processing` | S1 per record, G1, the S2 rules (band, window length), S2 per window, G2 | G1: a record preprocessed again with the change it asks (a trigger delay corrected); traces and records it excludes left out of the windows. G2: the phase shift again, in groups sharing the same change (the velocity range first) | G1's filter or mute, G2's mute: an earlier stage |
| `pick` | S3 and G3 per window, G4 over the line | G3: the picking again with its change (corridor, coherence rule, longest wavelength, mode rule). G4: an outlier picked again along its neighbours' median curve, then G3 on it | G3's filter, mute or band: an earlier stage |
| `invert` (a job) | S4 per window (bounds from its curve), G5, G6 | G5: longer sampling (the iterations 100 models a chain need, or twice), a Vs bound widened, a layer more or fewer. G6: a non-unique model inverted again, sampling twice as long | - |
| `redo` | the stage given, for the windows given (by xmid or by flag), then what follows up to G4; the inversion as a job | the gates' own, as above | |

Every retry is an attempt in the run's QC log, triggered by `<gate>:<flag>` (a gate's retry) or
`backtrack` (the agent's `redo`); its results move to `attempts/<n>_<stage>/`. The budgets
(rule 4): 2 retries per gate and window, 2 per window over the run, counted as each is
granted (a batch of windows cannot overshoot); a window asking for a retry past its budget is
rejected, "budget spent", with its last flags. `redo` is refused once the run's budget is spent.

## What the agent reads

The tools return the run, the gates' summary, and what to do next. On `active_p1`, windows of
24 receivers every 24 (2026-09-24):

```
run_processing -> next: Call pick with run_id 20260924-175047-f0a0.
G1: 2 pass
G2: 4 pass
Retries: 2 of 8.
Retried G1:shifted_trigger, 1.dat, 2.dat with preprocessing {"trigger":{"t0":0.0188}} (each its own): now 2 pass.
Changes at preprocessing, 2.dat: traces [1, 89, 90] left out of the windows
Changes at phase_shift, line: masw length 24 for the whole line: trial windows G3 passed, by length in receivers: 8/9 at 24.
G2 band_at_fmax, xmid 2.88-20.88 (4): The coherent band reaches the image's highest frequency: ... -> keep: fmax stays: a wider band let other ridges compete

pick -> next: 3 curves passed G3 and G4: call invert with run_id 20260924-175047-f0a0.
G3: 3 pass, 1 reject
G4: 4 pass
Retries: 4 of 8.
Retried G3:mode_jump, xmid 14.88 (1) with picking {"corridor":0.1} (each its own): now 1 reject.
G3 budget_spent, xmid 14.88 (1): G3: the retry budget is spent; the last attempt raised mode_jump, ... -> reject: budget spent
G4 gaps, line: No curve at xmid 14.88 (1): the section has gaps there. -> keep: the gaps stay in the report

job_status (the job ended, 500 iterations given):
G5: 3 pass
G6: 4 pass
Retried G5:not_converged, xmid 2.88-8.88 (2), 20.88 (1) with inversion {"n_iterations":17000,"n_burnin_iterations":1700}: now 3 pass.
G5 smoothing_misfit, xmid 2.88 (1): The smooth median misfits 2.4 at short wavelengths (3-5 m) where the layered median fits (1.0): ...
```

The records of the demo start 20 ms before the shot (their SEG-2 headers say `DELAY -0.020`):
G1 measures it and S1 corrects it, per record. Halving the corridor twice did not stop the pick
of xmid 14.88 from jumping between modes: the window ends rejected, and the line has a gap.

## When the agent asks

Only when stuck (the user's decision of milestone 14): no image or no curve left for the line,
windows rejected once the budget is spent, or a request the data do not allow (windows longer
than the line). Then one short question with 2 or 3 concrete options, its choice first. The
tools say when: `pick` answers "G4 rejected the line: no curve to invert, you are stuck. Ask the
user which to try, with options: ..."; `redo` refuses a spent budget the same way. Everywhere
else the agent decides from the summaries, and says which settings the gates changed.

## Found while building it

- **A grid too narrow fooled G2.** With vmax at 150 m/s on a line whose ground reaches 290 m/s,
  only 13 % of the columns peaked on the grid's edge (the ridge lay wholly outside), and G2
  read the correlation's artifact at 1 m/s as an alias, cutting the band to 12 Hz. Now G2 looks
  above its 30 m/s floor for the edge, the competing ridges and the aliases, and judges no
  second ridge while the grid is too narrow. With vmax at 250 m/s, G2 widens the range to
  375 m/s on the windows that need it, and every curve passes; at 150 m/s the widening (x1.5,
  twice) spends the run's budget of 8 retries on 4 windows before the picking.
- **The ladder blamed the length for the user's grid.** Its trial windows now get G2's
  velocity-range fix before G3 judges them. Still, with vmax at 250 m/s the trial windows of 24
  receivers pass 2 of 9, those of 32 pass 9 of 9: the ladder keeps 32, a change of the user's
  length the agent must report.
- **A band wholly below the usable one** (a record usable only from 122 Hz, a preset fmax of
  100 Hz) left the capped band empty: fmax now goes up to the usable band's top, the one case
  it is widened.
- **Retries in batches overshot the run's budget** (11 of 8): each retry now counts as it is
  granted.
- **Synthetic defects that did not work:** an air wave the picker follows needs M0 faster than
  340 m/s; neither scaling the geometry (G1's time windows no longer matched the waves) nor
  compressing the records' time axis (G1 then found them usable only from 122 Hz) gave one on
  the demo's records. An isolated G4 outlier cannot come from a data defect with overlapping
  windows: a defect on a few receivers changes every window holding them, a run G4 rightly
  calls geology. Both stay covered by the gates' tests on analytic curves and synthetic lines.

## Since 2026-09-25

- **G1's retries are the records'**: they count against each record's own budget (2 per gate),
  not the windows' run budget, which a narrow velocity range spent on G2 after G1's two
  corrections. The summary's "Retries: n of N" counts the windows' only.
- **A mode jump's first fix cuts the band** where the largest step between points consecutive
  in frequency sits, the side with fewer points going (`picking {"fmax": ...}` or `fmin`); once
  the band is cut, the corridor is halved. Halving the corridor twice never fixed the demo's
  jump at xmid 14.88 (gone since the pick stops where its ridge breaks).
- **A length given, by the user or the agent, is kept**; the ladder no longer climbs past it
  (`S2_rules.md`).
- **The traces a window leaves out are its records'** (on a real line, `active_p2`): each
  shot's image is made without the traces G1 excluded from that shot, and a trace flagged in at
  least half of the window's records is left out of all of them. The window used to leave out
  every trace any of its records excluded: with 66 shots a window, every window of `active_p2`
  ended empty. A record left with fewer than 3 of the window's receivers leaves the window.
  Passive windows keep the union (their records are stacked as one). Passive-active windows
  (2026-09-26) keep one set of receivers, since their correlation gathers are stacked sample
  by sample: the receivers half the shots excluded leave every shot, and a shot that excluded
  another of the window's receivers leaves the window.
- **What the host guarantees** (`paco.agent.host`, the user's decisions after the evaluation
  of the same day): the settings the gates changed are listed after every answer, as the tools
  gave them; an answer that asks or offers once a stage tool has run is asked again once,
  unless a tool said the agent is stuck (a question before any work is the request's
  clarification: asked again, Qwen3-8B ran 96 receivers unasked where 120 were asked, 2 of 3
  plays, 2026-09-26); `invert`, and `redo` of the inversion, are refused unless the user's
  message asks for models (invert, inversion, model, Vs, shear). Qwen3-8B missed these in 12,
  7 and 6 of 39 plays.
- **A step back says what came of it**: the summary's "Retried backtrack, ... now ..." gives
  the verdict of the gate that judges the stage redone (G3 for the picking). It said "no
  verdict" for every window, looking for a gate named "backtrack".

Nothing left to judge here.
