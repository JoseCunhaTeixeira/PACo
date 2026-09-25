# The coherence rules for S2, before the phase shift

Not a gate: rules checked once per line, after S1 and G1 and before any image is made, so that
the phase shift's parameters come from the data and the geometry (rule 7). Code:
`src/paco/qc/coherence.py` (the rules), `src/paco/qc/line.py` (`process_line`: S1, G1, the
rules, S2, then G2 to G4); rules: `coherence` in `qc_config.json` (`CoherenceRules`); tests:
`tests/test_qc_line.py`. What they changed is logged as notes of the line's phase shift, and
the agent reads them first in the summary; the ladder's trials are kept in the run's
`coherence.json`, their windows in `coherence/<length>/`.

## What they set

| Rule | How (default) | Why |
|---|---|---|
| The band | fmax capped at the smaller of every record's usable fmax (G1) and Nyquist; fmin raised to the highest usable fmin. Never widened: PAC's 100 Hz stays when the records go higher | Widening the band made G3 worse at every window length on the demo (below): above 100 Hz a second ridge competes and the picker jumps. The user's decision of milestone 13 |
| The window length | one length for the whole line. Given none: a ladder of lengths (5, 7, 9, 11, then 16, 24, 32, 48, 64, 96, 128 receivers, at most half the line), each tried on 9 windows spread along the line (S2, the picking and G3); the first at which 80 % of them pass G3 is proposed, else the one that passed most (the shortest on a tie), and one length more is tried for comparison. A length given (by the user or the agent) is kept, its 9 trial windows tried for the record | Lateral resolution first: the shortest windows that give a curve, among those most of the line can use. The user's decisions of milestone 13 (the ladder, 9 trials at 80 %) and of 2026-09-25 (the short lengths first; the ladder proposes, the agent decides) |
| The frequency step | 1/T by construction: the image's step follows the record length, since the padding went (milestone 9) | A finer step only interpolates |
| The velocity range | left to G2, which flags a ridge on the grid's edges per window | The trial images could set it; not needed on the demo (1 to 1,000 m/s) |
| The offsets | reported, not applied: G3's `near_field` flag (below) | The user's decision of milestone 13 |

## The ladder proposes, the agent decides (2026-09-25)

`run_processing` returns the lengths tried as `lengths`, for the agent to choose from, for the
line's length and the depth or detail the request needs. With no settings on `active_p1`:

```
line: 96 receivers 0.25 m apart (23.75 m); windows of up to 48 receivers (half the line)
5 receivers (1.00 m): 5/9 trial windows passed G3, wavelengths 5.0-10.0 m, 92 windows on the line
7 receivers (1.50 m): 6/9 trial windows passed G3, wavelengths 5.0-11.5 m, 90 windows on the line
9 receivers (2.00 m): 6/9 trial windows passed G3, wavelengths 5.5-14.5 m, 88 windows on the line
11 receivers (2.50 m): 7/9 trial windows passed G3, wavelengths 5.0-22.0 m, 86 windows on the line
16 receivers (3.75 m): 8/9 trial windows passed G3, wavelengths 5.0-22.0 m, 81 windows on the line (proposed)
24 receivers (5.75 m): 9/9 trial windows passed G3, wavelengths 5.0-26.0 m, 73 windows on the line
```

The wavelengths are the passing curves' median shortest and longest: a model reaches about half
the longest. The failures at 5 to 11 receivers are the windows at the line's ends (1 or 2 points
next to the shot). Its `next` says the length is the ladder's proposal, to change with
`masw.length` when the request needs more depth or lateral detail, saying why; the ideal
scripted agent then takes 24 receivers for "as deep as this line allows" and 7 for "as much
lateral detail as the data allow". A given length is no longer climbed past (the rule of
milestone 13: 7 receivers, 6/9, became 16 again).

## On the demo profile, with the 2-window-lengths cut (2026-09-24)

`active_p1` (96 receivers 0.25 m apart, 2 s at 2 kHz, one shot off each end), windows every 4
receivers, G3 on the whole line for each length and upper frequency (2026-09-24):

| Window length | fmax 100 Hz | 150 Hz | 324 Hz (G1's usable) | kept curves reach |
|---|---|---|---|---|
| 5 receivers (1 m) | 0 of 23 | | 0 of 23 | |
| 8 (1.75 m) | 0 of 23 | | 0 of 23 | |
| 12 (2.75 m) | 0 of 22 | | 1 of 22 | 5 m |
| 16 (3.75 m) | 12 of 21 | 8 of 21 | 4 of 21 | 7 m |
| 24 (5.75 m) | 17 of 19 | 15 of 19 | 6 of 19 | 11 m |
| 32 (7.75 m) | 15 of 17 | 13 of 17 | 6 of 17 | 15 m |
| 48 (11.75 m) | 13 of 13 | | 11 of 13 | 21 m |

Short windows fail for want of a curve (no ridge within 2 window lengths, too few points,
mode jumps), and every step of the band above 100 Hz loses curves to mode jumps. G2's own
retry before this milestone (fmax x 1.5) would have made the loop worse; its band flags are now
kept, not retried (`G2.md`).

`process_line("active_p1")` with no settings at all (44 s): G1 caps nothing (the records are
usable to 324 Hz); the ladder tries 5, 8, 12, 16 and 24 receivers on 9 windows each (0, 0, 0,
7 and 9 pass) and keeps 24. On the whole line, 66 of its 73 windows pass G3 and 7 retry (mode
jumps, for the loop); the 38 windows at the line's ends carry `near_field`, kept. The summary
the agent reads starts with:

```
Changes at phase_shift, line: masw length 24 for the whole line: trial windows G3 passed, by
  length in receivers: 0/9 at 5, 0/9 at 8, 0/9 at 12, 7/9 at 16, 9/9 at 24.
```

With the ladder as first built (5 trials, two thirds), it kept 16 receivers (4 of 5 trials
passed), and only 50 of the line's 81 windows passed G3 (62 %): five trials overestimated the
line. Nine trials at two thirds still kept 16 (7 of 9); nine at 80 % keep 24.

### The near-offset rule, measured, reported and not applied

The spec asks for the nearest source offset to be at least λmax/2 against near-field effects.
With λmax at 2 window lengths, that is `masw.distance_min` = 1.5 window lengths from the source
to the window's middle. On the demo it drops the near shot from the windows at both ends of the
line, and G3 falls from 17 to 11 passes of 19 at 24 receivers (8 mode jumps), from 12 to 5 of
21 at 16. So it is reported, not applied (the user's decision): G3's `near_offset` metric and
its kept `near_field` flag name the windows whose nearest shot is closer than half their
longest wavelength kept (the ten at the ends of the dense line).

## To judge

- Nine trials at 80 % propose 16 receivers on the demo since 2026-09-25 (3.75 m windows, 79
  of the line's 81 curves passing G3 and G4). Two of the nine trial windows sit at the line's
  ends, next to the shots, where short windows keep 1 or 2 points: they alone keep 5 to 11
  receivers under 80 %. Should the trials leave the end windows out? And on a longer or more
  varied line, is 9 trials enough to speak for it?
- The near field is only reported, on half the demo's windows (both ends of a 24 m line with a
  shot off each end): is a flag on half the line informative, or noise the agent will learn to
  skip?
- The band is never widened. On a line whose records carry a clean fundamental mode above
  PAC's 100 Hz, that loses the short wavelengths: should the ladder also try a wider band?
