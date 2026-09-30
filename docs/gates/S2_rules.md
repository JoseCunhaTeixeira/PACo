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
| The band | fmax capped at the smaller of the median usable fmax of the records G1 kept and Nyquist; fmin raised to their median usable fmin. Never widened: PAC's 100 Hz stays when the records go higher | Widening the band makes G3 worse at every window length on the demo (below): above 100 Hz a second ridge competes and the picker jumps. The median, not the worst record: on `active_p2` the worst records' limits (24 to 86 Hz) would cut the whole line's long wavelengths, where the median record is usable from 2.3 to 298 Hz |
| The far limit | `masw.distance_max` at the line's reach, where the traces' median SNR falls under 2 dB (sigpipe's `line_reach`, G1's `reach_snr_db`), unless the user gave one | A window stacks no shot whose traces there are mostly noise: PAC's 100 m takes `active_p2`'s traces beyond about 60 m, which are noise |
| The near limit | on an active line, a window stacks no shot nearer to its nearest receiver than half the longest wavelength the kept length's trial curves reached (the line's note `near_field`), where it has a farther one; a window with near shots only keeps them, and G3 flags their near field (`near_field`). A `masw.distance_min` the user gives rules instead | Nearer, the wave is not yet a plane surface wave and the long wavelengths read slow (Park et al.). A window keeps its near shots when it has no other: leaving out every window's near shots costs the demo a third of its curves (below) |
| The window length | one length for the whole line. Given none: a ladder of short lengths (5, 7, 9 and 11 receivers, at most half the line: longer windows lose the lateral detail, and on poor images pass G3 more often only because they resolve velocity better), each tried on 27 windows spread evenly along the line, its ends included (S2, the picking and G3). Up the ladder while 80 % of them pass G3 and each length's picks are at least 10 % more precise than the best so far: the first length whose passed curves' median velocity uncertainty is within 20 % is proposed; the climb stopping first, the most precise that passed (the shorter on a tie); none passing, the one that passed most (the shortest on a tie). The next length, when there is one, is tried for comparison. A length given (by the user or the agent) is kept, its trial windows tried for the record | Lateral resolution, and picks precise enough for the inversion: the shortest windows pass G3 with loose picks (`active_p2`: 40 % at 5 receivers, 26 % at 11) |
| The frequency step | 1/T by construction: no padding before the phase shift, so the image's step follows the record length | A finer step only interpolates |
| The velocity range | left to G2, which flags a ridge on the grid's edges per window | The trial images could set it; not needed on the demo (1 to 1,000 m/s) |

## The ladder proposes, the agent decides

`run_processing` returns the lengths tried as `lengths`, for the agent to choose from, for the
line's length and the depth or detail the request needs. After a line stating the receivers,
their spacing and the longest length allowed (half the line), each length tried gets one line:
its span in metres, how many of its trial windows passed G3, the passed curves' median
shortest and longest wavelengths (a model reaches about half the longest), the picks' median
precision, the windows the line gets at that length, and which length is proposed.

The trials' shares follow the whole line's. On `active_p1`, the 27 trials pass 17 at 5
receivers, 20 at 7 and 9 and 23 at 11; processed and picked with the gates' retries, the line
gives curves on 65 % of its windows at 5 receivers, 68 % at 7, 77 % at 9 and 85 % at 11, the
failures mostly the windows at the line's ends (1 or 2 points next to the shot).

Its `next` says the length is the ladder's proposal, to change with `masw.length` when the
request needs more depth or lateral detail, saying why; the ideal scripted agent then takes 24
receivers for "as deep as this line allows" and 7 for "as much lateral detail as the data
allow". A length given is never climbed past: a climb would turn a given 7 receivers (6 of 9
trials passing) into 16.

## The precision rule

On `active_p2` (96 receivers, 1.5 m apart), the ladder's trial windows give, by length: 5
receivers, 25 of 27 passed with picks at 40 % (the median velocity uncertainty of the curves
passed); 7: 24, 30 %; 9: 24, 30 %; 11: 23, 26 %; 16: 13, 18 %; 24: 13, 9 %; 32: 11, 5 %; 48: 15,
5 %. The shortest that passes (5) gives curves the inversion can hardly use; the longest ones
are precise but fail G3 on half the line, and are not tried. The ladder climbs while the
lengths pass and buy precision, stops at the first within 20 %, and takes the most precise that
passed when none is (`CoherenceRules.max_uncertainty`, `min_precision_gain`): 7 receivers on
`active_p2` (30 %, where 9 gives no better), 7 on the demo (39 %).

## On the demo profile, the picks cut at 2 window lengths

`active_p1` (96 receivers 0.25 m apart, 2 s at 2 kHz, one shot off each end), windows every 4
receivers, G3 on the whole line for each length and upper frequency, the picking's
`max_wavelength` at 2 window lengths:

| Window length | fmax 100 Hz | 150 Hz | 324 Hz (G1's usable) | kept curves reach |
|---|---|---|---|---|
| 5 receivers (1 m) | 0 of 23 | | 0 of 23 | |
| 8 (1.75 m) | 0 of 23 | | 0 of 23 | |
| 12 (2.75 m) | 0 of 22 | | 1 of 22 | 5 m |
| 16 (3.75 m) | 12 of 21 | 8 of 21 | 4 of 21 | 7 m |
| 24 (5.75 m) | 17 of 19 | 15 of 19 | 6 of 19 | 11 m |
| 32 (7.75 m) | 15 of 17 | 13 of 17 | 6 of 17 | 15 m |
| 48 (11.75 m) | 13 of 13 | | 11 of 13 | 21 m |

Short windows fail for want of a curve (no ridge within 2 window lengths, too few points, mode
jumps), and every step of the band above 100 Hz loses curves to mode jumps: a retry widening
fmax (x 1.5) makes the loop worse, so G2's band flags are kept, not retried (`G2.md`).

### The near field

The spec asks for the nearest source offset to be at least λmax/2 against near-field effects.
Applied to every window alike, as `masw.distance_min` at 1.5 window lengths from the source to
the window's middle (λmax at 2 window lengths), it drops the near shot from the windows at both
ends of the demo line, and G3 falls from 17 to 11 passes of 19 at 24 receivers (8 mode jumps),
from 12 to 5 of 21 at 16. So a window with near shots only keeps them, and G3's `near_offset`
metric and its kept `near_field` flag name the windows whose nearest shot is closer than half
their longest wavelength kept. With the picker following its ridge, the demo's 24-receiver
trial curves reach 34 m: a near field of 17 m, where 2 of the 4 windows keep near shots only.

## The trials on longer lines

On `active_p2` (142.5 m), 25 of 27 trials pass at 5 receivers (93 %) and the line gives curves
on 88 % of its windows; on `passive_p2`, 19 of 27 at 11 receivers (70 %), against 65 % of the
line's windows.

## Design notes

- **27 trials spread over the whole line, its ends included**: the trials' shares then follow
  the line's (above). Without the end windows, they propose 5 receivers where the line gives
  curves on 65 % of its windows.
- **80 % of the trials, not two thirds**: with the picks cut at 2 window lengths, a ladder at
  two thirds keeps 16 receivers on `active_p1` (with 5 trials or 9), where only 50 of the line's
  81 windows pass G3 (62 %); at 80 % it keeps 24, where 66 of 73 pass (90 %).
- **The band capped, never widened** (above), and the near field left to each window (above).

## A passive line's segments: their length and FK selection

PACo tunes the parameters the passive workflow has: the segments' length, set by the window's
span, and the FK selection, which matters to the dispersion image; the correlograms must
converge, and the step stays the segments' length. Once the window length is chosen,
and for the stages the user did not set, `segments.py` tries on three windows spread along the
line (`SegmentRules`) each segment length (the line's own, and 10, 20, 40 and 80 times the
slowest wave's crossing of the windows' span at G1's 80 m/s, within 0.1 s and the shortest
record, each end to end) with each FK selection (the line's own, none, and the bands none and
80-1,500 m/s at thresholds 0.05 to 0.3). Each segment is sliced, whitened, normalized, tapered
and correlated once, as the pipeline does, flipped and not, and its f-k lopsidedness measured
(sigpipe's `fk_ratio`, the selection's own measure); a candidate keeps the segments beyond its
threshold, flipped where their energy runs the other way, and stacks them.

A candidate is judged by the dispersion image its stack makes: the span of wavelengths its M0
pick holds (the picker's, as S3 runs it; how much of the curve and of the depth the image gives),
then the pick's coherence. It must keep 1 % of the segments at least (G2's `min_fk_kept_share`,
one floor for both: good segments are rare), and its correlograms must have converged: the
virtual shots of its kept segments' two halves (every other one) agree over their arrivals, the
median correlation of their traces there 0.7 at least. The best replaces the line's own
settings when it widens their span by a fifth, or when theirs had not converged. G2's share of
coherent columns tells nothing here: a passive line's stacked images come near 100 % whatever
the settings. The trials are kept in `segments.json`, the choice in the line's notes.

On the demo, with 24-receiver windows, the trials start from the defaults (1 s segments end to
end, the FK selection at 0.2, the correlations stacked phase-weighted). passive_p1 takes 0.72 s
segments (ten crossings of its windows' 5.75 m) and the FK selection at 0.1: its M0 picked over
7.8 times its shortest wavelength, 8 % of the segments kept, converged (0.85), against 5.0 with
the defaults, whose correlograms do not converge (0.48). passive_p2 takes 8.62 s segments (20
crossings of 34.5 m), the FK selection kept at 0.2: its M0 over 17.4 times its shortest
wavelength, 22 % of the segments kept, converged (0.86), against 1.0 with the defaults (the
picker finds almost nothing). With 9 m windows, 0.7 s segments, the FK selection at 0.5 and a
linear stack, passive_p2 keeps 0.2 % of its segments and its correlograms do not converge
(0.54); the trials take 4.5 s segments and the FK selection at 0.2 within 80 to 1,500 m/s: its
M0 over 7.8 times its shortest wavelength against 3.1, 32 % of the segments kept, converged
(0.98).
