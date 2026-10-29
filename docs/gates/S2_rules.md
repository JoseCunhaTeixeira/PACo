# The coherence rules for S2, before the phase shift

> The lines named below are those the measurements were made on: `active_p1` and `passive_p1`,
> the demo lines, and `active_p2` and `passive_p2`, the real lines of which PAC's test
> profiles, `active_test_profile` and `passive_test_profile`, are light copies.

Not a gate: rules checked once per line, after S1 and G1 and before any image is made, so that
the phase shift's parameters come from the data and the geometry. Code:
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
| The near limit | none: a window stacks its nearest shots too, G3 reporting its nearest shot (`near_offset`), unless the user gives `masw.distance_min` | A near field gives `active_p2` no cleaner image and breaks its passive-active images; on the demo it costs curves (below) |
| The window length | one length for the whole line. Given none: a ladder of the autonomous search's lengths shorter than the line (5, 7, 9, 11, 13, 15, 17, 21, 25, 33, 41 and 47 receivers), each tried on the same 27 sample windows spread evenly along the line, its ends included (S2, the picking and G3; a passive line's each with segments as long as its window asks, below), the mute trial and the line loop on as many. The length whose sample windows G3 passes most is kept; on a tie, the one whose curves reach deepest (the longest median wavelength), then the shorter. A length given (by the user or the agent) is kept, its trial windows tried for the record | The length whose images give the most curves along the line, as its sample windows show it, and of those the deepest. No share of the line caps it: a 24-geophone spread may need most of its length, and `active_p1`'s windows pass best at 47 or 48 receivers, half its line (below) |
| The frequency step | 1/T by construction: no padding before the phase shift, so the image's step follows the record length | A finer step only interpolates |
| The velocity range | left to G2, which flags a ridge on the grid's edges per window | The trial images could set it; not needed on the demo (1 to 1000 m/s) |

## The ladder proposes, the agent decides

`run_processing` returns the lengths tried as `lengths`, for the agent to choose from, for the
line's length and the depth or detail the request needs. After a line stating the receivers,
their spacing and the longest length it may try, each length tried gets one line:
its span in metres, how many of its trial windows passed G3, the passed curves' median
shortest and longest wavelengths (a model reaches about half the longest), the picks' median
precision, the windows the line gets at that length, and which length is proposed.

The trials' shares follow the whole line's. On `active_p1`, picked past the windows' reach,
the 27 trials pass 17 at 5 receivers, 20 at 7 and 9 and 23 at 11; processed and picked with the
gates' retries, the line gives curves on 65 % of its windows at 5 receivers, 68 % at 7, 77 % at 9
and 85 % at 11, the failures mostly the windows at the line's ends (1 or 2 points next to the
shot).

Its `next` says the length is the ladder's proposal, to change with `masw.length` when the
request needs more depth or lateral detail, saying why; the ideal scripted agent then takes 24
receivers for "as deep as this line allows" and 7 for "as much lateral detail as the data
allow". A length given is kept, whatever the ladder would choose.

## The most sample windows passing

Every length tried on PAC's lines, the trial windows G3 passed of 27 (the picks stopping at
three window lengths, a passive line's segments sized to each length):

| Receivers | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 21 | 25 | 33 | 41 | 47 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `active_p1` | 0 | 0 | 0 | 3 | 22 | 26 | 26 | 27 | 26 | 25 | 27 | 27 |
| `active_p2` | 24 | 25 | 25 | 23 | 22 | 24 | 23 | 20 | 22 | 23 | 21 | 16 |
| `active_p2` passive-active | 18 | 17 | 15 | 22 | 21 | 21 | 21 | 20 | 16 | 17 | 14 | 6 |
| `passive_p2` | 10 | 9 | 21 | 14 | 27 | 25 | 16 | 24 | 20 | 13 | 17 | 17 |

The ladder keeps 47 receivers on `active_p1` (27 of 27, as 21 and 41, its curves the deepest, to
30 m), 9 on `active_p2` (25 of 27, as 7, its curves to 29 m against 22 m), 11 in passive-active
(22 of 27) and 13 on `passive_p2` (27 of 27). On `active_p1` the picks' median uncertainty grows
with the length (19 % at 13 receivers, 27 % at 47) as its curves reach deeper (8 m, 30 m): a
precision rule would stop at 13.

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

The spec asks for the nearest source offset to be at least λmax/2 against near-field effects
(Park et al.: nearer, the wave is not yet a plane surface wave and the long wavelengths read
slow). The workflow sets none, as the images ask:

- On `active_p2`, active, with shots up to 63 m, near fields of 0 to 1.5 window lengths from a
  window's nearest receiver give the same images at 11 and 15 receivers; in passive-active at 11
  receivers, 15 to 22.5 m breaks the band above about 40 Hz into stripes.
- On the demo line, `masw.distance_min` at 1.5 window lengths from the source to the window's
  middle (λmax at 2 window lengths) drops the near shot from the windows at both ends, and G3
  falls from 17 to 11 passes of 19 at 24 receivers (8 mode jumps), from 12 to 5 of 21 at 16.

G3 reports each window's nearest shot (`near_offset`), never flagging it: on a line whose
windows stack the shots next to them, every window's nearest is within half its longest
wavelength.

## The trials on longer lines

On `active_p2` (142.5 m), 25 of 27 trials pass at 5 receivers (93 %) and the line gives curves
on 88 % of its windows; on `passive_p2`, 19 of 27 at 11 receivers (70 %), against 65 % of the
line's windows.

## A passive line's segments: their length and FK selection

The segments' length is set by the window's span: the ladder tries each length with segments
ten times as long as the slowest wave takes to cross its window (G1's 80 m/s), end to end,
within 0.1 s and the shortest record, and the line keeps the segments its length was chosen
with. Longer windows need longer segments: with PAC's 1 s for every length, `passive_p2`'s
trial windows pass G3 on 21 of 27 at 13 receivers, against 27 of 27 with its own 2.25 s.

When the user gives the segments' length, `segments.py` tries their FK selection on three
windows at the line's length, the ladder's: the line's own, and the selection with no band
and within 80-1500 m/s at thresholds 0.05 to 0.3, never off. Each segment is sliced, whitened,
normalized, tapered and correlated once, as the pipeline does, flipped and not, and its f-k
lopsidedness measured (sigpipe's `fk_ratio`, the selection's own measure); a candidate keeps the
segments beyond its threshold, flipped where their energy runs the other way, and stacks them.
It is judged by the dispersion image its stack makes: the span of wavelengths its M0 pick holds,
then the pick's coherence. It must keep 1 % of the segments at least (G2's `min_fk_kept_share`)
and its correlograms must have converged: the virtual shots of its kept segments' two halves
agree over their arrivals, the median correlation of their traces there 0.7 at least. The best
replaces the line's own selection when it widens their span by a fifth, or when theirs had not
converged. The trials are kept in `segments.json`, the choice in the line's notes.

The segments' length is not tried that way: on `passive_p2` at 13 receivers, the same trial
over the segments' length too takes 4.5 s segments and the FK selection at 0.1 within 80 to
1,500 m/s (their M0 over 4.7 times its shortest wavelength, against 3.5), and G3 then passes 51
of the line's 84 curves, against 80 with the length's own segments: three windows did not stand
for the line.
