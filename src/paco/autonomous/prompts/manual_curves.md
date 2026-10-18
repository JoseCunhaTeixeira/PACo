The curves (P2): one set of settings for the whole line, the windows one receiver apart, each
stacking its shots by the same rule, every record and image alike; then each window's pick, a
decision on every window, and the freeze. Notation: dx the receiver spacing, N a window's
receivers, its length L = (N-1)·dx, λ = v/f.

## Windows

- A window averages the ground under it: no change narrower than L is resolved, however fine
  the step, which is one receiver, fixed. N runs from 5 to half the line's receivers.
- Under 2·dx a ridge may be its alias (G3 flags aliasing_zone). The longest wavelength grows with
  L: PAC and PACo draw the limit at 3·L (G3 flags beyond_reach); trust up to about 2·L on a clean
  image, less on a noisy one. The data may stop a curve earlier: the source's low frequencies,
  or the near field.
- The depth informed is about λmax/2; the thinnest layer resolved, about λmin/3.
- Precision grows with L: the picks' uncertainty falls from about 40 % at 5 receivers to under
  10 % at 24 and more. Short windows pass G3 more often because loose picks pass.
- To choose, start from the objective: a target depth z needs λmax near 2·z, so L near z; a
  lateral priority starts at 5 to 9 receivers; a balanced one in the middle of the range, on a
  geometric scale. Then try 3 or 4 lengths around it on windows spread along the line, its ends
  included, reading the share passing, the shortest and longest wavelengths, the uncertainty
  and the windows the line would have. advise window_length gives PACo's own choice.
- After the picks, the card's checks: most curves stopping well under 2·L, at the same low
  frequency for every length, mean the data limit the depth: shorten. Many curves beyond 3·L
  where the target needs that depth: lengthen, or cut the tail (picking max_wavelength), or
  accept a shallower model and say so.

## Shots

- near_field_m: a window stacks no shot nearer than that to its nearest receiver, where it has a
  farther one; one with near shots only keeps them (G3 flags near_field). Close to the source the
  wave is not yet a plane surface wave: the long wavelengths read slow and the curve bends down
  at low frequencies. Keep the nearest shot at least λmax/2 from the window.
- masw distance_max: the farthest shot from a window's middle; past the survey's reach the
  traces are mostly noise. masw distance_min is a hard bound that loses windows: leave it to a
  user who asks for it.
- Shots on both sides average out dipping layers: keep both when the distances allow. More shots
  raise the SNR, not a near-field bias.
- To choose: before any pick, no near-field distance (or about L) and distance_max at the reach.
  After the first picks, compare each window's nearest shot with half its λmax (G3's
  near_offset), and try a near-field distance at half the line's median λmax on the flagged
  windows and a control set. At the line's ends the rule may leave a window only a far, weak
  shot: a shorter distance gives the near shot back, at the price of a near-field bias at its
  longest wavelengths. Try it, and say which you keep.

## Preprocessing

- The same for every record. trigger t0 corrects a delay common to every trace, when the headers
  are wrong; one varying from trace to trace rejects the record (G1).
- muting (vmin, vmax, width, taper) keeps the surface waves' cone, removing faster arrivals and
  slower noise. Start from the survey's cone; PACo's candidates are 80 to 1,500 m/s, the cone
  widened by 2.25 or 1.5 each side, and the tight cone (its median divided and multiplied by
  1.5). Muting too much cuts the far offsets and the low frequencies: λmax must keep 0.8 of its
  value without a mute.
- The air wave, 330 to 345 m/s, cannot be muted when the surface waves travel near that speed:
  G3 rejects a curve that is the air wave.
- A band-pass applied alike to every trace does not change a phase-shift image: limit the
  frequencies with the image's band.
- Expected velocities come from the data (the cone, the first breaks) and the user's notes. G1's
  exclusions of traces and records apply by themselves.

## Images

- The phase shift's grid (dispersion fmin, fmax, vmin, vmax, nv) and the stacking of each
  window's shot images (linear, or root with its n), the same for every window.
- fmin at least the records' median usable low edge; fmax at most their median usable high edge
  and Nyquist. Widening above the preset's 100 Hz lost curves to mode jumps on the demo line: a
  hypothesis to test, not a default.
- Above v = 2·dx·f the ridges alias: G2 limits the picking below the alias.
- The velocity grid brackets the ridge with a margin: a ridge on its edge (G2 ridge_at_vmax,
  ridge_at_vmin) asks for a wider one. The frequency step is 1/T: a finer one only interpolates.
- Linear stacking by default; the root keeps what every shot sees, a lever where one shot is
  noisy or in its near field.
- Diagnosis, the cheapest lever first. Few coherent columns, low prominence: noise or
  refractions; a mute, a shorter distance_max, a longer window, the root. Competing ridges, a
  mode jump at high frequencies: a lower fmax or picking band, a longer window. band_at_fmin with
  the same low frequency at every length: the source; accept the depth. A coherent band much
  narrower than the usable band: the near field at its low end (the near-field distance up), or
  attenuation at its high end (distance_max down).

## Passive and passive-active lines

- A passive line's windows correlate the noise of every record: each record cut into segments
  (slicing segment_duration, segment_step), each segment kept by its FK selection (selection
  threshold: a lower one keeps more; vmin, vmax: the apparent velocities kept), whitened and
  normalized (onebit), correlated against the window's first receiver, the correlations stacked
  (stacking: phase_weighted with its nu, linear, root): a virtual shot at the window's end. The
  shots' settings (distances, near field, trigger, mute) mean nothing there; the windows' length,
  the image's grid and the picking work as on an active line.
- The segments set the image: its frequency step is 1/segment_duration, and its longest period.
  Longer segments give lower frequencies and a finer step, but fewer to stack, a noisier image.
  advise segments gives PACo's trial (a few minutes): lengths of 10 to 80 crossings of the window
  at the slowest velocity, FK thresholds of 0.05 to 0.3.
- G2 reads a passive window's correlations (their SNR, usable band and lateral coherence:
  noisy_correlations) and the share of segments its FK selection kept (few_segments_kept): both
  ask for more of the data, a lower threshold. look correlations and look selection show them.
- G3's near_field note on a passive window means nothing: its virtual source is its own first
  receiver. decide and freeze leave it out.
- passive-active, a mode of an active line: each window correlates its shots beyond its ends, as
  noise, and stacks the correlations (stacking), the trigger and the mute as the active mode's. A
  lever where the active images are poor, at the line's ends or near the shots. A change of mode
  makes a new run.

## Picking

- M0 is usually the slowest coherent ridge at low and middle frequencies, its velocity rising
  with wavelength, continuous: a step of more than 30 % between points (G3 mode_jump) is another
  mode or noise. Higher modes dominate at high frequencies, and where a stiff layer lies over a
  softer one: G3 keeps inverse_dispersion, which can be geology.
- Points under 2·dx and beyond 3·L stay, flagged. The inversion's depth follows the longest
  wavelength, so a tail beyond the window's reach or in the near field drives the model's deep
  part: say so, or cut it.
- G4 along the line: a curve off two agreeing neighbours that fits neither side is an outlier:
  pick it again along their median curve. A change shared with one side is geology: keep it.
  Never edit a curve to make it fit.
- Order of work: commit a line, read G3 and G4 grouped by flag and by stretches of windows; fix
  single windows for their fixable flags (G3's suggested change, a band cut at a jump, a narrower
  corridor, a lower coherence rule, a guide from the neighbours, a lasso when you see the
  images); when many windows fail for one reason, suspect the line's settings and go back with a
  hypothesis. A window's own pick is its recipe: a later commit picks it the same way.

## Deciding and freezing

- Accept a window whose curve passed G3 with no blocking flag, and G4; its notes need your reason.
  Reject one only once tried hard: the line's pick, a pick after each fixable flag (or a reason
  naming why none applies), a guide from its neighbours when one side has two passing, a lasso
  when you see the images. decide refuses otherwise, saying what is owed.
- When every window is decided, freeze. Ask the user first when more than 30 % of the windows
  would be rejected.
