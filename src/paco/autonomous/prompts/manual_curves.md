The curves (P2): one set of settings for the whole line, the windows one receiver apart, each
stacking its shots by the same rule, every record and image alike; then each window's pick, a
decision on every window, and the freeze. Notation: dx the receiver spacing, N a window's
receivers, its length L = (N-1)·dx and its aperture N·dx, λ = v/f the wavelength of a point at
phase velocity v and frequency f.

## Where to start: the search, then your eye

- After the survey, call search, once: the full grid of the line's settings, made in code
  (minutes). Every setting of PAC's page for the line's method takes the values the physics sets
  (the windows' length over the line's range; the farthest shot as shares of the survey's reach;
  the near field as shares of the window; PACo's mutes from the survey's cone; on a passive line
  the segments as crossings of the window by the slowest wave, the FK thresholds and bands, the
  whitening, the normalization; the stacking), and every combination of them is tried on the same
  windows, judged by G1 to G3 as try_settings judges it; the values are the grid file's (PACo's
  grid.yaml). Where the user's words call for it (their group velocities, the values they want),
  give search's grid the changes, for this study alone. It does not start from PACo's workflow,
  which is another way to process the line.
- Only the really bad are dropped: no image, or fewer than a quarter of the windows passing (no
  curve along the line). Every other variant is judged by eye in sheets of 12, a variant a row at
  its windows nearest twelve places along the line (as many as it has), drawn as PAC draws images,
  with no score (none reads an image's quality) and no pick (a pick drawn would sway the eye), in
  short calls of their own: each sheet's 3 cleanest kept and mixed into new sheets, round after
  round, until 3 are left. Their sheet, the final, comes to you: choose the cleanest image by eye
  (choose), as Judging the images below says, the objective in mind, with its reason. The picks come
  after: PACo picks the line once you commit, and you look at them then and pick again those that
  need it. Should the judging stop, search again judges the sheets left; commit waits until you
  chose.
- Then commit your choice, its changes as choose gives them, each setting with its reason in plain
  words. Nothing is tried after it: every combination was seen, by eye, along the whole line. Check
  after the commit that it makes sense: the picks on M0 and not on a higher mode or an artefact, few
  points beyond the windows' reach, none aliased, the same ridge from window to window, the physics
  explaining why it wins. try_settings serves only when the search chose none (no variant gave an
  image to judge). The image's band stays the preset's or the records' usable band: a narrower one
  hides the ridge, it does not mend it.

## Physics

- Depth and layers: a curve informs the ground down to about λmax/2, and the thinnest layer it
  resolves is about λmin/3. Three layers need λmax/λmin of at least 1.34 (G3's narrow_span).
- Dispersion: where the ground stiffens with depth, longer wavelengths sample deeper and faster
  ground, so the velocity rises with the wavelength. A stiff layer over a softer one gives an
  inverse trend, which can be geology.
- A window resolves the slowness to about 1/(f·L), so Δv/v ≈ λ/L: long wavelengths need long
  windows. Under f = vmax/L a peak at the grid's top velocity is noise common to every trace, not
  a ridge, and short windows cannot separate modes: at 5 receivers a second ridge stands in most
  columns.
- The picks' uncertainty is the array's resolving power at each point, from the geometry alone:
  about 0.32·λ/A, with A = N·dx·10^(1/√(N·dx)), capped at 40 % of the velocity. It grows with the
  wavelength over the aperture: a curve reaching deeper at the same length has a larger median
  uncertainty, and a median at the 40 % cap means most points lie far beyond what the window
  resolves. It does not measure a pick's quality.
- The window's reach: PAC and PACo draw the longest trusted wavelength at 3·L (G3 flags points
  beyond, beyond_reach); trust about 2·L on a noisy image. The depth figures count the points
  beyond the reach too: read the share beyond the reach before trusting a depth.
- Aliasing: a wavelength under 2·dx aliases (v < 2·dx·f). G3 flags points there
  (aliasing_zone), and G2 cuts a window's picking at its first aliased column.
- The near field: within about λmax/2 of the shot the wave is not yet a plane surface wave; its
  long wavelengths read slow and the curve bends down at low frequencies.
- The phase shift normalizes every trace's spectrum and stacks the traces along trial
  velocities: every trace weighs the same, the image's noise floor is 1/√N, and a filter or a
  delay common to every trace cannot change the image. Its frequency step is 1/T.
- A window images the ground under it as one: across a lateral change, or with noisy traces in
  it, its traces disagree (G2's lateral coherence falls) and the image stripes, fringes and breaks
  its ridge. On active_p2, windows of 20 receivers across the 36 to 40 m stretch gave 0.71 to 0.83
  of coherent columns and a lateral coherence of 0.78, where 11 receivers gave 1.00 and 0.96.

## Windows

- A window averages the ground under it: no change narrower than L is resolved, however fine the
  step, which is one receiver, fixed. N runs from 5 to half the line's receivers.
- Length trades depth and precision against lateral detail and coverage: longer windows resolve
  longer wavelengths with smaller uncertainties, cover fewer positions, and pass G3 less often at
  first, since short windows pass with loose picks. On active_p2 (dx 1.5 m), trial windows
  passing of 27 and their median uncertainty: 5 receivers 25 at 40 %, 7 receivers 24 at 30 %, 11
  receivers 23 at 26 %, 16 receivers 13 at 18 %, 24 receivers 13 at 9 %, 48 receivers 15 at 5 %.
- PACo's ladder (advise window_length) tries the search's lengths shorter than the line on 27
  windows spread along the line, its picks stopping at 3·L, a passive line's with segments of 10
  crossings of each window at 80 m/s: it keeps the length G3 passes most of the 27, the deepest
  curves on a tie (9 receivers on active_p2, 11 in passive-active, 13 on passive_p2, 47 on
  active_p1). Its mute trial and its line loop use as many trial windows.
- Choose the length by the images: the one whose images are cleanest (by eye, one ridge unbroken
  over the widest band; lateral coherence near 1), its curves serving the objective. A target
  depth z needs λmax near 2·z, so L of at least 2·z/3 (λmax within 3·L), but only as far as the
  images stay as clean. The search tries the lengths over the whole range with the other
  settings: length and mute interact (on active_p2 a mute moved PACo's ladder from 7 to 11
  receivers), and so do length and the shots' distances or the stacking.
- After the picks, the card's checks: most curves stopping well under 2·L, at the same low
  frequency at every length, mean the data limit the depth: shorten. Many curves beyond 3·L where
  the target needs that depth: lengthen, or cut the tail (picking max_wavelength, in window
  lengths: 3 cuts at 3·L), or accept a shallower model and say so.

## Shots

- near_field_m (active and passive-active: a shot correlated brings its near field too): a window
  stacks no shot nearer than that to its nearest receiver, where it has a farther one; one with
  near shots only keeps them. G3 reports each window's nearest shot (near_offset), never judges
  it. Keep the nearest shot at least λmax/2 from the window.
- masw distance_max: the farthest shot from a window's middle, at the survey's reach by default;
  the window's far half extends L/2 beyond it. Far shots lose the high frequencies first, near
  ones carry them but bring the near field to the low ones: judge the ridge's high end and its
  low frequencies by eye. masw distance_min
  drops every window's nearest shots, where near_field_m drops them only while a farther one
  remains: it can leave a window at the line's ends with no shot.
- Shots on both sides average out dipping layers: keep both when the distances allow. More shots
  raise the SNR, not a near-field bias.
- PACo's rule sets the near-field distance at half the median λmax of the trial curves, once the
  length is chosen; dropping the near shots of every window cost a third of the curves on PAC's
  active test line. A better one may come from each window's own λmax, read within its reach:
  try it on the windows whose nearest shot is within their λmax/2 and a control set. At the
  line's ends the rule may leave a window only a far, weak shot: a shorter distance gives the
  near shot back, at the price of a near-field bias at its longest wavelengths. Try it, and say
  which you keep.

## Preprocessing

- muting (vmin, vmax, width, taper) keeps the surface waves' cone, from x/vmax to x/vmin plus the
  pulse's width, and removes what blurs their ridge: the faster arrivals (direct and refracted P,
  head waves) and the slower noise. It is the first lever of a line with shots, and the
  surface-wave window of a passive-active line: the grid tries PACo's mutes, and no mute.
  A mute cut hard at the slowest arrival rings in the image: PACo gives one without its width the
  wave train's tail (0.05 s) and without its taper 10 ms, as its own mute trial does.
- The candidates: PACo's standard 80 to 1,500 m/s, the cone widened by 2.25 or 1.5 each side, and
  the tight cone (its median divided and multiplied by 1.5); advise mute runs PACo's mute trial at
  the line's length. Beyond them: long wavelengths arrive early, at the cone's fast edge, so a mute
  that cuts the fast side alone (refractions) and keeps the slow edge wide may lose less depth.
- Compare the candidates on the images, not only a mute against none: their velocities matter.
  The mute you drew by eye on the survey's gathers is one of them. On active_p2 in
  passive-active, PACo's mute trial kept the cone widened by 2.25 (68 to 2,086 m/s), whose images
  are quieter over 16 to 45 m (background 0.10 to 0.14) than the standard mute's (0.15 to 0.21).
- Keeping a mute: PACo keeps the mute most windows pass, the tightest on a tie (the narrowest
  velocity window); on active_p2 a tie of four mutes at 26 of 27 keeps the tight cone. Keep a mute
  that passes as many windows with longer curves within the windows' reach; drop one that loses
  windows, or whose median λmax falls under 0.8 of no mute's: muting too much cuts the far offsets
  and the low frequencies. In passive-active on active_p2, the standard mute passed 15 trial
  windows against 11 without.
- trigger t0 corrects a delay common to every trace, when the headers are wrong: it changes no
  image, only where a mute's window falls, so it matters on muted records. A t0 varying from trace
  to trace rejects the record (G1).
- The air wave, 330 to 345 m/s, cannot be muted when the surface waves travel near that speed: G3
  rejects a curve that is the air wave.
- A band-pass applied alike to every trace does not change a phase-shift image: limit the
  frequencies with the image's band. G1's exclusions of traces and records apply by themselves.

## Images

- The phase shift's grid (dispersion fmin, fmax, vmin, vmax, nv) and, on an active line, the
  stacking of each window's shot images (image_stacking: linear, or root with its n), the same for
  every window.
- fmin low: the preset's 0 Hz, or a few Hz. The picker's continuity and its contrast floor (1 %
  above its column's median) stop a curve where its ridge fades, so a low fmin costs nothing,
  while a high one cuts the long wavelengths, the depth: on active_p2 the ridge reaches 7 Hz where
  the records' usable band starts near 17 Hz. Raise fmin only to cut a low-frequency artefact the
  picking follows.
- fmax at most the records' median usable high edge and Nyquist. Above the preset's 100 Hz a
  second ridge competes (on active_p1, 17 of 19 windows passing at 100 Hz, 15 at 150 Hz, 6 at 324
  Hz). A wider band pays only where λ at fmax stays above 2·dx, for thinner shallow layers.
- The velocity grid brackets the ridge with a margin, from 30 m/s up: a ridge on its edge (G2
  ridge_at_vmax, ridge_at_vmin) asks for a wider one. G2 counts a peak at vmax only above f =
  vmax/L, and a peak under 30 m/s is an artefact (a correlation's zero lag, a mute's edge).
- On an active line, image_stacking is linear by default (89 of 90 windows passing on active_p2,
  79 with the root); the root keeps what every shot sees, a lever where one shot is noisy or in
  its near field. A passive or passive-active line stacks correlations instead, phase-weighted
  (below).

## Judging the images

- A good image holds one sharp ridge, continuous over a wide band and rising toward the low
  frequencies, smooth, the rest of it pale, the pick on its crest: the atlas's first image.
- The search's final (its 3 finalists, a row each, the images alone), and each experiment's variants
  (side by side, in sheets of its current settings and three, each with its pick; look experiment
  E<n>:current,3,5 shows others), are drawn as PAC draws images: each frequency's row from its least
  value (yellow) to its largest (black). Judge each panel against the atlas and name its fault and
  cause before the lever: vertical stripes (a hard mute's edges, noisy or dead traces, a lateral
  change under the window); horizontal fringes under the ridge (a short window's side lobes, a hard
  mute); a ridge broken into pieces, or jumping at a vertical edge (a lateral change under a long
  window, noise); a broad dark area at the low frequencies (a short window's resolution, Δv/v ≈ λ/L,
  not noise: a longer window sharpens it, and is better only where its ridge stays as unbroken at
  the higher frequencies); a weak ridge (scaled row by row, any peak looks dark: the yellow around
  it tells); blocks (a coarse frequency step: segments too short for the window); dark ridges under
  the dashed line (aliases) or left of the dotted one (beyond the window's reach).
- No score reads an image's quality as the eye does: G2's background and
  its image score favour long windows (a short window's floor, 1/√N, and wider ridge raise it), a
  short window's smear can score well where its columns peak, an arrival too fast for surface
  waves holds a high SNR and lateral coherence, a pick on a ridge that is not M0 is on its ridge.
  Your eyes choose; the gates say which curves pass, and the scorecard the coverage and the
  depth.
- The image comes before the pick: a picking change cannot mend a bad image, only settings that
  change the image can (the length, the mute, the shots, the stacking).

## Passive and passive-active lines

- A passive line's windows correlate the noise of every record: each record cut into segments end to
  end (slicing segment_duration, segment_step its equal); each segment kept whole or dropped by its
  FK selection, which measures how much of its energy travels one way along the line, between the
  apparent velocities vmin and vmax, and keeps it over the threshold (a lower one keeps more);
  whitened and normalized (onebit: the spectral phase and the sign alone, so that noise bursts do
  not dominate); correlated against the window's first receiver; the correlations stacked: a virtual
  shot at the window's end. The shots' settings (distances, near field, trigger, mute) mean nothing
  there; the windows' length, the image's grid and the picking work as on an active line.
- The correlations' stacking, passive and passive-active alike: phase-weighted, the preset's nu
  2. It weights the linear stack by how well the correlations agree in phase, so that coherent
  arrivals stay and incoherent noise falls; a higher nu (4, 6) keeps less of what they disagree
  on: fewer stripes, while the ridge stays.
- The whitening and the temporal normalization, onebit by default, weigh every segment alike, its
  noise too; without the normalization (none) the strongest noise weighs most: cleaner where it
  travels along the line, worse where bursts dominate.
- The segments set the image: its frequency step is 1/segment_duration, its longest period the
  segment's length. Longer segments give lower frequencies and a finer step, but fewer to stack, a
  noisier image: they are not more data. Short segments work best, the user finds: 1 s or less, by
  the window's length and the waves' velocities, long enough for the slowest wave to cross the
  window a few times; the grid tries 2 to 16 crossings at the slowest group velocity. advise
  segments gives PACo's: 10 crossings of the window at 80 m/s, as its ladder sizes them; for
  segments the line's settings give, its FK selection trial (a few minutes): thresholds of 0.05 to
  0.3 with no band or 80 to 1,500 m/s, a candidate counting with 1 % of its segments kept and the
  virtual shots of its two halves agreeing (0.7), the longest λmax/λmin winning. On passive_p2 at
  13 receivers its 2.25 s segments passed G3 on 80 of 84 windows; 4.5 s with FK 0.1 in 80 to
  1,500 m/s, the best of a trial on three windows, passed 51.
- On a passive line the settings follow PAC's page: the windows' length, by the wavelengths they
  must resolve (as on an active line); the preprocessing, a filter at most (no mute); then the
  interferometry in its order: the slicing (the segments' length, long enough for the slowest waves
  to cross a window a few times and no longer, which also sets the image's frequency step), the
  segment selection (the FK threshold, 0 to 1, the preset's 0.2: lower keeps more segments and their
  noise, higher fewer and cleaner, while enough remain to stack; its velocity band from 80 to
  1,500 m/s to start, since noise records show no velocity to set it from, then wider, narrower or
  open), the spectral whitening, the temporal normalization, the correlations' stacking; then the
  dispersion image. The FK selection, its threshold and its band, gets the most values: keep the
  variants whose images and virtual shot gathers are cleanest (look selection shows each segment's
  ratio against the threshold).
- G2 reads a passive window's correlations (their SNR, usable band and lateral coherence:
  noisy_correlations) and the share of segments its FK selection kept (few_segments_kept): both
  ask for more of the data, a lower threshold. Its share of coherent columns says little on a
  passive stack. look correlations and look selection show them.
- Look at the virtual shot gathers at each step, passive and passive-active alike (each
  experiment shows them under its images, look correlations a line's window): a clean one holds
  one sharp arrival moving out from the virtual source at the surface waves' speed, little after
  it; energy all along the traces is noise, and its image stripes; an arrival far faster than the
  surface waves is body waves or noise common to the receivers, and its image smears at high
  velocities, however high its SNR. The atlas's correlations show the three.
- passive-active, a mode of an active line: each window correlates the shots beyond its ends,
  within distance_max of its middle, as noise, and stacks the correlations phase-weighted. The
  trigger and the mute work as the active mode's, the mute being its surface-wave window, since
  each shot is correlated whole: an image peaking at vmax unmuted asks for a mute first. A window
  whose correlations stay under G2's 6 dB is rejected: every shot in reach is already stacked. A
  lever where the active images are poor, at the line's ends or near the shots. A change of mode
  makes a new run.

## Picking

- M0 is usually the slowest coherent ridge at low and middle frequencies, its velocity rising
  with wavelength, continuous: a step of more than 30 % between points (G3 mode_jump) is another
  mode or noise, and good curves step at most about 26 %. Higher modes dominate at high
  frequencies, and where a stiff layer lies over a softer one: G3 keeps inverse_dispersion, which
  can be geology. A stretch where v rises in proportion to f is the edge of the window's
  resolution, not dispersion (G3 constant_wavelength).
- PACo's picker follows the ridge from where it is sharpest, within a corridor, across gaps of up
  to 2 Hz, and stops where the ridge stands less than 1 % above its column. Its parameters
  (picking: the threshold, the corridor, min_relative_coherence, wavelength_step, the band) are the
  cheapest lever; advise picking gives PACo's.
- G3's flags and their fixes: sharpness under 0.8 or prominence under 0.5 of a plane wave's,
  reject (prominence asks a mute in the line's trials); on_data under 0.6, pick in the coherent
  band only; too_few_points (under 4), resample finer, then a lower min_relative_coherence;
  narrow_span (λmax/λmin under 1.34), a lower min_relative_coherence; mode_jump, cut the band at
  the largest step, then halve the corridor; no_ridge, mode_min_ratio 1.2; the air wave, reject.
  inverse_dispersion, aliasing_zone and beyond_reach are notes.
- Points under 2·dx and beyond 3·L stay, flagged. The inversion's depth follows the longest
  wavelength, so a tail beyond the window's reach or in the near field drives the model's deep
  part: say so, or cut it (picking max_wavelength, in window lengths; min_wavelength, the
  shortest, in receiver spacings: 2 stops at the aliasing limit).
- G4 along the line: a curve off two agreeing neighbours (each side within 7.5 % of itself) by
  more than 15 %, fitting neither side, is an outlier: pick it again along their median curve. A
  change shared with one side is geology: keep it. Never edit a curve to make it fit.
- Look at the picks after each commit, which shows twelve windows along the line (look sheet
  for others, look image for one): each on the crest of the M0 ridge, continuous along it; never in a smear, under the
  dashed line, left of the dotted one, or jumping to a higher mode. A pick off its ridge is
  picked again, or its window rejected, whatever G3 said.
- Order of work: commit a line, read G3 and G4 grouped by flag and by stretches of windows; fix
  single windows for their fixable flags (G3's suggested change, a band cut at a jump, a narrower
  corridor, a lower coherence rule, a guide from the neighbours, a lasso when you see the images),
  every fix in one pick, each way with its windows (how fixes gives every window its own change
  G3 suggested, at once); when many windows fail for one reason,
  suspect the line's settings and go back with a hypothesis. A window's own pick is its recipe: a
  later commit picks it the same way.

## PACo's workflow, and where to beat it

- The workflow's order: the user's settings locked; the cone, the reach and G1 on the records;
  then PAC's page in its order: the windows (the farthest shot at the reach, the length by the
  ladder, no near field), the preprocessing (the mute trial at that length), the interferometry (a
  passive line's segments, sized with its length), the dispersion image (the band capped at the
  usable band); then the line loop: the changes the gates ask of the records and images (a mute, a
  trigger, a velocity grid), each tried on the ladder's 27 sample windows and kept when more pass,
  or as many with deeper curves, 4 at most; then each window's picks, stopping at 3·L, G3's
  retries and G4's guide.
- Its scores count G3's passes alone, on the same 27 sample windows in every trial. On active_p2
  in passive-active it covers 86 of 86 windows at 11 receivers, its picks at a median uncertainty
  of 23 %.
- Where you can do better: a length whose images are cleaner than its ladder's (the one G3 passes
  most of its 27 trial windows: 9 receivers on active_p2); on a tie, the mute whose curves reach
  deeper within reach (it keeps the tightest); a mute cutting one side only; nearer shots for the
  high frequencies (its distance_max is the reach); on a passive line, segment lengths other than
  its 10 crossings, and FK bands between its candidates; on a passive or passive-active line, the
  whitening, the normalization and the stacking's nu, which it leaves at the preset's; a score
  weighing coverage and depth within reach together, as the scorecard does. The grid tries them;
  advise gives PACo's choice, to compare with yours on the same windows.

## Diagnosis

- Few coherent columns, low prominence: noise or refractions. A mute, a shorter distance_max, a
  longer window, the root; on a passive or passive-active line, longer segments or a lower FK
  threshold.
- Stripes, fringes or a broken ridge along part of the line, with a low lateral coherence: the
  windows cross a lateral change or noisy traces there. Shorter windows; keep the longer ones
  only where their images stay clean.
- Competing ridges, a mode jump at high frequencies: a lower fmax or picking band, a longer
  window.
- A ridge on the grid's edge: a wider velocity grid; a peak at vmax under f = vmax/L is noise
  common to every trace, not the ridge.
- band_at_fmin with the same low frequency at every length: the source's band; accept the depth.
- A coherent band much narrower than the usable band: the near field at its low end (the
  near-field distance up), or attenuation at its high end (distance_max down).
- Curves beyond 3·L at the 40 % cap: the window resolves no such wavelength; a longer window for
  the depth where its images stay as clean, or cut the tail.
- Many windows failing one way along a stretch: the line's settings, or the site there (a buried
  change, a noise source): look at two images of the stretch before changing the line.

## Deciding and freezing

- Accept a window whose curve G3 and G4 did not reject, with no blocking flag: a flagged curve
  goes on, as every flagged unit; its flags and notes need your reason. Reject one only once tried
  hard: the line's pick, a pick after it when it carried fixable flags
  (or a reason naming them: why none applies), a guide from its neighbours when one side has two
  passing, a lasso when you see the images. decide refuses otherwise, saying what is owed; your
  own picks owe nothing more.
- Look at a window's image and pick before accepting it with flags or notes, or rejecting it: a
  curve G3 passed off its ridge is not accepted, and one it flagged or rejected on a clean ridge is
  worth a lasso.
- When every window is decided, freeze. Ask the user first when more than 30 % of the windows
  would be rejected.
