The curves (P2): one set of settings for the whole line, the windows one receiver apart, each
stacking its shots by the same rule, every record and image alike; then each window's pick, a
decision on every window, and the freeze. Notation: dx the receiver spacing, N a window's
receivers, its length L = (N-1)·dx and its aperture N·dx, λ = v/f the wavelength of a point at
phase velocity v and frequency f.

## Where to start

- Your first commit is PACo's own choice for the line, the result to beat: the length its ladder
  chooses (advise window_length) and the mute its trial keeps (advise mute), or on a passive line
  its segments (advise segments), with the preset's other settings and the user's. The line then
  starts where the workflow ends, and every experiment runs against it as the current settings.
- Your first experiment tries against it the mute you drew by eye on the survey's gathers, PACo's
  standard mute (80 to 1,500 m/s) and lengths around PACo's. Commit a change only when its images
  are cleaner by eye, along the whole line and at its ends, and its scores no worse; else PACo's
  choice stays, and you say why it could not be beaten.

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
- PACo's ladder (advise window_length) tries 5, 7, 9 and 11 receivers on 27 windows spread along
  the line: it climbs while 80 % pass and the uncertainty improves by 10 %, and stops at the first
  length passing with 20 % or less. It favours short windows; where the objective asks depth, a
  longer window is the hypothesis to test against it.
- Choose the length by the images: the shortest window whose images are clean (the image score
  near its best, coherent columns and lateral coherence near 1) and whose curves serve the
  objective. A target depth z needs λmax near 2·z, so L of at least 2·z/3 (λmax within 3·L), but
  only as far as the images stay as clean; a lateral priority starts at 5 to 9 receivers; a
  balanced one in the middle of the range. Sweep the lengths around PACo's (its ladder's, and two
  longer) on 15 windows or more spread along the line, its ends included, each with and without
  the mute: length and mute interact (on active_p2 a mute moved PACo's ladder from 7 to 11
  receivers).
- After the picks, the card's checks: most curves stopping well under 2·L, at the same low
  frequency at every length, mean the data limit the depth: shorten. Many curves beyond 3·L where
  the target needs that depth: lengthen, or cut the tail (picking max_wavelength, in window
  lengths: 3 cuts at 3·L), or accept a shallower model and say so.

## Shots

- near_field_m: a window stacks no shot nearer than that to its nearest receiver, where it has a
  farther one; one with near shots only keeps them (G3 notes near_field). Keep the nearest shot
  at least λmax/2 from the window.
- masw distance_max: the farthest shot from a window's middle, at the survey's reach by default;
  the window's far half extends L/2 beyond it. A shorter one may help the high frequencies, which
  attenuate first. masw distance_min is a hard bound that loses windows: leave it to a user who
  asks for it.
- Shots on both sides average out dipping layers: keep both when the distances allow. More shots
  raise the SNR, not a near-field bias.
- PACo's rule sets the near-field distance at half the median λmax of the trial curves, once the
  length is chosen; dropping the near shots of every window cost a third of the demo's curves. A
  better one may come from each window's own λmax, read within its reach: try it on the windows
  G3 flags near_field and a control set. At the line's ends the rule may leave a window only a
  far, weak shot: a shorter distance gives the near shot back, at the price of a near-field bias
  at its longest wavelengths. Try it, and say which you keep.

## Preprocessing

- muting (vmin, vmax, width, taper) keeps the surface waves' cone, from x/vmax to x/vmin plus the
  pulse's width, and removes what blurs their ridge: the faster arrivals (direct and refracted P,
  head waves) and the slower noise. It is the first lever of a line with shots, and the
  surface-wave window of a passive-active line: try it in the first experiment, against no mute.
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
- Keeping a mute: PACo keeps one only with 2 more windows passing than no mute, a tie going to no
  mute; on active_p2 a tie kept no mute though the cone's curves reached 59 m against 28 m. Keep a
  mute that passes as many windows with longer curves within the windows' reach; drop one that
  loses windows, or whose median λmax falls under 0.8 of no mute's: muting too much cuts the far
  offsets and the low frequencies. In passive-active on active_p2, the standard mute passed 15
  trial windows against 11 without.
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
  frequencies, the rest of it dark, the pick on its crest: the atlas's first image.
- Each experiment shows its current settings and best variants side by side, each with its
  pick, at the windows where they differ most (look experiment E<n>:current,3,5 shows others).
  Judge each panel against the atlas, before its scores, and name its fault and cause before the
  lever: vertical stripes (a hard mute's edges, noisy or dead traces, a lateral change under the
  window); horizontal fringes under the ridge (a short window's side lobes, a hard mute); a ridge
  broken into pieces (a lateral change, noise); a broad smear at the low frequencies (the
  window's resolution, Δv/v ≈ λ/L: a longer window); no ridge at all (too little signal); bright
  ridges under the dashed line (aliases) or left of the dotted one (beyond the window's reach).
- The scorecard measures the same on every window: G2's coherent columns (the share of the
  image's columns with a coherent peak) and lateral coherence (how far the window's traces
  agree), near 1 when clean; G2's background (the image's median level off its ridge, over the
  ridge's peak) and its stripes (that level's jumps between columns), low when clean; the share
  of each pick on its ridge (G3's on_data). On active_p2 the clean images read a background of
  0.10 to 0.18 and stripes under 0.008, the striped ones up to 0.36 and 0.024. The score averages
  them, the background as 1 minus it; its P10, the weakest windows', shows a bad stretch the
  median hides. Rank many variants by it, then let your eyes choose among the two or three best.
- The measures cannot see everything: a short window's smear can score well where its columns
  peak, an arrival too fast for surface waves holds a high SNR and lateral coherence, a pick on
  a ridge that is not M0 is on its ridge. Where you see one, say so, and trust what you see.
- The image comes before the pick: a picking change cannot mend a bad image, only settings that
  change the image can (the length, the mute, the shots, the stacking).

## Passive and passive-active lines

- A passive line's windows correlate the noise of every record: each record cut into segments
  (slicing segment_duration, segment_step); each segment kept whole or dropped by its FK selection,
  which measures how much of its energy travels one way along the line, between the apparent
  velocities vmin and vmax, and keeps it over the threshold (a lower one keeps more); whitened and
  normalized (onebit: the spectral phase and the sign alone, so that noise bursts do not
  dominate); correlated against the window's first receiver; the correlations stacked: a virtual
  shot at the window's end. The shots' settings (distances, near field, trigger, mute) mean
  nothing there; the windows' length, the image's grid and the picking work as on an active line.
- The correlations' stacking, passive and passive-active alike: phase-weighted, the preset's (nu
  2). It weights the linear stack by how well the correlations agree in phase, so that coherent
  arrivals stay and incoherent noise falls. Keep it: linear or root replace it only when they
  clearly pass more windows with curves as long, and nu is a late, small lever.
- The segments set the image: its frequency step is 1/segment_duration, its longest period the
  segment's length. Longer segments give lower frequencies and a finer step, but fewer to stack, a
  noisier image: they are not more data. advise segments gives PACo's trial (a few minutes):
  lengths of 10 to 80 crossings of the window at 80 m/s, FK thresholds of 0.05 to 0.3 with no band
  or 80 to 1,500 m/s; a candidate counts with 1 % of its segments kept and the virtual shots of its
  two halves agreeing (0.7), and wins by the longest λmax/λmin. On passive_p2, 4.5 s segments with
  FK 0.2 in 80 to 1,500 m/s gave a span of 7.8 against 3.1.
- The FK selection decides whose noise each window stacks, and wins or loses a passive line's
  images: give it the most experiments. Sweep its threshold (0 to 1, the preset's 0.2: lower keeps
  more segments and their noise, higher fewer and cleaner, while enough remain to stack) and its
  velocity band (vmin and vmax: none, 80 to 1,500 m/s, a band around the surface waves the images
  show), in grids with the windows' length, and keep the variant whose images and virtual shot
  gathers are cleanest (look selection shows each segment's ratio against the threshold). Then
  the segments' length, then the window's length; the stacking stays phase-weighted.
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
- G3's near_field note on a passive window means nothing: its virtual source is its own first
  receiver. decide and freeze leave it out.
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
  inverse_dispersion, aliasing_zone, beyond_reach and near_field are notes.
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
  corridor, a lower coherence rule, a guide from the neighbours, a lasso when you see the images);
  when many windows fail for one reason, suspect the line's settings and go back with a
  hypothesis. A window's own pick is its recipe: a later commit picks it the same way.

## PACo's workflow, and where to beat it

- The workflow's order follows what depends on what: the user's settings locked; the cone, the
  reach (distance_max) and G1 on the records; the band capped at the usable band; the length by
  the ladder; the mute trial at that length; the near field from that length's curves; then the
  line loop: the changes the gates ask of the records and images (a mute, a trigger, a velocity
  grid), each tried on 15 windows, half asking, and kept with 2 more passing and at most 20 % of
  λmax lost, 4 at most; then each window's picks, G3's retries and G4's guide.
- Its scores count G3's passes alone, so it keeps short windows with loose picks, and a tie goes
  to the less cutting change. On active_p2 in passive-active it covers 86 of 86 windows, with its
  curves at the 40 % uncertainty cap and half their points beyond the windows' reach.
- Where you can do better: a longer length where the images stay as clean (its ladder stops at
  11 receivers); a mute kept on a tie when it lengthens the curves within reach; a mute cutting one
  side only; the near field from each window's own λmax within its reach; on a passive line,
  segment lengths and FK bands between its candidates; a score weighing coverage and depth within
  reach together, as the scorecard does. Each is a hypothesis: test it against PACo's choice on
  the same windows.

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
  the depth, or cut the tail.
- Many windows failing one way along a stretch: the line's settings, or the site there (a buried
  change, a noise source): look at two images of the stretch before changing the line.

## Deciding and freezing

- Accept a window whose curve passed G3 with no blocking flag, and G4; its notes need your reason.
  Reject one only once tried hard: the line's pick, a pick after it when it carried fixable flags
  (or a reason naming them: why none applies), a guide from its neighbours when one side has two
  passing, a lasso when you see the images. decide refuses otherwise, saying what is owed; your
  own picks owe nothing more.
- Look at a window's image and pick before accepting it with notes or rejecting it: a curve G3
  passed off its ridge is not accepted, and one it failed on a clean ridge is worth a lasso.
- When every window is decided, freeze. Ask the user first when more than 30 % of the windows
  would be rejected.
