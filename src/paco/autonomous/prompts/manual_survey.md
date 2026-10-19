The survey (P1): what the line is before any setting is chosen. Call survey first, look at its
records, then write your conclusion in the journal: the curves' tools open once it is written.
What it measures bounds every later choice, and gives the first hypotheses.

## What the survey gives, and what it means

- The geometry: the receivers N_r and their spacing dx, the line's length, the shots (where, on
  which side, how far from the line), the sampling and its Nyquist, the records' length T, the
  trigger the headers state.
- The limits it sets. A wavelength λ = v/f under 2·dx aliases: at a velocity v, the frequencies
  above v/(2·dx). An image's frequency step is 1/T (the record after the mute; on a passive line,
  the segment's length), and a finer step only interpolates. Windows run from 5 receivers to half
  the line, N_r - N + 1 of them at a step of one receiver for a length of N.
- The reach: where the traces' median SNR, by distance from the shot, falls under G1's limit of
  2 dB. Past it a trace is mostly noise, and the phase shift gives every trace the same weight (it
  normalizes each trace's spectrum): traces past the reach add noise at full weight. The farthest
  shot a window takes (masw distance_max) starts at the reach.
- The surface waves' cone: the apparent velocities of the gathers' traces (each one's distance
  over the time of its envelope's peak), its slow edge, median and fast edge. Surface waves
  travel at about 0.9 of the shear velocity, slower than the direct and refracted P waves: what
  arrives faster than the cone is body waves and refractions, what arrives slower is noise. The
  cone is where a mute starts.
- Fast arrivals: G1's share of the energy arriving faster than the cone (refractions, head
  waves). Over 0.15 it asks for a mute; a strong head wave reads about 0.18, the real lines under
  0.1.
- The usable band (G1): where each record's spectrum stands above its noise. It measures the
  records' amplitude, while the phase shift normalizes each frequency: an image's ridge can stay
  coherent below the band's low edge. It bounds fmax (with Nyquist), not fmin.
- The records' quality: dead, clipped and NaN traces, left out by G1; a receiver off its
  neighbours' amplitude decay, left out of every window; the trigger error, corrected above 10 ms
  on muted records, a scatter of breaks over 50 ms rejecting the record; the lateral coherence.
- A passive line (no source positions) records ambient noise: no shot, so no reach nor cone, and
  no usable band from G1: its images keep the preset's 0 to 100 Hz. Its records' number and
  length are its data; its images' frequency step is 1/the segments' length (the preset's 1 s:
  1 Hz). G1 measures its traces (dead, clipped, NaN) and the receivers off their neighbours'
  spectra; G2 judges each window's correlations.
- The modes the line takes: an active line the active one, or passive-active (the shots beyond
  each window's ends correlated, as noise); a passive line the passive one alone.

## Looking at the records

The survey's numbers come from medians; the gathers show what they stand for. Before your
conclusion, when you can see:

- Look at the gathers of three shots, one at each end of the line and one in its middle (look
  gather), with the cone's slow and fast edges drawn (velocities). Name what you see, as the
  atlas's gathers do: the surface waves, the strong, slow fan, their low frequencies first and
  the train spreading with distance; the refractions, straight and fast ahead of it; the air
  wave, a straight line at 330 to 345 m/s; other sources' trains, with no moveout from the shot;
  dead, clipped or ringing traces.
- Set the mute by eye: draw a candidate on a gather (look gather with mute) and move its edges
  until its window holds the whole fan and little else: the fast edge just ahead of the fan's
  first energy at the far offsets, the slow edge just after its last, the width the train's tail.
  Check it on the other two gathers: one mute serves the whole line. Keep it, with the cone's
  and PACo's standard, as the candidates to try.
- Look at the spectra of the same records: where the surface waves' energy stands, at the far
  offsets too, bounds the image's fmax; the image's fmin goes lower than where it fades, since
  the phase shift normalizes each frequency.
- On a passive line no shot draws a fan: look at a record or two for dead or ringing traces and
  bursts of noise.

## Your conclusion

Write the survey in the journal, as a conclusion: what you saw on the gathers and spectra (the
surface waves' velocities, refractions, the air wave, noise, bad traces), the reach, the usable
band, the mutes you drew with their velocities, and the first hypotheses to test, a mute on the
surface waves' fan among them on a line with shots. They are where the line's settings start
from.
