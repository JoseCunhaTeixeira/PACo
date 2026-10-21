- target_depth_m: the depth the message asks the models to reach, in metres (down to 15 m: 15),
  else null.
- priority: what the message puts first: depth (as deep as possible), lateral (as much lateral
  detail as possible), or balanced; null when it says neither.
- effort: how much work the message asks: quick (a first look, quickly), thorough (as well as
  possible, take your time), or normal; null when it says none.
- new_study: true when the message asks to start over (a new study, from scratch, start again),
  else false.
- site_notes: what the message says about the ground (soft clay over chalk, water table near
  2 m), in its words, 200 characters at most, else null.
- vmin_m_s and vmax_m_s: the images' phase velocities the message gives (velocities up to
  350 m/s: vmax_m_s 350), else null.
- fmin_hz and fmax_hz: the images' frequency band it gives (images from 5 to 60 Hz: 5 and 60),
  else null.
- iterations: the inversion's iterations it gives (with 20,000 iterations: 20000), else null.
- vs_min_m_s and vs_max_m_s: the shear-wave velocities it allows the models (Vs between 150 and
  300 m/s: 150 and 300), else null.
- A setting the message does not type stays null: PACo keeps those it types as given.
