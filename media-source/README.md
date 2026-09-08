# README media source

The media refresh consists of a wide generated hero, a 60-second motion-design
overview, a small animated README preview, and a visual transcript.

## Rebuild the motion assets

Requires Node.js, ffmpeg and ffprobe. The renderer uses Helvetica Neue on macOS,
with Arial and sans-serif fallbacks on other platforms. Exact typography may
differ on systems without these fonts.

```bash
cd media-source
npm ci
npm run stills  # seven review frames in .build/
npm run render
```

Rendering is entirely local. It does not start the router, read API credentials,
send inference requests or regenerate the hero.

Outputs:

- `../assets/darwin-router-overview.mp4`: 1920×1080, 30 fps, 60 seconds,
  H.264/yuv420p with AAC stereo and fast-start metadata.
- `../assets/darwin-router-preview.gif`: 960×540, 12 fps, a 7.2-second excerpt.

Temporary frames, the intermediate encode and generated WAV live in `.build/`
and are ignored by Git. `render.cjs` contains all layout, animation, timing and
procedural music synthesis. There is no external music or stock-footage dependency.

## Artwork and sources

The hero was generated using the built-in imagegen tool. The complete prompt is
saved in [hero-prompt.txt](hero-prompt.txt); the selected output is checked in as
`assets/darwin-router-hero.png`. Image generation is nondeterministic, so rebuilds
reuse the checked-in image.

The video combines original motion graphics and a newly synthesized instrumental
score with seconds 35–45 of the repository's original `assets/demo_video.mp4`.
The archival excerpt is cropped to the dashboard viewport and labelled as an
intermediate snapshot. It carries no original narration in this cut.

The 75% → 85%, $0.0022 → $0.0011 and 57 → 91 figures are the historical results
reported by the README at revision `d461e3c7fad2b44a9a32c7362135e51543971441`.
Implementation claims were checked against `router/index.py`, `router/judge.py`,
`router/proxy.py` and `router/replay.py`. This refresh does not rerun the benchmark.

The original poster, original demo and generated concept video remain available
in `assets/`. The README now leads with the new media and retains the original
narrated recording in a disclosure section.
