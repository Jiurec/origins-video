# Agent guide — origins-video

Remotion 4 project (React + TypeScript) for programmatic motion-graphics videos.

## Before writing video code
Load the Remotion skills in `.agents/skills/` (start with `remotion-best-practices`).

## Rules
- All animation must be driven by `useCurrentFrame()` / `useVideoConfig()` with
  `interpolate()` / `spring()`. Never use CSS transitions/animations, `setTimeout`,
  or `Math.random()` (use `random(seed)` from `remotion`) — renders must be deterministic.
- Use `<Sequence>` / `<Series>` / `@remotion/transitions` (`<TransitionSeries>`) for scenes.
- Register every composition in `src/Root.tsx`; keep one folder per scene/video under `src/`.
- Put assets in `public/` and reference them with `staticFile()`.
- Load fonts with `@remotion/google-fonts/<Font>` or `@remotion/fonts`.
- All `remotion` / `@remotion/*` packages must share the exact same version.
  Install new ones with `npx remotion add <pkg>`.

## Verify changes
- `npm run lint` (ESLint + tsc) must pass.
- Spot-check frames: `npx remotion still <CompId> out/frame.png --frame=<n>`.
- Only do a full render (`npx remotion render <CompId>`) when asked.
