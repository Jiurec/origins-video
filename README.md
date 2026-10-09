# origins-video
A cosmic timeline — programmatic motion graphics built with [Remotion](https://www.remotion.dev) (v4.0.534).

## Setup

Requires Node.js 18+ (22 LTS recommended).

```console
npm i
```

## Commands

| Task | Command |
| --- | --- |
| Open Remotion Studio (live preview) | `npm run dev` |
| Render a video | `npx remotion render Origins out/origins.mp4` |
| Render a single frame | `npx remotion still Origins out/frame.png --frame=60` |
| Lint + typecheck | `npm run lint` |
| Upgrade Remotion | `npm run upgrade` |

## Project layout

- `src/index.ts` — entry point (`registerRoot`)
- `src/Root.tsx` — registers all compositions
- `src/Composition.tsx` — the `Origins` starter composition (1920×1080, 30 fps)
- `public/` — static assets (images, audio, video), referenced with `staticFile()`
- `remotion.config.ts` — CLI config (Rspack bundler, JPEG frames)

Installed Remotion packages: `remotion`, `@remotion/cli`, `@remotion/transitions`,
`@remotion/shapes`, `@remotion/paths`, `@remotion/noise`, `@remotion/google-fonts`,
`@remotion/motion-blur`, `@remotion/media`. Add more with `npx remotion add <package>`
so versions stay in lockstep.

## Working with AI coding agents

The official Remotion Agent Skills (`remotion-dev/skills`) are installed in
`.agents/skills/` (read by Codex, Cursor, Gemini CLI, etc.) and symlinked into
`.claude/skills/` for Claude Code. See `AGENTS.md` for project conventions.
Update them with `npx skills update`.

## License

Remotion is free for individuals and teams of up to 3; larger companies need a
[company license](https://www.remotion.pro/license).
