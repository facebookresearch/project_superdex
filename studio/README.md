# SuperDex Studio Docs Site

A [Docusaurus](https://docusaurus.io/) staticdocs site for **SuperDex Studio**
(the `superdex_studio` desktop app).

SuperDex Studio is a lightweight desktop GUI authoring and visualization
application for creating, editing, and validating the simulation assets —
robots, meshes, task prefabs, and scenes.

## Quick start

For most documentation changes, preview only this site. This is the fast path
and supports hot reloading:

```bash
yarn install  # Only when dependencies are not already installed
yarn start
# → http://localhost:3000/
```

The single-site preview defaults to `Latest`; when documentation snapshots are
present, use the version selector to preview them. Run `yarn build` when you
need to validate that this site compiles successfully for production.

Use the whole-site helper only when validation requires the Project SuperDex
homepage and all configured nested sites together, such as testing public base
paths, release routing, or cross-site links. It stages and builds every site and
can take several minutes. Run it from the fbsource root:

```bash
buck2 run fbsource//arvr/projects/superdex/ci:preview_website
```

## Structure

- `docs/` — the Docs tab: `overview` + the guides (User Interface, Bot Editor,
  IK Editor, Prefab Editor, Asset Tuner, Scene Editor, Model Editor,
  SuperDex CAD Exporter).
- `src/pages/index.mdx` — the landing page.
- `src/css/custom.css` — theme (white/black, DM Sans + Instrument Serif).
- `docusaurus.config.js`, `sidebars.js` — site config and navigation.
