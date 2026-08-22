# SuperDex Studio Docs Site

A [Docusaurus](https://docusaurus.io/) staticdocs site for **SuperDex Studio**
(the `superdex_studio` desktop app).

SuperDex Studio is a lightweight desktop GUI authoring and visualization
application for creating, editing, and validating the simulation assets —
robots, meshes, task prefabs, and scenes.

## Quick start

```bash
yarn install
yarn start
# → http://localhost:3000/
```

## Structure

- `docs/` — the Docs tab: `overview` + five guides (User Interface, Bot Editor,
  Prefab Editor, Model Editor, SuperDex CAD Exporter).
- `src/pages/index.mdx` — the landing page.
- `src/css/custom.css` — theme (white/black, DM Sans + Instrument Serif).
- `docusaurus.config.js`, `sidebars.js` — site config and navigation.
