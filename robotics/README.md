# SuperDex Robotics Docs Site

A [Docusaurus](https://docusaurus.io/) documentation site for **SuperDex
Robotics** (the `superdex.robotics` package).

SuperDex Robotics is a robotics SDK that provides robot definitions and
composition, controllers, sensors, actuators, and the framework that aggregates
them into complete simulation configs.

## Quick start

```bash
yarn install
yarn start
# → http://localhost:3000/
```

## Structure

- `docs/` — the Docs tab: `overview` + five guides (Bot Definition, Modifying
  Bots, Context/Lifetime, Controllers, Bot Assets).
- `docs/examples/` — the Examples section, derived from the Python examples
  in `superdex_robotics/examples/`.
- `docs/api_reference/{cpp,python}.mdx` — the API Reference tab: embeds generated
  Doxygen (C++) and Sphinx (Python) API docs in an iframe. The embed machinery is
  `src/components/api_reference.js` + `api_frame.js` + `static/api-embed.css` +
  `static/fonts/`.
- `src/pages/index.mdx` — the landing page.
- `src/css/custom.css` — theme (white/black, DM Sans + Instrument Serif).
- `docusaurus.config.js`, `sidebars.js` — site config and navigation.
