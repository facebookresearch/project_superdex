# SuperDex Lab Docs Site

A [Docusaurus](https://docusaurus.io/) site for **SuperDex Lab**
(`superdex_lab`).

SuperDex Lab is the simulation harness that abstracts the Markov decision
process and dynamics-constrained-optimization underpinning RL, MPC, and
system-ID.

## Quick start

```bash
yarn install
yarn start
# → http://localhost:3000/
```

Set `SUPERDEX_PUBLIC_BUILD=1` to switch the site off its internal URLs; without
it, canonical URLs and cross-site links resolve only inside Meta.

The flag changes URL wiring only, so a local run previews that wiring rather
than the published site: the deploy workflow (`.github/workflows/pages.yml`)
also sets `SUPERDEX_PUBLIC_ORIGIN` and `SUPERDEX_PUBLIC_BASE_URL`, which locally
default to `https://projectsuperdex.com` and `/`, and the published site is
built from the exported tree rather than from this one.

Use it with `yarn start`, not `yarn build`. A public **build** refuses to run
while `docs/internal/` is present, because the site's preset copies `docs/`
verbatim into `build/_src/` and would publish the internal pages; the published
site is built from the exported tree, where ShipIt has already removed them.

```bash
# macOS / Linux
SUPERDEX_PUBLIC_BUILD=1 yarn start
# → http://localhost:3000/lab/   (the flag moves baseUrl to /lab/)

# Windows PowerShell
$env:SUPERDEX_PUBLIC_BUILD = '1'; yarn start
```

## Structure

- `docs/overview.mdx` — the Docs landing page: an overview of the SuperDex Lab
  simulation harness and its components.
- `docs/superdex_gym/` — the SuperDex Gym docs: intro, setup, environment
  reference, examples, benchmarking, custom environments, batching, RLlib
  training, rendering, and training-history visualization.
- `src/pages/index.mdx` — the landing page (an overview of all SuperDex Lab
  features).
- `src/css/custom.css` — theme (white/black, DM Sans + Instrument Serif).
- `docusaurus.config.js`, `sidebars.js` — site config and navigation.
