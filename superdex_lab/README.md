# SuperDex Lab

[SuperDex Lab](https://projectsuperdex.com/lab/) provides Gymnasium-based
environment and benchmarking surfaces for Project SuperDex.

Provides the `superdex.lab` package: reinforcement-learning environments, reference
benchmark tasks, and the vectorized runners built on `superdex-robotics`.

```bash
pip install superdex-lab
```

The benchmark assets are not included with `pip install superdex-lab`. Clone the
[Project SuperDex repository](https://github.com/facebookresearch/project_superdex)
and set `SUPERDEX_ASSETS_PATH=<path-to-project_superdex>/assets` before running a
benchmark environment. See the
[setup guide](https://projectsuperdex.com/lab/docs/superdex_gym/setup/) for complete
instructions.

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## License

First-party code in this distribution is Apache-2.0 licensed; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
Third-party dependencies retain their own terms.
