# NLMFSP

Code accompanying the preprint [*Nonlocal Mean Field Schrödinger Bridge with
Learned Interactions*](https://arxiv.org/abs/2606.04265).

## What this is

The mean-field Schrödinger bridge steers a population from one distribution to
another at minimum energy. When the particles interact nonlocally, evaluating
the interaction terms means summing over every pair, which costs $O(N^2)$ per
step and puts large populations out of reach.

This repository implements a four-stage alternating scheme that replaces those
sums with neural surrogates $\tilde f_\psi(x,t)$ and $\tilde F_\psi(x,t)$,
trained on empirical interaction values along sampled trajectories. Each outer
iteration updates the backward potential, the forward surrogate, the forward
potential, and the backward surrogate in turn. The drift updates then evaluate
the interactions in $O(N)$.

Three ways of evaluating the interactions are provided, selected in the config:

| backend | cost per step | what it does |
|---|---|---|
| `exact` | $O(N^2)$ | pairwise sums over all particles |
| `nn` | $O(N)$ | the learned surrogates, this paper's method |
| `random_batch` | $O(NM)$ | the same sums over a random subset of size $M$ |

## Installation

Requires Python 3.11 or 3.12 — the pinned `torch<2.5` publishes wheels only for
those versions.

```bash
git clone https://github.com/inody/nlmfsp
cd nlmfsp
uv sync
```

## Quick example

```bash
uv run python scripts/reproduce.py --list
uv run python scripts/reproduce.py --config configs/paper/gmm/main_exact.json
```

`scripts/reproduce.py` is the entry point. It takes `--config`, `--seed`,
`--device`, `--name` and `--resume`; everything else about a run comes from its
config file.

## Reproducing the figures and tables

Each figure and table in the paper has a config under `configs/paper/`, holding
every setting that artifact needs:

```
configs/paper/{gmm,vneck,opinion}/{main_*,backend_table_*,strength_sweep_*}.json
```

Run one directly:

```bash
uv run python scripts/reproduce.py --config configs/paper/vneck/main_nn.json
```

The backend-comparison tables average several seeds. For the V-neck table:

```bash
for SEED in 310 311 312 313 314 315 316 317; do
    uv run python scripts/reproduce.py \
        --config configs/paper/vneck/backend_table_nn.json --seed "$SEED"
    uv run python scripts/reproduce.py \
        --config configs/paper/vneck/backend_table_random_batch.json --seed "$SEED"
done
uv run python scripts/render_vneck_pedestrian_visual_backend_table.py \
    --seeds 310 311 312 313 314 315 316 317
```

`configs/paper/manifest/MANIFEST.md` records which run produced each artifact
and the settings it used, so the configs state what actually ran rather than
what was intended.

**Before running anything, read [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).**
Trained weights are not published, so a trajectory figure means training the run
first: the two GMM configurations above took 8.2 h and 3.3 h on one RTX 6000 Ada.
That file also covers what re-running on other hardware does and does not tell
you — in particular the wall-clock timings, which came from a GPU that is no
longer available — and two places where the code and the manuscript use
different conventions.

## Repository structure

```
configs/paper/       one self-contained config per figure and table
  manifest/          the runs that produced the published results
scripts/
  reproduce.py       the entry point
  plot_*.py          figure generation
  render_*.py        table generation
src/nlmfsp/
  fbsde.py           the FBSDE solver
  trainer.py         the four-stage alternating loop
  interaction.py     exact, learned and random-batch backends
  networks.py        value, gradient and surrogate networks
  drifts.py          nonlocal drift kernels
  costs.py           nonlocal cost kernels
tests/               regression coverage; run with `uv run pytest`
examples/            the training engine reproduce.py drives
```

## Citation

```bibtex
@article{inoue2026nonlocal,
  title   = {Nonlocal Mean Field {S}chr\"{o}dinger Bridge with Learned Interactions},
  author  = {Inoue, Daisuke and Kalise, Dante and Lauri\`{e}re, Mathieu},
  journal = {arXiv preprint arXiv:2606.04265},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.04265},
  url     = {https://arxiv.org/abs/2606.04265}
}
```

`CITATION.cff` carries the same record in machine-readable form.

## License

Apache License 2.0; see [`LICENSE`](LICENSE). Parts of `src/nlmfsp/networks.py`
are adapted from [DeepGSB](https://github.com/ghliu/DeepGSB) (Liu et al., 2022),
also Apache-2.0 — see [`NOTICE`](NOTICE).
