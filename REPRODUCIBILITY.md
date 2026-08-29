# Reproducibility notes

Notes for anyone reproducing the results in *Nonlocal Mean Field Schrödinger
Bridge with Learned Interactions*. They cover what the published runs used, two
places where the code and the manuscript use different conventions, and what can
and cannot be reproduced on other hardware.

## Where the settings live

Every figure and table in the paper has a config under `configs/paper/`, and each
config carries all of the settings that artifact needs:

```bash
uv run python scripts/reproduce.py --list
uv run python scripts/reproduce.py --config configs/paper/vneck/main_nn.json
```

`configs/paper/manifest/MANIFEST.md` maps each artifact to the run that produced
it and records that run's resolved configuration and metrics. The configs were
derived from those runs, so they state what actually executed rather than what
was intended.

## Surrogate learning rate

The surrogate networks use a learning rate of $5\times10^{-4}$ in the GMM and
opinion-dynamics experiments and $10^{-3}$ in the V-neck experiment. Revision R07
of the manuscript stated $5\times10^{-4}$ for all three; R08 corrects this. The
configs in this repository record the values that were used.

## Finite-particle convention for the exact interaction

For the V-neck directional congestion cost, the exact backend evaluates a
leave-one-out average, excluding the particle itself:

$$F_i = \frac{1}{N-1}\sum_{j\ne i} K(x_i, x_j),$$

implemented in `PedestrianVisualCost` (`src/nlmfsp/costs.py`). The manuscript
writes the empirical interaction as $\frac1N\sum_{j=1}^N K(x_i,x_j)$, which
includes the self term. The random-batch backend draws a uniform subset
$\mathcal B$ and computes $\frac{1}{M_{\mathrm{RB}}}\sum_{j\in\mathcal B}$, so it
is unbiased for the manuscript's form.

The two conventions differ by

$$\frac{K(x_i,x_i) - F_i}{N} = \frac{0.5 - F_i}{N},$$

bounded by $5\times10^{-4}$ at $N=1000$, roughly one percent of the interaction
error reported for the V-neck backend comparison. This is an implementation
detail of order $N^{-1}$; it does not affect the mean-field formulation or any
conclusion in the paper, and neither the code nor the published results were
changed on account of it.

This applies only to `PedestrianVisualCost`. `GaussianConvolutionMeanCost`
divides by the full reference count, and no drift in `src/nlmfsp/drifts.py`
implements a leave-one-out path, so the opinion random-batch runs use the same
convention in both backends.

## Reproducing the backend comparison tables

The tables average several seeds. The V-neck table uses seeds 310–317 with
population standard deviations:

```bash
uv run python scripts/render_vneck_pedestrian_visual_backend_table.py \
    --seeds 310 311 312 313 314 315 316 317
```

This reproduces the published table exactly from the runs recorded in the
manifest. The seed sets for the other tables are listed in `MANIFEST.md`.

## Hardware, and what that means for the timings

The published results came from two machines, as stated in the paper's appendix:
an NVIDIA L40S for the GMM experiments and an NVIDIA GeForce RTX 3090 for the
V-neck and opinion-dynamics experiments.

**The RTX 3090 environment is no longer available**, and its successor has
different GPUs. Losses and interaction errors are reproducible anywhere, since
they do not depend on the hardware. **Wall-clock timings are not**: they can be
re-measured, but the numbers will not match the paper, and comparing them across
different GPUs is not meaningful. The timing comparisons in the paper are
internally consistent, in that each backend pair ran on the same machine and
device; that is what makes them a fair comparison rather than their absolute
values.

## Requirements

Python 3.11 or 3.12. The pinned `torch<2.5` publishes wheels only for those
versions, so `uv sync` will refuse to install on 3.13.

## Attribution

The value and gradient networks (`ToyValueNet`, `ToyGradientNet` in
`src/nlmfsp/networks.py`) follow the architecture of the corresponding networks
in DeepGSB (Liu et al., 2022), which is distributed under the Apache License 2.0.
See `NOTICE`.
