# Reproducibility notes

Notes for anyone reproducing the results in *Nonlocal Mean Field Schrödinger
Bridge with Learned Interactions*. They cover what the published runs used, two
places where the code and the manuscript use different conventions, and what
re-running them on other hardware does and does not tell you.

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

## What reproducing a figure costs, and what is not shipped

**Trained weights are not published.** This repository carries the resolved
configuration and the recorded metrics of every run behind the paper
(`configs/paper/manifest/`), but not the checkpoints: they come to roughly 29 GB,
which does not belong in a git repository. Anything that replays particles from a
trained model — the trajectory figures, the PCA snapshots — therefore requires
training the run first.

The cost is not small. Reproducing the two GMM configurations behind
`fig:gmm_trajectory_comparison` on one NVIDIA RTX 6000 Ada took:

| | wall-clock |
|---|---|
| `configs/paper/gmm/main_exact.json` | 8.2 h |
| `configs/paper/gmm/main_nn.json` | 3.3 h |

That is one figure pair. The strength sweeps multiply this by the number of
interaction weights, and the backend tables by the number of seeds.

The ratio between those two numbers is itself the paper's main claim, and it
reproduced: the learned surrogate finished in a little over a third of the time
the exact evaluation needed, at a comparable analytical loss.

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

## Checkpoint selection

A full clean-tree campaign ran all 148 configurations to completion on an RTX
6000 Ada. All runs produced finite metrics, checkpoints and figures, and no run
exited with an error. The main trajectory and interaction-response conclusions
were reproduced.

The final checkpoint is not always the best checkpoint. Six of the 148 runs
ended at an analytical loss more than five times their best value, even though a
usable earlier checkpoint was present. We therefore do not rerun or replace
those experiments: figure and table scripts select the checkpoint with the best
analytical loss where checkpoint selection affects the reported artifact. Do
not assume that `checkpoint_final.pt` is the appropriate checkpoint for a new
plot or comparison.

## Hardware, and what that means for the timings

The published results came from two machines, as stated in the paper's appendix:
an NVIDIA L40S for the GMM experiments and an NVIDIA GeForce RTX 3090 for the
V-neck and opinion-dynamics experiments.

**The RTX 3090 environment is no longer available**, and its successor has
different GPUs.

Losses and interaction errors can be recomputed on other hardware: they are
properties of the trained model rather than of the machine. Do not expect them
to agree to the digit, though — GPU kernels, BLAS implementations and
random-number generation all differ across platforms, so runs on other hardware
will land nearby rather than on the published values.

**The published timing values cannot be reproduced under the hardware that
produced them**, since that environment no longer exists. The timing experiment
itself can be re-run on any GPU: what you get is a valid measurement for your
machine, not a check against the numbers in the paper, and timings from
different GPUs should not be compared with each other.

If you do re-run it, keep the comparison internal to one machine. What makes the
paper's timing results meaningful is not their absolute values but that each
backend pair ran on the same machine and device, so the ratio between backends
is the quantity that carries over.

## Requirements

Python 3.11 or 3.12. The pinned `torch<2.5` publishes wheels only for those
versions, so `uv sync` will refuse to install on 3.13.

## Attribution

The value and gradient networks (`ToyValueNet`, `ToyGradientNet` in
`src/nlmfsp/networks.py`) follow the architecture of the corresponding networks
in DeepGSB (Liu et al., 2022), which is distributed under the Apache License 2.0.
See `NOTICE`.
