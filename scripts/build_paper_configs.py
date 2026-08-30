#!/usr/bin/env python3
"""Derive self-contained canonical configs from the frozen manifest snapshots.

Phase 1-1 of the release plan.

The paper's settings were never expressed in one place: a problem JSON supplied
the distributions, a shell script overrode several fields on the command line,
and environment variables overrode the script. This reads the resolved values
recorded by the runs themselves (frozen in Phase 0-2) and writes them into
configs that need no override at all.

Configs are named per paper artifact, not per problem/backend pair, because the
same problem and backend can carry different settings in different artifacts
(V-neck NN uses M_int=100 for the figures and 200 for the table).

Run from the repository root::

    uv run python scripts/build_paper_configs.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = REPO_ROOT / "configs" / "paper" / "manifest" / "snapshots"
PROBLEMS = REPO_ROOT / "configs" / "problems"
OUT_ROOT = REPO_ROOT / "configs" / "paper"

# Structural keys are copied from the problem JSON the run referenced; they
# describe the problem itself and are not recorded in a run's resolved config.
STRUCTURAL_KEYS = ("problem_type", "dim", "distributions", "base_functions", "visualization")

# Resolved Config field -> problem-JSON training key. Names diverge where the
# CLI layer renamed things; everything else maps onto itself.
RENAMED = {
    "n_interaction_steps": "interaction_steps",
    "learning_rate_value": "lr_value",
    "learning_rate_grad": "lr_grad",
    "learning_rate_interaction": "lr_interaction",
    "lr_decay_gamma": "lr_decay",
    "lr_decay_step": "lr_step_size",
    "use_multistep_td_loss": "use_multistep_td",
}
SAME = (
    "n_outer_iterations", "n_inner_steps", "n_agents", "n_agents_test", "T", "dt",
    "activation", "hidden_dims", "t_hidden_dims", "interaction_hidden_dims",
    "interaction_t_hidden_dims", "interaction_activation", "learn_interaction_drift",
    "learn_interaction_cost", "standardize_interaction_targets", "interaction_input_scale",
    "interaction_query_chunk_size", "interaction_ref_chunk_size", "interaction_time_block_size",
    "resample_interaction_sample", "evaluate_interaction_cpu", "sigma", "lambda_ipf",
    "lambda_td", "lambda_fk", "lambda_boundary", "td_horizon", "td_bootstrap_from_target",
    "fk_loss_norm", "detach_interaction_drift_divergence", "optimizer_type",
    "optimizer_momentum", "optimizer_weight_decay", "lr_scheduler_type", "grad_clip",
    "log_interval", "checkpoint_interval", "seed", "ema_decay", "ema_start_step",
    "ema_update_interval", "use_time_embedding", "time_input_mode", "time_embed_dim",
    "use_time_index_normalization", "time_index_interval",
)


def interactions_block(cfg: dict) -> dict:
    """Rebuild the `interactions` block from resolved values.

    The backend selection lives here rather than under `training`, which is why
    a random-batch run needs `interaction_backend` and `random_batch_size` in
    this block to be reproducible without a command-line override.
    """
    block: dict = {"interaction_backend": resolve_backend(cfg)}
    if cfg.get("random_batch_size") is not None:
        block["random_batch_size"] = cfg["random_batch_size"]
    block["drift_type"] = cfg.get("drift_type", "zero")
    drift = {k: cfg.get(f"int_drift_{k}") for k in ("weight", "sigma", "strength")}
    drift = {k: v for k, v in drift.items() if v is not None}
    if drift and block["drift_type"] != "zero":
        block["drift_params"] = drift
    block["cost_type"] = cfg.get("cost_type", "zero")
    cost = {k: cfg.get(f"int_cost_{k}") for k in ("weight", "sigma", "strength")}
    cost = {k: v for k, v in cost.items() if v is not None}
    if cost and block["cost_type"] != "zero":
        block["cost_params"] = cost
    return block


def training_block(cfg: dict) -> dict:
    """Rebuild the `training` block, dropping values the schema cannot carry."""
    out = {}
    for field in SAME:
        if cfg.get(field) is not None:
            out[field] = cfg[field]
    for field, key in RENAMED.items():
        if cfg.get(field) is not None:
            out[key] = cfg[field]
    # Runs record the device they happened to use; a released config must not
    # pin one, so the reproduction entry point resolves it instead.
    out["device"] = "auto"
    return out


def build(artifact: str, role: str, template: str, name: str, description: str) -> dict:
    """Assemble one canonical config from a snapshot plus its problem template.

    Provenance goes into `description`: the schema rejects unknown top-level
    keys, and a config that cannot say where it came from is what created this
    problem in the first place.
    """
    raw = snapshot(artifact, role)
    cfg = raw["config"]
    problem = json.loads((PROBLEMS / template).read_text())
    provenance = (f" [derived from run {raw['scenario_name']}/{raw['experiment_id']}; "
                  f"see configs/paper/manifest/MANIFEST.md]")
    out = {"name": name, "description": description + provenance}
    for key in STRUCTURAL_KEYS:
        if key in problem:
            out[key] = problem[key]
    out["interactions"] = interactions_block(cfg)
    out["training"] = training_block(cfg)
    # Key order the schema expects: problem_type first, then the rest.
    ordered = {k: out[k] for k in ("problem_type",) if k in out}
    ordered.update({k: v for k, v in out.items() if k != "problem_type"})
    return ordered


# Each entry is one paper artifact/role: the snapshot it derives from, the
# problem JSON supplying structure, and where the canonical config is written.
# Which problem JSON supplies the structural parts (distributions, obstacles,
# visualization limits) for each artifact family. Everything else is read from
# the run itself.
TEMPLATES = {
    ("gmm", "exact"): "gmm_navigation_gaussian.json",
    ("gmm", "nn"): "gmm_navigation_gaussian_learned.json",
    ("gmm", "random_batch"): "gmm_navigation_gaussian_no_obstacle.json",
    ("vneck", "exact"): "vneck_navigation.json",
    ("vneck", "nn"): "vneck_navigation_learned.json",
    ("vneck", "random_batch"): "vneck_navigation.json",
    ("opinion", "exact"): "opinion_1k_bounded_confidence_cost_off.json",
    ("opinion", "nn"): "opinion_1k_bounded_confidence_learned_cost_off.json",
    ("opinion", "random_batch"): "opinion_1k_bounded_confidence_cost_off.json",
}
# The GMM backend table used the no-obstacle problem; the rest use the obstacle one.
TEMPLATE_OVERRIDES = {
    ("gmm_backend_table", "nn"): "gmm_navigation_gaussian_learned_h64_no_obstacle.json",
    ("gmm_backend_table", "random_batch"): "gmm_navigation_gaussian_no_obstacle.json",
}

# Human-readable descriptions per artifact, filled in with the sweep value.
ARTIFACTS = {
    "gmm_main": "GMM navigation, {backend}. Figures gmm_nav_intdrift2.",
    "gmm_strength": "GMM interaction-strength sweep at w={w}, {backend}. Figure gmm_interaction_strength_compare.",
    "gmm_backend_table": "GMM backend comparison, {backend}. Table tab:gmm_backend_comparison; average over seeds 200-204.",
    "gmm_agents_sweep": "GMM runtime vs population size at N={n}, {backend}. Figure gmm_nav_intdrift2/time_sweep_time_per_iter.",
    "vneck_main": "V-neck with directional congestion, {backend}. Figures vneck_pedestrian_visual.",
    "vneck_strength": "V-neck congestion-strength sweep at w={w}, {backend}. Figure vneck_pedestrian_visual/*_timeseries.",
    "vneck_backend_table": "V-neck backend comparison, {backend}. Table tab:vneck_backend_comparison; average over seeds 310-317.",
    "vneck_agents_sweep": "V-neck runtime vs population size at N={n}, {backend}. Figure vneck_pedestrian_visual/time_per_iteration_vs_agents.",
    "opinion_main": "Opinion bounded confidence (d=10), {backend}. Figures opinion_bounded_confidence_n400.",
    "opinion_strength_sweep": "Opinion interaction-strength sweep at w={w}, {backend}. Figure opinion_bounded_confidence_strength_sweep.",
    "opinion_backend_table": "Opinion backend comparison, {backend}. Table tab:opinion_backend_comparison; average over seeds 43-45.",
    "opinion_agents_sweep": "Opinion runtime vs population size at N={n}, {backend}. Figure opinion_bounded_confidence_agents_sweep.",
}
FAMILY = {"gmm": "gmm", "vneck": "vneck", "opinion": "opinion"}


# Sweep points that were also run for a different number of outer iterations.
# Three opinion strength points have both a 30- and a 90-iteration run, so the
# iteration count has to appear in the name or the two would collide.
def sweep_label(artifact: str, cfg: dict, varying_outer: bool = False) -> str:
    """The part of a name that distinguishes one sweep point from its siblings.

    Seeds are deliberately not part of it: ``--seed`` is a published option, so
    runs that differ only by seed share a config.
    """
    parts = []
    if artifact.endswith("agents_sweep"):
        parts.append(f"n{cfg['n_agents']}")
    if artifact.endswith(("strength", "strength_sweep")):
        weight = cfg.get("int_cost_weight") if cfg.get("cost_type") != "zero" else cfg.get("int_drift_weight")
        parts.append("w" + f"{weight:g}".replace(".", "p"))
    if varying_outer:
        parts.append(f"it{cfg['n_outer_iterations']}")
    return ("_" + "_".join(parts)) if parts else ""


def resolve_backend(cfg: dict) -> str:
    """Which interaction backend a run used.

    Runs from before the backend field existed record it as null. For those, the
    learning flags decide, exactly as the CLI layer does: a run that trains a
    surrogate is a learned run, otherwise the sums were evaluated directly.
    Reading the null as "exact" would silently mislabel ten learned runs in the
    GMM population sweep.
    """
    backend = cfg.get("interaction_backend")
    if backend:
        return backend
    if cfg.get("learn_interaction_drift") or cfg.get("learn_interaction_cost"):
        return "nn"
    return "exact"


def outer_varies() -> set:
    """Sweep points whose runs disagree on the number of outer iterations."""
    seen: dict[str, set] = {}
    for path in sorted(SNAPSHOTS.glob("*.config.json")):
        artifact = path.name.split("__")[0]
        cfg = json.loads(path.read_text())["config"]
        family = next(f for f in FAMILY if artifact.startswith(f))
        backend = resolve_backend(cfg)
        key = f"{family}/{artifact[len(family) + 1:]}_{backend}{sweep_label(artifact, cfg)}"
        seen.setdefault(key, set()).add(cfg.get("n_outer_iterations"))
    return {k for k, v in seen.items() if len(v) > 1}


def collect() -> dict:
    """One entry per config the paper needs, keyed by its output path."""
    ambiguous = outer_varies()
    out = {}
    for path in sorted(SNAPSHOTS.glob("*.config.json")):
        artifact = path.name.split("__")[0]
        raw = json.loads(path.read_text())
        cfg = raw["config"]
        family = next(f for f in FAMILY if artifact.startswith(f))
        backend = resolve_backend(cfg)
        template = (TEMPLATE_OVERRIDES.get((artifact, backend))
                    or TEMPLATES[(family, backend)])

        kind = artifact[len(family) + 1:]
        base = f"{family}/{kind}_{backend}{sweep_label(artifact, cfg)}"
        rel = f"{family}/{kind}_{backend}{sweep_label(artifact, cfg, base in ambiguous)}"
        if rel in out:
            continue  # siblings differing only by seed share one config
        weight = cfg.get("int_cost_weight") if cfg.get("cost_type") != "zero" else cfg.get("int_drift_weight")
        description = ARTIFACTS[artifact].format(
            backend={"exact": "analytical evaluation", "nn": "learned surrogate",
                     "random_batch": "random batch"}[backend],
            w=f"{weight:g}" if weight is not None else "?", n=cfg.get("n_agents"))
        out[rel] = (raw, template, description)
    return out


def main() -> None:
    written = []
    for rel_path, (raw, template, description) in sorted(collect().items()):
        cfg = raw["config"]
        problem = json.loads((PROBLEMS / template).read_text())
        provenance = (f" [derived from run {raw['scenario_name']}/{raw['experiment_id']}; "
                      f"see configs/paper/manifest/MANIFEST.md]")
        config = {"name": rel_path.replace("/", "_"),
                  "description": description + provenance}
        for key in STRUCTURAL_KEYS:
            if key in problem:
                config[key] = problem[key]
        config["interactions"] = interactions_block(cfg)
        config["training"] = training_block(cfg)
        ordered = {k: config[k] for k in ("problem_type",) if k in config}
        ordered.update({k: v for k, v in config.items() if k != "problem_type"})

        out_path = OUT_ROOT / f"{rel_path}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")
        written.append(out_path.relative_to(REPO_ROOT))
    print(f"wrote {len(written)} canonical configs")


if __name__ == "__main__":
    main()
