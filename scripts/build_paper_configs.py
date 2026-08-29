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


def snapshot(artifact: str, role: str) -> dict:
    """Load the single snapshot for one artifact role."""
    matches = sorted(SNAPSHOTS.glob(f"{artifact}__{role}__*.config.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one snapshot for {artifact}/{role}, got {len(matches)}")
    return json.loads(matches[0].read_text())


def interactions_block(cfg: dict) -> dict:
    """Rebuild the `interactions` block from resolved values.

    The backend selection lives here rather than under `training`, which is why
    a random-batch run needs `interaction_backend` and `random_batch_size` in
    this block to be reproducible without a command-line override.
    """
    block: dict = {}
    backend = cfg.get("interaction_backend")
    if backend is not None:
        block["interaction_backend"] = backend
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
SPEC = [
    ("gmm/main_exact", "gmm_main", "exact", "gmm_navigation_gaussian.json",
     "GMM navigation, analytical interaction evaluation (w=2). Figures gmm_nav_intdrift2."),
    ("gmm/main_nn", "gmm_main", "nn", "gmm_navigation_gaussian_learned.json",
     "GMM navigation, learned surrogate (w=2, M_int=100). Figures gmm_nav_intdrift2."),
    ("gmm/backend_table_nn", "gmm_backend_table", "nn_seed200",
     "gmm_navigation_gaussian_learned_h64_no_obstacle.json",
     "GMM backend comparison, learned surrogate (M_int=200, width 64). Table tab:gmm_backend_comparison; "
     "sweep seed over 200-204."),
    ("gmm/backend_table_rb", "gmm_backend_table", "rb_seed200",
     "gmm_navigation_gaussian_no_obstacle.json",
     "GMM backend comparison, random batch (M_RB=128). Table tab:gmm_backend_comparison; "
     "sweep seed over 200-204."),
    ("gmm/strength_sweep_exact", "gmm_strength", "exact_w2", "gmm_navigation_gaussian.json",
     "GMM interaction-strength sweep, analytical. Figure gmm_interaction_strength_compare; "
     "sweep int_drift_weight over 1, 2, 4, 8."),
    ("gmm/strength_sweep_nn", "gmm_strength", "nn_w2", "gmm_navigation_gaussian_learned.json",
     "GMM interaction-strength sweep, learned surrogate. Figure gmm_interaction_strength_compare; "
     "sweep int_drift_weight over 1, 2, 4, 8."),

    ("vneck/main_exact", "vneck_main", "exact", "vneck_navigation.json",
     "V-neck with directional congestion, analytical evaluation. Figures vneck_pedestrian_visual."),
    ("vneck/main_nn", "vneck_main", "nn", "vneck_navigation_learned.json",
     "V-neck with directional congestion, learned surrogate (M_int=100). Figures vneck_pedestrian_visual."),
    ("vneck/backend_table_nn", "vneck_backend_table", "nn_seed310", "vneck_navigation_learned.json",
     "V-neck backend comparison, learned surrogate (M_int=200, width 64). "
     "Table tab:vneck_backend_comparison; sweep seed over 310-317."),
    ("vneck/backend_table_rb", "vneck_backend_table", "rb_seed310", "vneck_navigation.json",
     "V-neck backend comparison, random batch (M_RB=128). "
     "Table tab:vneck_backend_comparison; sweep seed over 310-317."),

    ("opinion/main_exact", "opinion_main", "exact", "opinion_1k_bounded_confidence_cost_off.json",
     "Opinion bounded confidence (d=10), analytical evaluation. Figures opinion_bounded_confidence_n400."),
    ("opinion/main_nn", "opinion_main", "nn", "opinion_1k_bounded_confidence_learned_cost_off.json",
     "Opinion bounded confidence (d=10), learned surrogate (M_int=100). "
     "Figures opinion_bounded_confidence_n400."),
    ("opinion/main_rb", "opinion_main", "rb", "opinion_1k_bounded_confidence_cost_off.json",
     "Opinion bounded confidence (d=10), random batch. Figures opinion_bounded_confidence_n400."),
    ("opinion/backend_table_nn", "opinion_backend_table", "nn_seed43",
     "opinion_1k_bounded_confidence_learned_cost_off.json",
     "Opinion backend comparison, learned surrogate (M_int=100). "
     "Table tab:opinion_backend_comparison; sweep seed over 43-45."),
    ("opinion/backend_table_rb", "opinion_backend_table", "rb_seed43",
     "opinion_1k_bounded_confidence_cost_off.json",
     "Opinion backend comparison, random batch (M_RB=80). "
     "Table tab:opinion_backend_comparison; sweep seed over 43-45."),
]


def main() -> None:
    written = []
    for rel_path, artifact, role, template, description in SPEC:
        name = rel_path.replace("/", "_")
        config = build(artifact, role, template, name, description)
        out_path = OUT_ROOT / f"{rel_path}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        written.append(out_path.relative_to(REPO_ROOT))
    print(f"wrote {len(written)} canonical configs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
