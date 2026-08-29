import argparse
import json

import pytest
import torch

from nlmfsp import Config, FBSDESolver, Trainer
from nlmfsp.interaction import create_interactions
from nlmfsp.problem_setup import load_problem_config, setup_problem_artifacts


def _base_args(**overrides):
    base = {
        "dt": 0.1,
        "n_agents": 16,
        "n_agents_test": 16,
        "n_outer_iterations": 2,
        "n_inner_steps": 2,
        "lr_value": 1e-3,
        "lr_grad": 1e-3,
        "lr_interaction": 1e-3,
        "lambda_ipf": 1.0,
        "lambda_td": 1.0,
        "lambda_fk": 0.1,
        "lambda_boundary": 0.0,
        "td_horizon": 1,
        "td_bootstrap_from_target": False,
        "ema_decay": 0.0,
        "ema_start_step": 0,
        "ema_update_interval": 1,
        "sigma": 1.0,
        "hidden_dims": [32, 32],
        "activation": "tanh",
        "lr_decay": 0.999,
        "grad_clip": 1.0,
        "log_interval": 1,
        "checkpoint_interval": 10,
        "seed": 42,
        "device": "cpu",
        "interaction_steps": 2,
        "interaction_hidden_dims": None,
        "interaction_t_hidden_dims": None,
        "interaction_activation": None,
        "learn_interaction_drift": True,
        "learn_interaction_cost": True,
        "evaluate_interaction_cpu": False,
        "resample_interaction_sample": False,
        "t_hidden_dims": None,
        "fk_loss_norm": "l1",
        "use_time_embedding": False,
        "time_embed_dim": 128,
        "use_time_index_normalization": False,
        "time_index_interval": 100.0,
        "optimizer_type": "adam",
        "optimizer_momentum": None,
        "optimizer_weight_decay": 0.0,
        "lr_scheduler_type": "exponential",
        "lr_step_size": None,
        "drift_type": "mean_reversion",
        "cost_type": "quadratic_deviation",
        "int_drift_weight": 1.0,
        "int_drift_sigma": 1.0,
        "int_cost_weight": 1.0,
        "int_cost_sigma": 1.0,
        "int_drift_strength": None,
        "int_cost_strength": None,
        "base_drift_type": "zero",
        "base_cost_type": "zero",
        "base_cost_weight": None,
        "base_drift_value": [0.0, 0.0],
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_problem_config(tmp_path, *, training_extra=None, base_cost_extra=None):
    training = {
        "n_outer_iterations": 2,
        "n_inner_steps": 2,
        "T": 1.0,
        "dt": 0.1,
        "lr_value": 1e-3,
        "lr_grad": 1e-3,
        "sigma": 1.0,
        "lambda_ipf": 1.0,
        "lambda_td": 1.0,
        "lambda_fk": 0.1,
        "lambda_boundary": 0.0,
        "activation": "tanh",
        "hidden_dims": [32, 32],
        "n_agents": 16,
        "seed": 42,
        "device": "cpu",
    }
    if training_extra:
        training.update(training_extra)

    base_cost = {"type": "zero"}
    if base_cost_extra:
        base_cost.update(base_cost_extra)

    cfg = {
        "problem_type": "gaussian_transport",
        "name": "strict_test",
        "distributions": {
            "rho_0": {"type": "gaussian", "mean": [0.0, 0.0], "std": 1.0},
            "rho_1": {"type": "gaussian", "mean": [1.0, 1.0], "std": 1.0},
        },
        "interactions": {"drift_type": "zero", "cost_type": "zero"},
        "base_functions": {"drift": {"type": "zero"}, "cost": base_cost},
        "training": training,
        "visualization": {},
    }
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(cfg))
    return path


def test_load_problem_config_rejects_legacy_training_key(tmp_path):
    path = _write_problem_config(tmp_path, training_extra={"batch_size": 32})
    with pytest.raises(ValueError, match="batch_size"):
        load_problem_config(path)


def test_load_problem_config_rejects_unknown_nested_key(tmp_path):
    path = _write_problem_config(tmp_path, base_cost_extra={"legacy_flag": True})
    with pytest.raises(ValueError, match="legacy_flag"):
        load_problem_config(path)


def test_load_problem_config_rejects_removed_training_mode(tmp_path):
    path = _write_problem_config(tmp_path, training_extra={"training_mode": "alternating"})
    with pytest.raises(ValueError, match="training_mode"):
        load_problem_config(path)


def test_load_problem_config_rejects_unknown_base_drift_type(tmp_path):
    cfg_path = _write_problem_config(tmp_path)
    cfg = json.loads(cfg_path.read_text())
    cfg["base_functions"]["drift"]["type"] = "mystery"
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="base_functions\\.drift\\.type"):
        load_problem_config(cfg_path)


def test_load_problem_config_rejects_unknown_base_cost_type(tmp_path):
    cfg_path = _write_problem_config(tmp_path)
    cfg = json.loads(cfg_path.read_text())
    cfg["base_functions"]["cost"]["type"] = "mystery"
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="base_functions\\.cost\\.type"):
        load_problem_config(cfg_path)


def test_load_problem_config_requires_obstacle_mode_and_soft_beta(tmp_path):
    cfg_path = _write_problem_config(tmp_path)
    cfg = json.loads(cfg_path.read_text())
    cfg["base_functions"]["cost"] = {
        "type": "multi_obstacle",
        "obstacles": [
            {"center": [0.0, 0.0], "radius": 1.0, "penalty": 10.0},
        ],
    }
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="mode"):
        load_problem_config(cfg_path)

    cfg["base_functions"]["cost"]["obstacles"][0]["mode"] = "soft"
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="beta"):
        load_problem_config(cfg_path)


def test_interaction_label_fn_requires_time_argument():
    args = _base_args()
    config = Config(
        dim=2,
        T=1.0,
        dt=args.dt,
        n_agents=args.n_agents,
        learning_rate_value=args.lr_value,
        learning_rate_grad=args.lr_grad,
        learning_rate_interaction=args.lr_interaction,
        interaction_backend="nn",
        learn_interaction_drift=True,
        learn_interaction_cost=True,
    )
    _, _, f_interaction_fn, _, _, _ = create_interactions(config)
    xs = torch.randn(4, config.dim)
    empirical = torch.randn(4, config.dim)
    with pytest.raises(TypeError):
        f_interaction_fn(xs, empirical)


def test_base_cost_fn_requires_two_arguments():
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=4,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )
    solver = FBSDESolver(config)

    def base_cost_only_x(xs):
        return torch.zeros(xs.shape[0], device=xs.device)

    solver.base_cost_fn = base_cost_only_x

    x0 = torch.randn(4, 2)
    xs, ts, dws = solver.sample_forward_trajectory(x0, seed=123)

    with pytest.raises(TypeError):
        solver.compute_td_loss(
            xs,
            ts,
            dws,
            direction="backward",
        )


def test_setup_problem_artifacts_requires_problem_config_path(tmp_path):
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=16,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )
    with pytest.raises(ValueError, match="Problem configuration path"):
        setup_problem_artifacts(config)

    problem_path = _write_problem_config(tmp_path)
    config.problem_config_path = str(problem_path)
    artifacts = setup_problem_artifacts(config)
    assert artifacts[2] is not None
    assert artifacts[3] is not None


def test_setup_problem_artifacts_accepts_loaded_problem_config(tmp_path):
    config = Config(
        dim=2,
        T=1.0,
        dt=0.1,
        n_agents=16,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
    )
    problem_path = _write_problem_config(tmp_path)
    problem_cfg = load_problem_config(problem_path)

    artifacts = setup_problem_artifacts(config, problem_cfg)

    assert artifacts[2] is not None
    assert artifacts[3] is not None


def test_load_checkpoint_rejects_legacy_interaction_key(tmp_path):
    config = Config(
        dim=2,
        T=0.2,
        dt=0.1,
        n_agents=4,
        learning_rate_value=1e-3,
        learning_rate_grad=1e-3,
        checkpoint_dir=str(tmp_path),
        log_dir=str(tmp_path),
        output_dir=str(tmp_path),
        verbose=False,
    )
    solver = FBSDESolver(config)
    trainer = Trainer(config, solver)
    trainer.save_checkpoint(1)

    ckpt = tmp_path / "checkpoint_0001.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload["solver_state"]["interaction_net"] = {"legacy": torch.tensor([1.0])}
    torch.save(payload, ckpt)

    with pytest.raises(ValueError, match="interaction_net"):
        trainer.load_checkpoint(str(ckpt))
