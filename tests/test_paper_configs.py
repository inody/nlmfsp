"""Tests for the canonical paper configs under ``configs/paper/``.

These configs are the released reproduction surface: each one corresponds to a
figure or table in the manuscript and is meant to be runnable with no command
line override at all. That property is what these tests protect, because the
settings behind the published results were previously spread across a problem
JSON, a shell script and environment variables.

See `the release plan` Phase 1.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nlmfsp.problem_setup import load_problem_config

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIGS = sorted((REPO_ROOT / "configs" / "paper").glob("*/*.json"))
ENTRY_POINT = REPO_ROOT / "examples" / "validate_generalized_schrodinger_bridge.py"
MANIFEST = REPO_ROOT / "configs" / "paper" / "manifest" / "MANIFEST.md"

# Values the manuscript states for each experiment (main.tex sections 5.1-5.3),
# checked against the configs so a silent drift cannot go unnoticed.
PAPER_VALUES = {
    "gmm": {"T": 1.0, "dt": 0.01, "n_agents": 1000},
    "vneck": {"T": 2.0, "dt": 0.01, "n_agents": 1000, "sigma": 1.0},
    "opinion": {"T": 3.0, "dt": 0.01, "n_agents": 400, "sigma": 1.0},
}


def config_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


@pytest.fixture(scope="module")
def configs():
    assert PAPER_CONFIGS, "no canonical paper configs found"
    return {config_id(p): json.loads(p.read_text()) for p in PAPER_CONFIGS}


@pytest.mark.parametrize("config_path", PAPER_CONFIGS, ids=config_id)
def test_paper_config_passes_strict_validation(config_path):
    """Every canonical config must satisfy the strict problem schema."""
    load_problem_config(config_path)


@pytest.mark.parametrize("config_path", PAPER_CONFIGS, ids=config_id)
def test_paper_config_is_self_contained(config_path):
    """The settings that used to come from shell overrides must be in the file.

    A config missing any of these would silently fall back to a default and stop
    describing the run it claims to reproduce.
    """
    config = json.loads(config_path.read_text())
    training = config.get("training", {})
    for key in ("n_agents", "T", "dt", "sigma", "n_outer_iterations", "n_inner_steps", "seed"):
        assert training.get(key) is not None, f"{config_path.name} does not pin '{key}'"

    interactions = config.get("interactions", {})
    assert interactions.get("interaction_backend") is not None, (
        f"{config_path.name} does not pin the interaction backend"
    )


@pytest.mark.parametrize("config_path", PAPER_CONFIGS, ids=config_id)
def test_paper_config_does_not_pin_a_device(config_path):
    """Device is resolved at run time; pinning one makes the config unportable.

    The published results came from GPUs that are no longer available, so a
    config that hardcodes ``cuda`` would fail for everyone else.
    """
    device = json.loads(config_path.read_text()).get("training", {}).get("device")
    assert device in (None, "auto"), f"{config_path.name} pins device={device!r}"


@pytest.mark.parametrize("config_path", PAPER_CONFIGS, ids=config_id)
def test_paper_config_records_its_provenance(config_path):
    """Each config must name the run it was derived from.

    The schema rejects extra top-level keys, so provenance lives in the
    description; losing it would recreate the problem these configs solve.
    """
    description = json.loads(config_path.read_text()).get("description", "")
    assert "derived from run" in description, (
        f"{config_path.name} does not record which run it came from"
    )


@pytest.mark.parametrize("experiment", sorted(PAPER_VALUES))
def test_configs_match_the_values_stated_in_the_manuscript(configs, experiment):
    """Configs for one experiment must agree with each other and with the paper."""
    expected = PAPER_VALUES[experiment]
    # Sweeps vary N or the interaction weight by construction, so the values the
    # manuscript quotes apply to the main configurations.
    relevant = {name: cfg for name, cfg in configs.items()
                if name.startswith(f"{experiment}/main_")}
    assert relevant, f"no main configs for {experiment}"

    for name, config in relevant.items():
        training = config["training"]
        for key, value in expected.items():
            assert training.get(key) == value, (
                f"{name}: {key}={training.get(key)!r}, manuscript states {value!r}"
            )


def test_backend_sizes_match_the_published_tables(configs):
    """M_int and M_RB are the quantities the backend tables are labelled with."""
    expected = {
        "gmm/backend_table_nn": ("interaction_steps", 200),
        "gmm/backend_table_random_batch": ("random_batch_size", 128),
        "vneck/backend_table_nn": ("interaction_steps", 200),
        "vneck/backend_table_random_batch": ("random_batch_size", 128),
        "opinion/backend_table_nn": ("interaction_steps", 100),
        "opinion/backend_table_random_batch": ("random_batch_size", 80),
    }
    for name, (key, value) in expected.items():
        config = configs[name]
        actual = config["training"].get(key, config["interactions"].get(key))
        assert actual == value, f"{name}: {key}={actual!r}, published table says {value!r}"


def test_manifest_exists_and_covers_every_config(configs):
    """Every config must trace back to a run recorded in the manifest."""
    assert MANIFEST.exists(), "manifest missing; run the manifest generator (kept in the development repository)"
    manifest_text = MANIFEST.read_text()
    for name, config in configs.items():
        description = config.get("description", "")
        exp_id = description.split("derived from run ")[1].split(";")[0].split("/")[-1]
        assert exp_id in manifest_text, f"{name}: run {exp_id} is not in the manifest"


@pytest.mark.slow
@pytest.mark.parametrize("config_path", PAPER_CONFIGS, ids=config_id)
def test_paper_config_runs_end_to_end(config_path, tmp_path):
    """Each config must actually drive the solver, not merely parse.

    Kept deliberately tiny: this checks the config reaches training, not that it
    reproduces published numbers, which needs hardware that no longer exists.
    """
    result = subprocess.run(
        [
            sys.executable, str(ENTRY_POINT),
            "--problem-config", str(config_path),
            "--n-outer-iterations", "1",
            "--n-inner-steps", "2",
            "--interaction-steps", "1",
            "--n-agents", "10",
            "--n-agents-test", "10",
            "--log-interval", "1",
            "--checkpoint-interval", "9999",
            "--skip-plots",
            "--no-experiment-dir",
            "--device", "cpu",
        ],
        capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"{config_id(config_path)} failed:\n"
        f"STDOUT:\n{result.stdout[-3000:]}\nSTDERR:\n{result.stderr[-3000:]}"
    )
