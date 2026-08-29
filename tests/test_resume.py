"""Release gate for the ``--resume`` path exposed by ``scripts/reproduce.py``.

The canonical-config tests cover a run from start to finish. Resume is a
separate code path: it restores the configuration from a checkpoint rather than
from the config file, and it decides how many outer iterations are left. Since
``--resume`` is part of the published command line, it needs its own check.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = REPO_ROOT / "examples" / "validate_generalized_schrodinger_bridge.py"
REPRODUCE = REPO_ROOT / "scripts" / "reproduce.py"
CONFIG = REPO_ROOT / "configs" / "paper" / "gmm" / "main_exact.json"

# Deliberately tiny: this checks that training resumes, not that it converges.
SMALL_RUN = [
    "--n-agents", "10", "--n-agents-test", "10",
    "--n-inner-steps", "2", "--n-outer-iterations", "2",
    "--checkpoint-interval", "1", "--log-interval", "1",
    "--skip-plots", "--device", "cpu",
]


def run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True,
                          timeout=timeout, cwd=REPO_ROOT)


@pytest.mark.slow
def test_interrupted_run_resumes_to_completion(tmp_path):
    scenario = f"resume_gate_{tmp_path.name}"
    experiments = REPO_ROOT / "outputs" / "experiments" / scenario

    started = run([sys.executable, str(ENGINE), "--problem-config", str(CONFIG),
                   "--name", scenario, *SMALL_RUN])
    assert started.returncode == 0, f"initial run failed:\n{started.stderr[-2000:]}"

    exp_dirs = sorted(experiments.glob("exp_*"))
    assert len(exp_dirs) == 1, f"expected one experiment directory, got {exp_dirs}"
    checkpoints = exp_dirs[0] / "checkpoints"
    assert (checkpoints / "checkpoint_0002.pt").exists(), "initial run produced no second checkpoint"

    # Simulate a run that was killed after the first outer iteration: drop
    # everything the second iteration wrote, keeping the first checkpoint.
    (checkpoints / "checkpoint_0002.pt").unlink()
    (checkpoints / "checkpoint_final.pt").unlink(missing_ok=True)

    resumed = run([sys.executable, str(REPRODUCE), "--config", str(CONFIG),
                   "--resume", str(exp_dirs[0]), "--device", "cpu"])
    assert resumed.returncode == 0, f"resume failed:\n{resumed.stderr[-2000:]}"

    combined = resumed.stdout + resumed.stderr
    assert "Resume Mode" in combined, "resume did not take the resume path"
    assert "Remaining: 1" in combined, (
        "resume should have had exactly one outer iteration left; "
        f"output was:\n{combined[-2000:]}"
    )
    assert (checkpoints / "checkpoint_0002.pt").exists(), (
        "resume did not write the checkpoint for the remaining iteration"
    )
