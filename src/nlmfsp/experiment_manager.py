"""Experiment management for organizing and tracking training runs.

This module provides utilities for creating organized experiment directories,
saving configurations, and managing multiple runs of the same scenario.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config, migrate_legacy_config_dict


class ExperimentManager:
    """Manage experiment directories and metadata.

    Creates a hierarchical directory structure:
        outputs/experiments/{scenario_name}/{experiment_id}/
            ├── config.json
            ├── metrics.json
            ├── checkpoints/
            ├── figures/
            └── logs/

    Example:
        >>> config = Config(...)
        >>> manager = ExperimentManager(
        ...     config,
        ...     scenario_name="option_a_gaussian_transport"
        ... )
        >>> manager.save_config(config, cli_args)
        >>> trainer = Trainer(config, solver, experiment_dir=manager.exp_dir)
    """

    def __init__(
        self,
        config: Config,
        scenario_name: Optional[str] = None,
        method: str = "analytical",
        auto_timestamp: bool = True,
    ):
        """Initialize experiment manager.

        Args:
            config: Configuration object
            scenario_name: Name of the experiment scenario (e.g., "gaussian_transport").
                If None, will be auto-generated from config.
            method: Legacy method grouping or canonical backend identifier.
                Accepted legacy values are "analytical" / "learned".
                Canonical backend values are "exact" / "nn" / "random_batch".
            auto_timestamp: If True, add timestamp to experiment ID
        """
        self.config = config
        if method in {"exact", "nn", "random_batch"}:
            self.interaction_backend = str(method)
        else:
            backend = str(getattr(config, "interaction_backend", "exact"))
            if backend not in {"exact", "nn", "random_batch"}:
                backend = "nn" if method == "learned" else "exact"
            self.interaction_backend = backend
        self.method_group = "learned" if self.interaction_backend == "nn" else "analytical"
        self.method = self.interaction_backend
        self.auto_timestamp = auto_timestamp

        if not scenario_name:
            raise ValueError("scenario_name is required for ExperimentManager.")

        self.scenario_name = scenario_name

        # Scenario names should be fully specified by config / CLI.
        self.full_scenario_name = self.scenario_name

        # Generate experiment ID
        self.exp_id = self._generate_experiment_id()

        # Setup directory paths
        self.experiment_root = Path(config.experiment_root)
        self.scenario_dir = self.experiment_root / self.full_scenario_name
        self.exp_dir = self.scenario_dir / self.exp_id

        # Create directory structure
        self._setup_directories()

        # Update config paths to point to experiment directory
        self._update_config_paths()

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID.

        Returns:
            Experiment ID string. By default this is timestamp-only,
            e.g. ``exp_20260311_064431``.
        """
        if self.auto_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"exp_{timestamp}"
        return "exp_manual"

    def _setup_directories(self) -> None:
        """Create experiment directory structure."""
        # Create main experiment directory
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)
        (self.exp_dir / "figures").mkdir(exist_ok=True)
        (self.exp_dir / "logs").mkdir(exist_ok=True)

        print(f"\nExperiment Directory Created:")
        print(f"  Scenario: {self.full_scenario_name}")
        print(f"  Exp ID:   {self.exp_id}")
        print(f"  Path:     {self.exp_dir}")

    def _update_config_paths(self) -> None:
        """Update config output paths to point to experiment directory."""
        self.config.output_dir = str(self.exp_dir)
        self.config.checkpoint_dir = str(self.exp_dir / "checkpoints")
        self.config.log_dir = str(self.exp_dir / "logs")

    @staticmethod
    def _collect_provenance() -> Dict[str, Any]:
        """Record what produced a run: code revision, machine, and versions.

        Runs made before this existed cannot be tied to a revision at all, which
        is why the paper's provenance had to be reconstructed from timestamps.
        ``git_dirty`` matters as much as the commit: a hash alone does not
        identify the code when the working tree had uncommitted changes.
        """
        provenance: Dict[str, Any] = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }

        repo_root = Path(__file__).resolve().parents[2]

        def git(*args: str) -> Optional[str]:
            try:
                result = subprocess.run(
                    ["git", *args], cwd=repo_root, capture_output=True,
                    text=True, timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            return result.stdout.strip() if result.returncode == 0 else None

        commit = git("rev-parse", "HEAD")
        provenance["git_commit"] = commit
        status = git("status", "--porcelain")
        # None means git was unavailable or this is not a checkout; in that case
        # dirtiness is unknown rather than false.
        provenance["git_dirty"] = None if status is None else bool(status)

        try:
            import torch

            provenance["torch_version"] = torch.__version__
            provenance["cuda_version"] = torch.version.cuda
            if torch.cuda.is_available():
                provenance["gpu"] = torch.cuda.get_device_name(0)
                provenance["gpu_count"] = torch.cuda.device_count()
        except Exception:
            pass

        return provenance

    def save_config(
        self,
        config: Config,
        cli_args: Optional[Any] = None,
    ) -> None:
        """Save experiment configuration to JSON.

        Uses Config.to_dict() for automated parameter serialization, eliminating
        the need for manual parameter listing and ensuring new config fields
        are automatically included.

        Args:
            config: Configuration object to save
            cli_args: Optional command-line arguments (argparse.Namespace)
        """
        config_dict = {
            "experiment_id": self.exp_id,
            "scenario_name": self.full_scenario_name,
            "interaction_backend": self.interaction_backend,
            "method": self.method,
            "method_group": self.method_group,
            "timestamp": datetime.now().isoformat(),
            "provenance": self._collect_provenance(),
            "config": config.to_dict(),
        }

        # Add CLI arguments if provided
        if cli_args is not None:
            cli_args_dict = vars(cli_args).copy()
            # Remove non-serializable objects (ArgumentParser)
            cli_args_dict.pop("_parser", None)
            config_dict["cli_args"] = cli_args_dict
        else:
            cli_args_dict = {}

        # Merge config + CLI for convenience (CLI overrides)
        merged_config: Dict[str, Any] = config_dict["config"].copy()
        merged_config.update(cli_args_dict)
        config_dict["config_merged"] = merged_config

        # Save to file
        config_path = self.exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"  Config saved: {config_path}")

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save final metrics to JSON.

        Args:
            metrics: Dictionary of metrics to save
        """
        metrics_path = self.exp_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Metrics saved: {metrics_path}")

    def create_scenario_summary(self) -> None:
        """Create or update summary of all experiments in this scenario.

        Scans all experiment directories and creates a summary table.
        """
        summary = {
            "scenario_name": self.full_scenario_name,
            "interaction_backend": self.interaction_backend,
            "method": self.method,
            "method_group": self.method_group,
            "last_updated": datetime.now().isoformat(),
            "experiments": [],
        }

        # Scan all experiment directories
        if self.scenario_dir.exists():
            for exp_dir in sorted(self.scenario_dir.iterdir()):
                if not exp_dir.is_dir() or exp_dir.name == "summary.json":
                    continue

                config_path = exp_dir / "config.json"
                metrics_path = exp_dir / "metrics.json"

                exp_info = {
                    "experiment_id": exp_dir.name,
                    "has_config": config_path.exists(),
                    "has_metrics": metrics_path.exists(),
                }

                # Load config if exists
                if config_path.exists():
                    with open(config_path) as f:
                        exp_config = json.load(f)
                        exp_info["config"] = exp_config.get("config", {})
                        exp_info["timestamp"] = exp_config.get("timestamp", "")
                        cfg = exp_config.get("config", {})
                        backend = exp_config.get("interaction_backend") or cfg.get(
                            "interaction_backend"
                        )
                        if backend not in {"exact", "nn", "random_batch"}:
                            if cfg.get("learn_interaction_drift") or cfg.get(
                                "learn_interaction_cost"
                            ) or exp_config.get("method") == "learned":
                                backend = "nn"
                            elif cfg.get("random_batch_size") is not None:
                                backend = "random_batch"
                            else:
                                backend = "exact"
                        exp_info["interaction_backend"] = backend
                        exp_info["method_group"] = exp_config.get(
                            "method_group",
                            "learned"
                            if exp_info["interaction_backend"] == "nn"
                            else "analytical",
                        )

                # Load metrics if exists
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        exp_metrics = json.load(f)
                        exp_info["final_losses"] = exp_metrics.get("final_losses", {})
                        exp_info["transport_quality"] = exp_metrics.get(
                            "transport_quality", {}
                        )

                summary["experiments"].append(exp_info)

        # Save summary
        summary_path = self.scenario_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nScenario summary updated: {summary_path}")
        print(f"  Total experiments: {len(summary['experiments'])}")

    def get_experiment_path(self, subdir: str = "") -> Path:
        """Get path to a subdirectory within the experiment.

        Args:
            subdir: Subdirectory name (e.g., "figures", "checkpoints")

        Returns:
            Path to the subdirectory
        """
        if subdir:
            return self.exp_dir / subdir
        return self.exp_dir

    @classmethod
    def from_existing(cls, exp_dir: Path | str) -> ExperimentManager:
        """Load existing experiment manager from directory.

        Args:
            exp_dir: Path to existing experiment directory

        Returns:
            ExperimentManager instance for the existing experiment

        Raises:
            FileNotFoundError: If experiment directory or config.json doesn't exist
        """
        exp_dir = Path(exp_dir)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        # Load config.json
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_data = json.load(f)

        # Reconstruct Config object
        config_dict = config_data.get("config", {})

        # Remove keys that are computed/derived (not constructor parameters)
        # n_time_steps is computed from T and dt, not a constructor parameter
        config_dict = migrate_legacy_config_dict(config_dict)

        interaction_backend = config_data.get("interaction_backend") or config_dict.get(
            "interaction_backend"
        )
        if interaction_backend not in {"exact", "nn", "random_batch"}:
            if config_dict.get("learn_interaction_drift") or config_dict.get(
                "learn_interaction_cost"
            ) or config_data.get("method") == "learned":
                interaction_backend = "nn"
            elif config_dict.get("random_batch_size") is not None:
                interaction_backend = "random_batch"
            else:
                interaction_backend = "exact"
        config_dict["interaction_backend"] = interaction_backend
        config = Config(**config_dict)

        # Create manager instance
        manager = cls.__new__(cls)
        manager.config = config
        manager.exp_dir = exp_dir
        manager.scenario_dir = exp_dir.parent
        manager.full_scenario_name = config_data.get("scenario_name", "unknown")
        manager.interaction_backend = interaction_backend
        manager.method = interaction_backend
        manager.method_group = config_data.get(
            "method_group",
            "learned" if interaction_backend == "nn" else "analytical",
        )
        manager.exp_id = config_data.get("experiment_id", exp_dir.name)
        manager.experiment_root = exp_dir.parent.parent
        manager.auto_timestamp = False

        # Update config paths
        manager._update_config_paths()

        print(f"\nLoaded Existing Experiment:")
        print(f"  Scenario: {manager.full_scenario_name}")
        print(f"  Exp ID:   {manager.exp_id}")
        print(f"  Path:     {manager.exp_dir}")

        return manager

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint in experiment directory.

        Priority:
        1. checkpoint_final.pt (if exists)
        2. Most recent checkpoint_XXXX.pt by iteration number
        3. None if no checkpoints found

        Returns:
            Path to checkpoint file, or None if not found
        """
        checkpoints_dir = self.exp_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None

        # Check for final checkpoint first
        final_ckpt = checkpoints_dir / "checkpoint_final.pt"
        if final_ckpt.exists():
            return final_ckpt

        # Find numbered checkpoints
        checkpoints = list(checkpoints_dir.glob("checkpoint_[0-9][0-9][0-9][0-9].pt"))
        if not checkpoints:
            return None

        # Sort by iteration number (extracted from filename)
        def get_iteration(path: Path) -> int:
            # Extract iteration from "checkpoint_0010.pt"
            return int(path.stem.split("_")[1])

        latest = max(checkpoints, key=get_iteration)
        return latest

    def __repr__(self) -> str:
        return (
            f"ExperimentManager(scenario={self.full_scenario_name}, "
            f"exp_id={self.exp_id}, path={self.exp_dir})"
        )
