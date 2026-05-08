"""Shared helpers for the per-phase runner scripts.

Each runner submits a batch of training configs, polls until they all finish,
and returns the experiment objects so the runner can persist whatever it needs
to disk for the next phase.
"""

import json
import os
import time
from pathlib import Path

from cs336_scaling.client import (
    get_budget,
    get_experiment,
    list_experiments,
    submit_experiment,
)
from cs336_scaling.training.training_config import TrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def submit_or_get_existing(config_dict):
    """Submit a config; if it already exists (HTTP 409), find and return its id.

    The API rejects duplicate submissions by config-hash. Re-running a phase
    script after a partial run should pick up the existing experiments rather
    than fail.
    """
    target_config = TrainingConfig.model_validate(config_dict)
    try:
        return submit_experiment(target_config).experiment_id
    except RuntimeError as exc:
        if "409" not in str(exc):
            raise
    target_id = target_config.unique_id
    for exp in list_experiments():
        if exp.training_config.unique_id == target_id:
            return exp.experiment_id
    raise RuntimeError(f"409 returned but no experiment matches unique_id={target_id}")


def submit_batch(configs, label=""):
    """Submit each config and return a list of experiment ids in order."""
    ids = []
    for i, cfg in enumerate(configs):
        exp_id = submit_or_get_existing(cfg)
        ids.append(exp_id)
        print(f"  [{label}] {i+1}/{len(configs)}  experiment_id={exp_id}")
    return ids


def poll_until_done(experiment_ids, poll_interval_seconds=15):
    """Block until every id is in a terminal state. Returns {id: ExperimentResponse}."""
    target = set(experiment_ids)
    completed = {}
    while True:
        # One bulk list call per cycle is cheaper than N gets.
        for exp in list_experiments():
            if exp.experiment_id in target and exp.status.status_type in ("completed", "failed"):
                completed[exp.experiment_id] = exp
        remaining = target - set(completed)
        if not remaining:
            break
        budget = get_budget()
        print(
            f"  waiting on {len(remaining)}/{len(target)}  "
            f"remaining_budget={budget.remaining_seconds:.0f}s"
        )
        time.sleep(poll_interval_seconds)
    # Re-order to match input order.
    return [completed[i] for i in experiment_ids]


def final_loss(exp):
    """Return the last reported val loss (full or partial), or None if there is none."""
    status = exp.status
    if status.status_type == "completed" and status.val_losses:
        return status.val_losses[-1]
    if status.status_type == "failed":
        reason = status.reason
        partial = getattr(reason, "partial_val_losses", None) or []
        if partial:
            return partial[-1]
    return None


def dump_results(filename, payload):
    """Write a JSON payload to data/<filename>. Pretty-printed for diff-friendliness."""
    path = DATA_DIR / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  wrote {path}")


def load_results(filename):
    path = DATA_DIR / filename
    with open(path) as f:
        return json.load(f)


def check_api_key():
    if not os.environ.get("A3_API_KEY"):
        raise RuntimeError("A3_API_KEY environment variable is not set")


def serialize_experiment(exp):
    """Pydantic → JSON-serialisable dict. Used for persisting raw results."""
    return exp.model_dump(mode="json")
