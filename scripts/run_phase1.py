"""Phase 1: LR sweep on the proxy model.

Submits all 9 Phase 1 configs, polls until each finishes, prints a table of
LR vs final loss, and saves the best-LR result to data/phase1.json so Phase 2
can pick it up.
"""

from cs336_scaling.client import get_budget

from scripts.experiments import PHASE1_CONFIGS
from scripts.runner import (
    check_api_key,
    dump_results,
    final_loss,
    poll_until_done,
    serialize_experiment,
    submit_batch,
)


def main():
    check_api_key()
    print(f"Budget before: {get_budget()}")
    ids = submit_batch(PHASE1_CONFIGS, label="phase1")
    experiments = poll_until_done(ids)

    print("\nLR sweep results (sorted by final val loss):")
    rows = []
    for cfg, exp in zip(PHASE1_CONFIGS, experiments):
        lr = cfg["optimizer_config"]["lr_scheduler"]["peak_value"]
        loss = final_loss(exp)
        rows.append((lr, loss, exp.status.status_type))
    rows.sort(key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0.0))
    for lr, loss, status in rows:
        loss_str = f"{loss:.4f}" if loss is not None else "    n/a"
        print(f"  peak_lr={lr:.2e}  final_loss={loss_str}  status={status}")

    valid_rows = [r for r in rows if r[1] is not None]
    if not valid_rows:
        raise RuntimeError("No Phase 1 run produced a valid loss")
    best_lr, best_loss, _ = valid_rows[0]
    print(f"\nproxy_optimal_lr = {best_lr:.4e}  (final_loss={best_loss:.4f})")

    if best_lr in (PHASE1_CONFIGS[0]["optimizer_config"]["lr_scheduler"]["peak_value"],
                   PHASE1_CONFIGS[-1]["optimizer_config"]["lr_scheduler"]["peak_value"]):
        print("WARNING: best LR is at an edge of the sweep — true optimum may be outside the range.")

    dump_results("phase1.json", {
        "proxy_optimal_lr": best_lr,
        "proxy_optimal_loss": best_loss,
        "experiments": [serialize_experiment(e) for e in experiments],
        "lr_to_loss": {f"{lr:.4e}": loss for lr, loss, _ in rows},
    })
    print(f"\nBudget after: {get_budget()}")


if __name__ == "__main__":
    main()
