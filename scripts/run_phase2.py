"""Phase 2: LR-transfer exponent calibration.

Reads the proxy_optimal_lr from data/phase1.json, generates 6 verification
configs (3 LRs × 2 sizes), submits, polls, and fits γ from the two best-LR
calibration points. Saves data/phase2.json for Phase 3.
"""

from cs336_scaling.client import get_budget

from scripts.experiments import (
    PHASE2_VERIFICATION_SIZES,
    fit_lr_exponent,
    make_phase2_configs,
)
from scripts.runner import (
    check_api_key,
    dump_results,
    final_loss,
    load_results,
    poll_until_done,
    serialize_experiment,
    submit_batch,
)


def main():
    check_api_key()
    phase1 = load_results("phase1.json")
    proxy_lr = phase1["proxy_optimal_lr"]
    print(f"Loaded proxy_optimal_lr = {proxy_lr:.4e}")

    configs = make_phase2_configs(proxy_optimal_lr=proxy_lr, lr_exponent=1.0)
    print(f"Submitting {len(configs)} Phase 2 configs across "
          f"{len(PHASE2_VERIFICATION_SIZES)} verification sizes")
    print(f"Budget before: {get_budget()}")

    ids = submit_batch(configs, label="phase2")
    experiments = poll_until_done(ids)

    # Group results by hidden_size, find the best LR within each group.
    by_size = {}
    for cfg, exp in zip(configs, experiments):
        h = cfg["architecture_config"]["hidden_size"]
        lr = cfg["optimizer_config"]["lr_scheduler"]["peak_value"]
        loss = final_loss(exp)
        by_size.setdefault(h, []).append((lr, loss, exp.status.status_type))

    optimal_lrs = {}
    print("\nPhase 2 results:")
    for h, rows in by_size.items():
        rows.sort(key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0.0))
        for lr, loss, status in rows:
            loss_str = f"{loss:.4f}" if loss is not None else "    n/a"
            print(f"  d_model={h}  peak_lr={lr:.2e}  final_loss={loss_str}  status={status}")
        valid = [r for r in rows if r[1] is not None]
        if not valid:
            raise RuntimeError(f"No valid Phase 2 result for d_model={h}")
        best_lr, _, _ = valid[0]
        optimal_lrs[h] = best_lr
        if best_lr in (rows[0][0], rows[-1][0]):
            print(f"  NOTE: best LR for d_model={h} is at an edge of the ×0.5/×1/×2 grid")

    gamma = fit_lr_exponent(proxy_lr, optimal_lrs)
    print(f"\nFitted γ across sizes {sorted(optimal_lrs)} = {gamma:.3f}")
    if abs(gamma - 1.0) > 0.5:
        print("  WARNING: γ is far from the SP-folklore value of 1.0; double-check Phase 1 result")

    dump_results("phase2.json", {
        "proxy_optimal_lr": proxy_lr,
        "phase2_optimal_lrs": optimal_lrs,
        "lr_exponent": gamma,
        "experiments": [serialize_experiment(e) for e in experiments],
    })
    print(f"\nBudget after: {get_budget()}")


if __name__ == "__main__":
    main()
