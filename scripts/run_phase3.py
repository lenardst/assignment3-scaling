"""Phase 3: Scaling-law sweep.

Reads proxy_optimal_lr and lr_exponent from earlier phases, generates the 13
Phase 3 configs, submits in waves so the budget reservation never exceeds the
total budget, polls, and saves all (N, D, final_loss) triples to data/phase3.json.

Submission is wave-based to respect the budget-reservation invariant: at any
moment, the API holds back the SUM of max_runtime_seconds across queued+running
experiments. Submitting all 13 at once would reserve ~10.6 hours, which can
collide with whatever Phase 1+2 reserved if any of those have not been refunded
yet. Waves let earlier completions refund before later submissions.
"""

import time

from cs336_scaling.client import get_budget

from scripts.experiments import (
    SCALING_FAMILIES,
    COMPUTE_LEVELS_PER_FAMILY,
    make_phase3_configs,
    non_embedding_params,
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

# Submit Phase 3 in waves to keep concurrent reservation under control.
# Each wave's reservation should fit comfortably in remaining budget.
WAVE_SIZE = 5


def chunk(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    check_api_key()
    phase2 = load_results("phase2.json")
    proxy_lr = phase2["proxy_optimal_lr"]
    gamma = phase2["lr_exponent"]
    print(f"Loaded proxy_optimal_lr={proxy_lr:.4e}  γ={gamma:.3f}")

    configs = make_phase3_configs(proxy_optimal_lr=proxy_lr, lr_exponent=gamma)
    family_keys = [
        (h, l)
        for h, l in SCALING_FAMILIES
        for _ in COMPUTE_LEVELS_PER_FAMILY[(h, l)]
    ]
    assert len(configs) == len(family_keys)
    print(f"Total Phase 3 runs: {len(configs)}")
    print(f"Budget before: {get_budget()}")

    all_ids = []
    all_experiments = []
    for wave_idx, wave in enumerate(chunk(list(zip(configs, family_keys)), WAVE_SIZE)):
        wave_configs = [c for c, _ in wave]
        print(f"\n=== Wave {wave_idx + 1} ({len(wave_configs)} runs) ===")
        ids = submit_batch(wave_configs, label=f"phase3-w{wave_idx + 1}")
        experiments = poll_until_done(ids)
        all_ids.extend(ids)
        all_experiments.extend(experiments)
        time.sleep(2)  # let refunds settle in the budget table

    print("\nPhase 3 results:")
    rows = []
    for cfg, fkey, exp in zip(configs, family_keys, all_experiments):
        hidden_size, num_hidden_layers = fkey
        N = non_embedding_params(hidden_size, num_hidden_layers)
        D = cfg["total_train_tokens"]
        loss = final_loss(exp)
        loss_str = f"{loss:.4f}" if loss is not None else "    n/a"
        actual_flops = 6 * N * D
        rows.append({
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "N": N,
            "D": D,
            "actual_flops": actual_flops,
            "peak_lr": cfg["optimizer_config"]["lr_scheduler"]["peak_value"],
            "final_loss": loss,
            "status": exp.status.status_type,
        })
        print(f"  d_model={hidden_size:3d} layers={num_hidden_layers:2d} "
              f"N={N:.2e} D={D:.2e} C={actual_flops:.1e} "
              f"final_loss={loss_str} status={exp.status.status_type}")

    valid = [r for r in rows if r["final_loss"] is not None]
    print(f"\n{len(valid)}/{len(rows)} runs produced a valid final loss")
    if len(valid) < 8:
        print("WARNING: fewer than 8 valid points — the Chinchilla fit will be unreliable")

    dump_results("phase3.json", {
        "proxy_optimal_lr": proxy_lr,
        "lr_exponent": gamma,
        "rows": rows,
        "experiments": [serialize_experiment(e) for e in all_experiments],
    })
    print(f"\nBudget after: {get_budget()}")


if __name__ == "__main__":
    main()
