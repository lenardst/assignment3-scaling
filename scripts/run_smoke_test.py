"""Smoke test runner.

Submits one cheap proxy-model run, waits for it to finish, prints the final
val loss, used wall-clock seconds, and the implied MFU. Always run this
before any other phase — if MFU is much lower than 0.4, the Phase 3 max_runtime
caps need to be widened (or compute targets reduced) before submission.
"""

from cs336_scaling.client import get_budget

from scripts.experiments import SMOKE_TEST_CONFIG, non_embedding_params
from scripts.leaderboard import B200_BF16_PEAK_FLOPS_PER_SECOND
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
    [exp_id] = submit_batch([SMOKE_TEST_CONFIG], label="smoke")
    [exp] = poll_until_done([exp_id])

    arch = exp.training_config.architecture_config
    N = non_embedding_params(arch.hidden_size, arch.num_hidden_layers)
    D = exp.training_config.total_train_tokens
    actual_flops = 6 * N * D

    status_type = exp.status.status_type
    print(f"\nstatus={status_type}")
    if status_type == "completed":
        used = exp.status.used_runtime_seconds
        measured_mfu = actual_flops / (used * B200_BF16_PEAK_FLOPS_PER_SECOND)
        print(f"final_val_loss = {final_loss(exp):.4f}")
        print(f"used_runtime   = {used:.1f}s")
        print(f"measured_mfu   = {measured_mfu:.3f}")
        print(f"all_val_losses = {exp.status.val_losses}")
    else:
        print(f"FAILED — investigate before submitting Phase 1: {exp.status}")

    dump_results("smoke_test.json", {
        "experiment_id": exp_id,
        "experiment": serialize_experiment(exp),
        "final_loss": final_loss(exp),
    })
    print(f"\nBudget after: {get_budget()}")


if __name__ == "__main__":
    main()
