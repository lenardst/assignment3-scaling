"""Fit the Chinchilla scaling law to Phase 3 results and predict the optimal
configuration for the 48 B200-hour leaderboard run.

Reads data/phase3.json, fits L(N, D) = E + A/N^α + B/D^β, and prints the
predicted (N_opt, D_opt, L_opt) for the leaderboard compute budget. Also
emits a complete TrainingConfig that can be passed to save_final_submission.

Does NOT call save_final_submission automatically — review the prediction
first, then submit manually.
"""

import json
from pathlib import Path

from scripts.experiments import (
    HEAD_DIM,
    TARGET_FLOPS,
    intermediate_size_for,
    make_training_config,
    non_embedding_params,
    round_tokens,
)
from scripts.fit_scaling_law import fit_chinchilla_loss, predict_optimal
from scripts.leaderboard import scale_learning_rate
from scripts.runner import DATA_DIR, dump_results, load_results


def find_nearest_valid_architecture(target_N):
    """Pick (hidden_size, num_hidden_layers) that approximate target_N and
    satisfy the API constraints (hidden_size divisible by HEAD_DIM, etc.).

    Uses a depth/width relation close to the example config (d_model ≈ 50 × layers),
    then rounds hidden_size down to a multiple of HEAD_DIM. Slight over- or
    under-shoot of N is acceptable.
    """
    best = None
    for layers in range(2, 32):
        # 12 * layers * d² = N → d = sqrt(N / (12*layers))
        d_raw = (target_N / (12 * layers)) ** 0.5
        d = max(HEAD_DIM, round(d_raw / HEAD_DIM) * HEAD_DIM)
        N = non_embedding_params(d, layers)
        # Heuristic preference: keep d/layers near 50 (close to example config 448/9 ≈ 50).
        aspect_penalty = abs(d / layers - 50) / 50
        size_penalty = abs(N / target_N - 1)
        score = size_penalty + 0.3 * aspect_penalty
        if best is None or score < best[0]:
            best = (score, d, layers, N)
    _, d, layers, N = best
    return d, layers, N


def main():
    phase3 = load_results("phase3.json")
    proxy_lr = phase3["proxy_optimal_lr"]
    gamma = phase3["lr_exponent"]
    rows = [r for r in phase3["rows"] if r["final_loss"] is not None]
    print(f"Fitting on {len(rows)} valid Phase 3 runs")

    fit_input = [{"N": r["N"], "D": r["D"], "final_loss": r["final_loss"]} for r in rows]
    E, A, B, alpha, beta = fit_chinchilla_loss(fit_input)
    print(f"\nFit: E={E:.4f}  A={A:.3e}  B={B:.3e}  α={alpha:.3f}  β={beta:.3f}")

    N_opt, D_opt, L_opt = predict_optimal(E, A, B, alpha, beta, TARGET_FLOPS)
    print(f"\nPredicted optimum at C={TARGET_FLOPS:.2e} FLOPs (48 B200-hours):")
    print(f"  N_opt = {N_opt:.3e} parameters")
    print(f"  D_opt = {D_opt:.3e} tokens")
    print(f"  L_opt = {L_opt:.4f}")

    largest_trained_N = max(r["N"] for r in rows)
    extrapolation = N_opt / largest_trained_N
    print(f"\nExtrapolation factor in N: {extrapolation:.1f}× beyond largest training point")
    if extrapolation > 5:
        print("  WARNING: prediction is more than 5× outside the fit range — treat with caution")

    hidden_size, num_hidden_layers, N_actual = find_nearest_valid_architecture(N_opt)
    print(f"\nNearest valid architecture for N≈N_opt:")
    print(f"  hidden_size={hidden_size}  num_hidden_layers={num_hidden_layers}")
    print(f"  N_actual={N_actual:.3e} (target was {N_opt:.3e})")

    # D rounded to the eval-block grid.
    D_actual = round_tokens(TARGET_FLOPS / (6 * N_actual))
    print(f"  total_train_tokens={D_actual:,}")

    peak_lr = scale_learning_rate(proxy_lr, hidden_size, gamma)
    # Leaderboard run is 48 B200-hours = 172,800 seconds.
    leaderboard_max_runtime = 48 * 3600

    final_config = make_training_config(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        peak_lr=peak_lr,
        total_train_tokens=D_actual,
        max_runtime_seconds=leaderboard_max_runtime,
    )

    print(f"\nProposed final config (peak_lr={peak_lr:.3e}, intermediate_size={intermediate_size_for(hidden_size)}):")
    print(json.dumps(final_config, indent=2))

    dump_results("predicted_optimal.json", {
        "fit": {"E": E, "A": A, "B": B, "alpha": alpha, "beta": beta},
        "predicted": {"N_opt": N_opt, "D_opt": D_opt, "L_opt": L_opt},
        "selected_architecture": {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "N_actual": N_actual,
            "D_actual": D_actual,
            "peak_lr": peak_lr,
            "extrapolation_factor": extrapolation,
        },
        "final_config": final_config,
    })

    print("\nReview the prediction. If it looks right, submit the final config:")
    print("  from cs336_scaling.client import save_final_submission")
    print(f"  save_final_submission(final_config, predicted_final_loss={L_opt:.4f})")


if __name__ == "__main__":
    main()
