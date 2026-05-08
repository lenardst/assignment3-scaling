import numpy as np

from scripts.leaderboard import B200_BF16_PEAK_FLOPS_PER_SECOND, scale_learning_rate

# Fixed by the API training harness — these values cannot be changed.
HEAD_DIM = 64
SEQ_LEN = 512
TRAIN_BATCH_SIZE = 128
N_EVALS = 16
TOKENS_PER_STEP = SEQ_LEN * TRAIN_BATCH_SIZE           # 65_536
TOKENS_PER_EVAL_BLOCK = TOKENS_PER_STEP * N_EVALS      # 1_048_576  (total_train_tokens must be a multiple)
# Four eval blocks ≈ 4 M tokens: minimum to get a stable final validation loss estimate.
D_MIN_TOKENS = 4 * TOKENS_PER_EVAL_BLOCK

# Smallest model where num_heads = hidden_size / HEAD_DIM = 2 is a valid integer.
# Cheap enough to sweep 9 LR values in under 30 minutes total.
PROXY_HIDDEN_SIZE = 128

# 48 B200-hours converted to FLOPs using peak BF16 throughput.
TARGET_FLOPS = 48 * 3600 * B200_BF16_PEAK_FLOPS_PER_SECOND


def non_embedding_params(hidden_size, num_hidden_layers):
    return 12 * num_hidden_layers * hidden_size ** 2


def intermediate_size_for(hidden_size):
    # Preserves the example config's SwiGLU expansion ratio (1280/448 ≈ 8/3).
    # Rounded to the nearest 128 so tensor shapes stay GPU-tile-aligned.
    return round(hidden_size * 1280 / 448 / 128) * 128


def round_tokens(n_tokens):
    # total_train_tokens must be an exact multiple of TOKENS_PER_EVAL_BLOCK.
    # Floor at 4 blocks so every run has at least D_MIN_TOKENS of training data.
    n_blocks = max(4, round(n_tokens / TOKENS_PER_EVAL_BLOCK))
    return n_blocks * TOKENS_PER_EVAL_BLOCK


def max_runtime_for(actual_flops, mfu=0.4, safety_factor=1.5):
    # mfu=0.4: conservative estimate for small models on B200s (large models reach ~0.5).
    # safety_factor=1.5: small but real margin. The API reserves max_runtime_seconds
    # against the budget at submit time (refunded on completion), so safety_factor
    # cannot be too generous — at 1.5×, MFU must be ≥ ~0.27 for runs to finish
    # within the cap. The smoke test measures actual MFU before we commit Phase 3.
    # Hard cap at 2 hours prevents any single run from consuming the entire budget.
    return int(min(actual_flops / (mfu * B200_BF16_PEAK_FLOPS_PER_SECOND) * safety_factor, 2 * 3600))


def make_training_config(hidden_size, num_hidden_layers, peak_lr,
                         total_train_tokens, max_runtime_seconds):
    num_heads = hidden_size // HEAD_DIM
    return {
        "architecture_config": {
            "attention_bias": False,
            "head_dim": HEAD_DIM,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size_for(hidden_size),
            "num_attention_heads": num_heads,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_heads,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000,
            "tie_word_embeddings": False,
            "dtype": "bfloat16",
            "vocab_size": 32_000,
        },
        "optimizer_config": {
            "lr_scheduler": {
                "peak_value": float(peak_lr),
                # Cosine decay to 10 % of peak — keeps the final LR non-zero to avoid
                # over-fitting the end of training (standard for LM pre-training).
                "final_lr_frac": 0.1,
                # 5 % linear warmup stabilises early training when parameters are random.
                "warmup_frac": 0.05,
                "init_value": 0.0,
            },
            # Mild L2 regularisation; stronger decay hurts small models with few parameters.
            "weight_decay": 1e-2,
            # Standard AdamW momentum coefficients for transformer training.
            "beta1": 0.9,
            # beta2=0.95 (vs default 0.999) tracks curvature faster on the short runs
            # in Phases 1–2, and matches what Chinchilla and PaLM used.
            "beta2": 0.95,
            "eps": 1e-8,
            "eps_root": 1e-8,
            # Gradient clipping at 1.0 is de-facto standard for transformer training.
            "grad_clip_norm": 1.0,
        },
        "train_batch_size": TRAIN_BATCH_SIZE,
        "val_batch_size": 32,
        "n_evals": N_EVALS,
        "total_train_tokens": total_train_tokens,
        "max_runtime_seconds": float(max_runtime_seconds),
        "model_seed": 0,
    }


# ─── Smoke test ───────────────────────────────────────────────────────────────
# Single cheap run submitted before any other phase.
# Verifies: config schema is accepted, result dict matches extract_results_from_api,
# and the harness actually trains. MFU can be estimated from wall-clock vs FLOPs.
# If MFU < 0.3, the max_runtime caps in Phase 3 become dangerously tight.

SMOKE_TEST_CONFIG = make_training_config(
    hidden_size=PROXY_HIDDEN_SIZE,
    num_hidden_layers=2,
    # 3.5e-3 is intentionally off the Phase 1 grid so the smoke test does not
    # collide with a Phase 1 config (the API rejects duplicate submissions with 409).
    peak_lr=3.5e-3,
    total_train_tokens=round_tokens(2e6),   # resolves to D_MIN_TOKENS (4 eval blocks)
    max_runtime_seconds=120,
)


# ─── Phase 1: LR sweep on proxy model ─────────────────────────────────────────
# Tune LR on the smallest viable model. Under the SP approximation to muP,
# the optimal LR here will be scaled down by (d_proxy/d_target)^γ
# when applied to larger models in Phases 2–3.
#
# LR range 3e-4 to 3e-1 (9 log-spaced points):
#   Lower end (3e-4) trains slowly but converges — establishes a floor.
#   Upper end (3e-1) is well above stable LRs for a 2-layer model and should
#   diverge visibly, guaranteeing the true optimum is interior to the sweep
#   rather than sitting at an edge.
#   9 points give ~0.5 log-decade resolution between adjacent values.

PHASE1_CONFIGS = [
    make_training_config(
        hidden_size=PROXY_HIDDEN_SIZE,
        num_hidden_layers=2,
        peak_lr=float(lr),
        total_train_tokens=round_tokens(16e6),
        max_runtime_seconds=120,
    )
    for lr in np.geomspace(3e-4, 3e-1, 9)
]


# ─── Phase 2: LR-transfer exponent calibration ────────────────────────────────
# The SP approximation scales LR as proxy_lr × (d_proxy / d_target)^γ.
# μP theory predicts γ=1.0, but SP does not guarantee this — the exponent
# must be measured empirically.
#
# We run two verification model sizes (320 and 512) to get two independent
# calibration points for γ. For each size, we test the predicted LR plus one
# step up and one step down (×0.5, ×1.0, ×2.0).
#
# After Phase 2, fit γ using fit_lr_exponent():
#   γ = log(LR_opt(s) / proxy_optimal_lr) / log(PROXY_HIDDEN_SIZE / s)
# for s ∈ {320, 512}. If the two estimates agree, γ is reliable.
# If they disagree, use the average and flag the uncertainty.
#
# d_model=320, 6 layers: N ≈ 7.4M (4× larger width than proxy).
# d_model=512, 8 layers: N ≈ 25.2M (4× larger again); second calibration anchor.

PHASE2_VERIFICATION_SIZES = [
    (320, 6),
    (512, 8),
]


def make_phase2_configs(proxy_optimal_lr, lr_exponent=1.0):
    configs = []
    for hidden_size, num_hidden_layers in PHASE2_VERIFICATION_SIZES:
        scaled_lr = scale_learning_rate(proxy_optimal_lr, hidden_size, lr_exponent)
        for lr in [scaled_lr / 2, scaled_lr, scaled_lr * 2]:
            configs.append(make_training_config(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                peak_lr=float(lr),
                total_train_tokens=round_tokens(16e6),
                max_runtime_seconds=600,
            ))
    return configs


def fit_lr_exponent(proxy_optimal_lr, phase2_optimal_lrs):
    """Fit γ from Phase 2 calibration points.

    Args:
        proxy_optimal_lr: best LR found in Phase 1 at d_model=128.
        phase2_optimal_lrs: dict of {hidden_size: best_lr} from Phase 2.

    Returns:
        Mean γ across all calibration sizes.
    """
    exponents = [
        np.log(best_lr / proxy_optimal_lr) / np.log(PROXY_HIDDEN_SIZE / hidden_size)
        for hidden_size, best_lr in phase2_optimal_lrs.items()
    ]
    return float(np.mean(exponents))


# ─── Phase 3: Scaling law sweep ───────────────────────────────────────────────
# Six model families spanning 64× in non-embedding parameters (1.8M–116.4M).
# The new (832, 14) family replaces the two small-compute (192, 4) runs and
# extends the upper end of the fit toward the leaderboard target (~300M–1B),
# reducing the extrapolation gap from ~15× to ~4×.
# The (192, 4) family is retained at one compute level to anchor the fit
# in the small-N regime where A/N^α dominates.
#
# IsoFLOPs coverage (families at each compute level):
#   C = 1e18:  (192), (320), (448), (576), (832)         — 5 families
#   C = 3e18:  (320), (448), (576), (704), (832)          — 5 families  ← best
#   C = 1e19:  (448), (576), (704)                        — 3 families
#
# At the two richest compute levels, all five N-values from 7.4M to 116.4M
# are represented, enabling IsoFLOPs parabola fits as a cross-check on the
# parametric Chinchilla fit.

SCALING_FAMILIES = [
    (192,  4),   # N ≈   1.8M  (one level; anchors A/N^α in the small-N regime)
    (320,  6),   # N ≈   7.4M
    (448,  9),   # N ≈  21.7M  (example config architecture)
    (576, 10),   # N ≈  39.8M
    (704, 12),   # N ≈  71.5M
    (832, 14),   # N ≈ 116.4M  (new; num_heads = 13; reduces extrapolation gap)
]

COMPUTE_LEVELS_PER_FAMILY = {
    # One level only: this small model needs no D-sweep since A/N^α dominates
    # at any reasonable D. C=1e18 gives D/N≈53 (overtrained), which is exactly
    # where the N-scaling signal is cleanest.
    (192,  4): [1e18],
    # Two levels from 1e18 to 3e18: cheaper family; both within the 2-hour cap.
    (320,  6): [1e18, 3e18],
    # Three levels: spans 10× in compute; this family also serves as the
    # "example config" checkpoint.
    (448,  9): [1e18, 3e18, 1e19],
    (576, 10): [1e18, 3e18, 1e19],
    # Two levels: C=1e19 is the practical ceiling before the 2-hour cap bites
    # at MFU < 0.35.
    (704, 12): [3e18, 1e19],
    # Two levels: upper end at 3e18 keeps well under the 2-hour cap even at
    # MFU=0.25, avoiding the risk of a truncated cosine schedule.
    (832, 14): [1e18, 3e18],
}


def make_phase3_configs(proxy_optimal_lr, lr_exponent=1.0):
    configs = []
    for hidden_size, num_hidden_layers in SCALING_FAMILIES:
        N = non_embedding_params(hidden_size, num_hidden_layers)
        peak_lr = scale_learning_rate(proxy_optimal_lr, hidden_size, lr_exponent)
        for target_flops in COMPUTE_LEVELS_PER_FAMILY[(hidden_size, num_hidden_layers)]:
            total_train_tokens = round_tokens(target_flops / (6 * N))
            actual_flops = 6 * N * total_train_tokens
            configs.append(make_training_config(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                peak_lr=float(peak_lr),
                total_train_tokens=total_train_tokens,
                max_runtime_seconds=max_runtime_for(actual_flops),
            ))
    return configs


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)
    lr = SMOKE_TEST_CONFIG["optimizer_config"]["lr_scheduler"]["peak_value"]
    tokens = SMOKE_TEST_CONFIG["total_train_tokens"]
    print(f"  peak_lr={lr:.2e}  tokens={tokens:,}  "
          f"max_runtime={SMOKE_TEST_CONFIG['max_runtime_seconds']:.0f}s")

    print()
    print("=" * 60)
    print("PHASE 1 — LR sweep on proxy model (d_model=128, 2 layers)")
    print("=" * 60)
    for cfg in PHASE1_CONFIGS:
        lr = cfg["optimizer_config"]["lr_scheduler"]["peak_value"]
        tokens = cfg["total_train_tokens"]
        print(f"  peak_lr={lr:.2e}  tokens={tokens:,}  max_runtime={cfg['max_runtime_seconds']:.0f}s")

    print()
    print("=" * 60)
    print("PHASE 2 — LR exponent calibration (d_model=320 and 512)")
    print("(shown with placeholder proxy_lr=5e-3, exponent=1.0)")
    print("=" * 60)
    for cfg in make_phase2_configs(proxy_optimal_lr=5e-3):
        arch = cfg["architecture_config"]
        lr = cfg["optimizer_config"]["lr_scheduler"]["peak_value"]
        print(f"  d_model={arch['hidden_size']}  peak_lr={lr:.2e}  "
              f"tokens={cfg['total_train_tokens']:,}  max_runtime={cfg['max_runtime_seconds']:.0f}s")

    print()
    print("  Example: fit_lr_exponent(5e-3, {320: 1.5e-3, 512: 8e-4})")
    gamma_example = fit_lr_exponent(5e-3, {320: 1.5e-3, 512: 8e-4})
    print(f"  → γ = {gamma_example:.3f}")

    print()
    print("=" * 60)
    print("PHASE 3 — Scaling law sweep")
    print("(shown with placeholder proxy_lr=5e-3, exponent=1.0)")
    print("=" * 60)
    for (hidden_size, num_hidden_layers), cfg in zip(
        [(h, l) for h, l in SCALING_FAMILIES
         for _ in COMPUTE_LEVELS_PER_FAMILY[(h, l)]],
        make_phase3_configs(proxy_optimal_lr=5e-3),
    ):
        N = non_embedding_params(hidden_size, num_hidden_layers)
        D = cfg["total_train_tokens"]
        lr = cfg["optimizer_config"]["lr_scheduler"]["peak_value"]
        runtime = cfg["max_runtime_seconds"]
        actual_flops = 6 * N * D
        print(f"  d_model={hidden_size:3d}  layers={num_hidden_layers:2d}  "
              f"N={N:.2e}  D={D:.2e}  C={actual_flops:.1e}  "
              f"lr={lr:.2e}  max_runtime={runtime:.0f}s")

    def expected_seconds(cfg):
        arch = cfg["architecture_config"]
        N = non_embedding_params(arch["hidden_size"], arch["num_hidden_layers"])
        D = cfg["total_train_tokens"]
        return 6 * N * D / (0.4 * B200_BF16_PEAK_FLOPS_PER_SECOND)

    phase1_seconds = sum(expected_seconds(c) for c in PHASE1_CONFIGS)
    phase2_seconds = sum(expected_seconds(c) for c in make_phase2_configs(5e-3))
    phase3_seconds = sum(expected_seconds(c) for c in make_phase3_configs(5e-3))
    n_phase3 = len(make_phase3_configs(5e-3))
    print()
    print(f"Phase 1 expected:  {phase1_seconds/3600:.2f} B200-hours  ({len(PHASE1_CONFIGS)} runs)")
    print(f"Phase 2 expected:  {phase2_seconds/3600:.2f} B200-hours  ({len(PHASE2_VERIFICATION_SIZES) * 3} runs)")
    print(f"Phase 3 expected:  {phase3_seconds/3600:.2f} B200-hours  ({n_phase3} runs)")
    total = phase1_seconds + phase2_seconds + phase3_seconds
    print(f"Total expected:    {total/3600:.2f} B200-hours  (of 12 available)")
    print(f"Remaining buffer:  {12 - total/3600:.2f} B200-hours")
    print()
    print(f"At MFU=0.3 (pessimistic):  {total/3600 * (0.4/0.3):.2f} B200-hours  "
          f"(buffer: {12 - total/3600 * (0.4/0.3):.2f})")
