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


def max_runtime_for(actual_flops, mfu=0.4, safety_factor=3.0):
    # mfu=0.4: conservative estimate for small models on B200s (large models reach ~0.5).
    # safety_factor=3.0: buffers against slow cluster startup and variable queue load.
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


# ─── Phase 1: LR sweep on proxy model ────────────────────────────────────────
# Tune LR on the smallest viable model. Under the SP approximation to muP,
# the optimal LR here will be scaled down by d_model_proxy/d_model_target
# when applied to larger models in Phase 3.
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


# ─── Phase 2: LR verification on intermediate model ──────────────────────────
# After Phase 1, take the best LR, scale it by (128/320)^exponent, and test
# that prediction plus one step up and one step down. If the middle run wins,
# exponent=1.0 holds. Otherwise fit exponent from the two calibration points.
#
# d_model=320, 6 layers: ~57× more non-embedding params than the proxy model,
# large enough that a wrong exponent creates a measurable gap; small enough
# to run in under 10 minutes each.

def make_phase2_configs(proxy_optimal_lr, lr_exponent=1.0):
    scaled_lr = scale_learning_rate(proxy_optimal_lr, 320, lr_exponent)
    return [
        make_training_config(
            hidden_size=320,
            num_hidden_layers=6,
            peak_lr=float(lr),
            total_train_tokens=round_tokens(16e6),
            max_runtime_seconds=600,
        )
        for lr in [scaled_lr / 2, scaled_lr, scaled_lr * 2]
    ]


# ─── Phase 3: Scaling law sweep ──────────────────────────────────────────────
# Five model families spanning ~40× in non-embedding parameters (1.8M–71.5M).
# The Chinchilla parametric fit has 5 free parameters (E, A, B, α, β), so 5
# distinct N values is the bare minimum; using 14 runs total gives redundancy.
#
# Depth/width chosen to roughly follow Kaplan et al.'s near-optimal shape
# (layers ≈ hidden_size / 48) while keeping num_heads = hidden_size / 64
# as a valid integer.
#
# The example config (448, 9) is included as family 3 so leaderboard runs can
# reuse those results.
#
# Compute levels per family:
#   - Three levels spanning ~10–30× so the loss curve's shape is well-sampled.
#   - Absolute levels chosen so D = C / (6N) >= D_MIN_TOKENS after rounding.
#   - The largest family (704, 12) gets only two levels because C=3e19 would
#     exceed the 2-hour per-run cap and consume too much of the total budget.

SCALING_FAMILIES = [
    (192,  4),   # N ≈  1.8M
    (320,  6),   # N ≈  7.4M
    (448,  9),   # N ≈ 21.7M  (example config architecture)
    (576, 10),   # N ≈ 39.8M
    (704, 12),   # N ≈ 71.5M
]

COMPUTE_LEVELS_PER_FAMILY = {
    (192,  4): [3e17, 1e18, 3e18],
    (320,  6): [3e17, 1e18, 3e18],
    (448,  9): [1e18, 3e18, 1e19],
    (576, 10): [1e18, 3e18, 1e19],
    (704, 12): [3e18, 1e19],
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
    print("PHASE 1 — LR sweep on proxy model (d_model=128, 2 layers)")
    print("=" * 60)
    for cfg in PHASE1_CONFIGS:
        lr = cfg["optimizer_config"]["lr_scheduler"]["peak_value"]
        tokens = cfg["total_train_tokens"]
        print(f"  peak_lr={lr:.2e}  tokens={tokens:,}  max_runtime={cfg['max_runtime_seconds']:.0f}s")

    print()
    print("=" * 60)
    print("PHASE 2 — LR verification (d_model=320, 6 layers)")
    print("(shown with placeholder proxy_lr=5e-3)")
    print("=" * 60)
    for cfg in make_phase2_configs(proxy_optimal_lr=5e-3):
        lr = cfg["optimizer_config"]["lr_scheduler"]["peak_value"]
        print(f"  peak_lr={lr:.2e}  tokens={cfg['total_train_tokens']:,}  max_runtime={cfg['max_runtime_seconds']:.0f}s")

    print()
    print("=" * 60)
    print("PHASE 3 — Scaling law sweep")
    print("(shown with placeholder proxy_lr=5e-3, exponent=1.0)")
    print("=" * 60)
    total_seconds = 0
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
        total_seconds += runtime
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
    print()
    print(f"Phase 1 expected:  {phase1_seconds/3600:.2f} B200-hours  ({len(PHASE1_CONFIGS)} runs)")
    print(f"Phase 2 expected:  {phase2_seconds/3600:.2f} B200-hours  (3 runs)")
    print(f"Phase 3 expected:  {phase3_seconds/3600:.2f} B200-hours  ({len(make_phase3_configs(5e-3))} runs)")
    total = phase1_seconds + phase2_seconds + phase3_seconds
    print(f"Total expected:    {total/3600:.2f} B200-hours  (of 12 available)")
    print(f"Remaining buffer:  {12 - total/3600:.2f} B200-hours")
