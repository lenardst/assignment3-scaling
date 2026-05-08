import numpy as np
from scipy.optimize import minimize

from scripts.experiments import non_embedding_params, TARGET_FLOPS


def huber_loss(residuals, delta=1e-3):
    # delta=1e-3 follows the Chinchilla paper (Hoffmann et al. 2022, Appendix B).
    # Huber loss down-weights outliers (e.g. a diverged run) relative to MSE,
    # so a single bad experiment cannot dominate the fit.
    abs_r = np.abs(residuals)
    return np.where(abs_r < delta, 0.5 * residuals ** 2, delta * (abs_r - 0.5 * delta)).sum()


def chinchilla_loss(log_E, log_A, log_B, alpha, beta, Ns, Ds):
    return np.exp(log_E) + np.exp(log_A) / Ns ** alpha + np.exp(log_B) / Ds ** beta


def objective(params, Ns, Ds, observed_losses):
    log_E, log_A, log_B, alpha, beta = params
    predicted_losses = chinchilla_loss(log_E, log_A, log_B, alpha, beta, Ns, Ds)
    # Residuals in log-space weight relative errors equally across the dynamic
    # range of losses (2.5–4.5 nats), preventing large-loss runs from dominating.
    return huber_loss(np.log(predicted_losses) - np.log(observed_losses))


def fit_chinchilla_loss(experiment_results):
    Ns = np.array([r["N"] for r in experiment_results], dtype=float)
    Ds = np.array([r["D"] for r in experiment_results], dtype=float)
    Ls = np.array([r["final_loss"] for r in experiment_results], dtype=float)

    # 8 initializations cover uncertainty in E (irreducible entropy, typically 1–2 nats)
    # and the exponent magnitudes (Chinchilla reports α≈0.34, β≈0.28; Kaplan reports
    # higher values ~0.5).  Taking the best result guards against local minima.
    # log_A = log_B = log(1e7) is a neutral starting point in the middle of plausible
    # ranges for the per-parameter and per-token scaling coefficients.
    best_result = None
    for log_E in [np.log(1.0), np.log(2.0)]:
        for alpha in [0.3, 0.5]:
            for beta in [0.3, 0.5]:
                x0 = [log_E, np.log(1e7), np.log(1e7), alpha, beta]
                result = minimize(
                    objective, x0, args=(Ns, Ds, Ls), method="L-BFGS-B",
                    # alpha, beta bounded to (0.01, 2.0): negative exponents are physically
                    # meaningless (loss would increase with more data/params); exponents
                    # above 2.0 are implausible for language model scaling.
                    bounds=[(None, None), (None, None), (None, None), (0.01, 2.0), (0.01, 2.0)],
                )
                if best_result is None or result.fun < best_result.fun:
                    best_result = result

    log_E, log_A, log_B, alpha, beta = best_result.x
    return np.exp(log_E), np.exp(log_A), np.exp(log_B), alpha, beta


def predict_optimal(E, A, B, alpha, beta, target_flops):
    # Closed-form solution from Chinchilla (Hoffmann et al. 2022, Eq. 5).
    # Minimise L(N, D) subject to C = 6ND → Lagrange multiplier gives the ratio
    # N_opt / D_opt = (alpha * A) / (beta * B) and N_opt ∝ C^(beta/(alpha+beta)).
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N_opt = G * (target_flops / 6) ** (beta / (alpha + beta))
    D_opt = target_flops / (6 * N_opt)
    L_opt = E + A / N_opt ** alpha + B / D_opt ** beta
    return N_opt, D_opt, L_opt


def extract_results_from_api(api_experiments):
    """Convert API experiment results into (N, D, final_loss) triples.

    Accepts both Pydantic ExperimentResponse objects (from get_experiment / list_experiments)
    and plain dicts (e.g. loaded from JSON via model_dump). Skips runs that have no
    final loss, e.g. unfinished, failed without partial losses, or empty val_losses.
    """
    results = []
    for exp in api_experiments:
        # Normalise to dict form so we can use a single code path.
        if hasattr(exp, "model_dump"):
            data = exp.model_dump(mode="json")
        else:
            data = exp

        cfg = data["training_config"]
        arch = cfg["architecture_config"]
        N = non_embedding_params(arch["hidden_size"], arch["num_hidden_layers"])
        D = cfg["total_train_tokens"]

        status = data["status"]
        status_type = status.get("status_type")
        if status_type == "completed" and status.get("val_losses"):
            final_loss = status["val_losses"][-1]
        elif status_type == "failed":
            # Timeouts may still report partial val_losses; salvage the last one.
            partial = status.get("reason", {}).get("partial_val_losses", [])
            if not partial:
                continue
            final_loss = partial[-1]
        else:
            continue

        results.append({"N": N, "D": D, "final_loss": final_loss})
    return results


if __name__ == "__main__":
    fake_results = [
        {"N": 1_769_472,  "D": 28_311_552,  "final_loss": 4.1},
        {"N": 1_769_472,  "D": 94_371_840,  "final_loss": 3.7},
        {"N": 1_769_472,  "D": 282_066_944, "final_loss": 3.5},
        {"N": 7_372_800,  "D": 7_340_032,   "final_loss": 3.6},
        {"N": 7_372_800,  "D": 23_068_672,  "final_loss": 3.2},
        {"N": 7_372_800,  "D": 68_157_440,  "final_loss": 3.0},
        {"N": 21_676_032, "D": 7_340_032,   "final_loss": 3.3},
        {"N": 21_676_032, "D": 23_068_672,  "final_loss": 2.9},
        {"N": 21_676_032, "D": 76_546_048,  "final_loss": 2.7},
        {"N": 39_813_120, "D": 4_194_304,   "final_loss": 3.1},
        {"N": 39_813_120, "D": 12_582_912,  "final_loss": 2.8},
        {"N": 39_813_120, "D": 41_943_040,  "final_loss": 2.6},
        {"N": 71_479_296, "D": 7_340_032,   "final_loss": 2.9},
        {"N": 71_479_296, "D": 23_068_672,  "final_loss": 2.6},
    ]

    E, A, B, alpha, beta = fit_chinchilla_loss(fake_results)
    print(f"Fit:  E={E:.4f}  A={A:.3e}  B={B:.3e}  alpha={alpha:.3f}  beta={beta:.3f}")

    N_opt, D_opt, L_opt = predict_optimal(E, A, B, alpha, beta, TARGET_FLOPS)
    print(f"\nPredicted optimal for 48 B200-hours ({TARGET_FLOPS:.2e} FLOPs):")
    print(f"  N_opt = {N_opt:.3e} parameters")
    print(f"  D_opt = {D_opt:.3e} tokens")
    print(f"  L_opt = {L_opt:.4f}")
