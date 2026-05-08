B200_BF16_PEAK_FLOPS_PER_SECOND = 4.5e15
PROXY_HIDDEN_SIZE = 128


def compute_flops_and_b200_hours(num_hidden_layers, hidden_size, total_train_tokens):
    non_embedding_parameters = 12 * num_hidden_layers * hidden_size ** 2
    total_flops = 6 * non_embedding_parameters * total_train_tokens
    b200_hours = total_flops / (B200_BF16_PEAK_FLOPS_PER_SECOND * 3600)
    return total_flops, b200_hours


def scale_learning_rate(proxy_lr, target_hidden_size, exponent=1.0):
    return proxy_lr * (PROXY_HIDDEN_SIZE / target_hidden_size) ** exponent


if __name__ == "__main__":
    flops, hours = compute_flops_and_b200_hours(9, 448, 1_048_576)
    print(f"Example run: {flops:.3e} FLOPs, {hours:.3e} B200-hours")
