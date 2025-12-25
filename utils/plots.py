
import matplotlib.pyplot as plt


def plot_losses(
    losses: dict[str, list[float]],
    log_y: bool = False,
    title: str = "Training Losses",
):
    """
    losses (dict): {"loss_name": [values per iteration], ...}
    """
    plt.figure(figsize=(6, 6))

    for name, values in losses.items():
        if len(values) == 0:
            continue
        plt.plot(values, label=name)

    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    if log_y:
        plt.yscale("log")

    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()