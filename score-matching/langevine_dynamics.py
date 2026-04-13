import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    N = 10_000  # num samples
    K = 100
    EPSILON = 0.1

    x = 4*(np.random.rand(N)-0.5)  # initial distribution = Uniform(-2, 2)
    mu, sigma = 1, 1.5
    score_fn = lambda x: -(x - mu) / sigma**2

    snapshots = []
    snapshot_iters = list(range(0, K + 1, 20))

    x_plt = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    y_true = 1/(2*np.pi*sigma**2)**.5 * np.exp(-0.5 * (x_plt - mu)**2 / sigma**2)

    for i in range(K):
        if i in snapshot_iters:
            snapshots.append((i, x.copy()))
        z = np.random.randn(N)
        x += EPSILON/2 * score_fn(x) + EPSILON**.5 * z
    snapshots.append((K, x.copy()))  # final snapshot

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)

    for ax, (it, samples) in zip(axes, snapshots):
        ax.hist(samples, density=True, bins=40, alpha=0.7, color="steelblue", label="Samples")
        ax.plot(x_plt, y_true, color="tomato", linewidth=2, label="True $p(x)$")
        ax.set_title(f"iter {it}")
        ax.set_xlabel("$x$")
        ax.legend(fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("density")

    fig.suptitle("Langevin Dynamics: Convergence to Target Distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent.parent / "media/langevine_dynamics.png", bbox_inches="tight", dpi=150)

if  __name__ == "__main__":
    main()