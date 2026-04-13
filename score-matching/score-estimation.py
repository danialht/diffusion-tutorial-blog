import numpy as np
from pathlib import Path


def mixture_of_gaussians_score(x, mus, sigmas, weights):
    """
    Score function of a mixture of Gaussians:
      p(x) = sum_k w_k * N(x; mu_k, sigma_k^2)

    By the chain rule:
      score(x) = nabla_x log p(x) = sum_k r_k(x) * score_k(x)

    where r_k(x) = w_k * N(x; mu_k, sigma_k^2) / p(x)  (posterior responsibility)
    and   score_k(x) = -(x - mu_k) / sigma_k^2
    """
    mus     = np.array(mus)      # (K,)
    sigmas  = np.array(sigmas)   # (K,)
    weights = np.array(weights)  # (K,)
    weights = weights / weights.sum()

    x = np.asarray(x)           # (N,) or scalar

    # component densities: shape (K, N)
    component_densities = (
        weights[:, None]
        * np.exp(-0.5 * ((x[None, :] - mus[:, None]) / sigmas[:, None])**2)
        / (np.sqrt(2 * np.pi) * sigmas[:, None])
    )

    p_x = component_densities.sum(axis=0)                     # (N,)
    responsibilities = component_densities / p_x              # (K, N)
    component_scores = -(x[None, :] - mus[:, None]) / sigmas[:, None]**2  # (K, N)

    return (responsibilities * component_scores).sum(axis=0)  # (N,)

def gaussian_mixture_score_image():
    mus  = [-2.0, 2.0]
    sigmas  = [0.8, 1.2]
    weights = [0.2, 0.8]

    x = np.linspace(-6, 6, 500)
    score = mixture_of_gaussians_score(x, mus, sigmas, weights)

    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 13,
        "axes.titlesize": 14,
    })
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    # plot density
    p = sum(
        w * np.exp(-0.5 * ((x - m) / s)**2) / (np.sqrt(2 * np.pi) * s)
        for w, m, s in zip(weights, mus, sigmas)
    )
    axes[0].plot(x, p, color="steelblue")
    axes[0].set_ylabel("$p(x)$")
    axes[0].set_title("Mixture of Gaussians — density and score")

    # plot score
    axes[1].plot(x, score, color="tomato")
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel(r"$\nabla_x \log p(x)$")
    axes[1].set_xlabel("$x$")

    plt.tight_layout()
    fig.savefig(Path(__file__).parent.parent / "media/score_mixture_gaussians.png", bbox_inches="tight", dpi=150)

def smoothing_point_distribution_in_2d():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 13,
        "axes.titlesize": 14,
    })

    rng = np.random.default_rng(42)

    # mixture of 3 Gaussians in 2D
    components = [
        dict(mu=np.array([-2.5,  1.5]), cov=np.array([[0.4, 0.1], [0.1, 0.3]]), n=600),
        dict(mu=np.array([ 2.0,  2.0]), cov=np.array([[0.3, -0.1], [-0.1, 0.5]]), n=800),
        dict(mu=np.array([ 0.0, -2.0]), cov=np.array([[0.6, 0.2], [0.2, 0.4]]), n=600),
    ]
    pts = np.vstack([
        rng.multivariate_normal(c["mu"], c["cov"], c["n"])
        for c in components
    ])

    lim = 5.5

    noise_levels = [0.5, 1.0, 1.5]
    noisy_pts = [pts + rng.normal(scale=s, size=pts.shape) for s in noise_levels]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    scatter_kw = dict(s=3, alpha=0.4, color="steelblue", linewidths=0)
    axes[0].scatter(pts[:, 0], pts[:, 1], **scatter_kw)
    axes[0].set_title("Original $p(x)$")

    for ax, pts_n, sigma in zip(axes[1:], noisy_pts, noise_levels):
        ax.scatter(pts_n[:, 0], pts_n[:, 1], **scatter_kw)
        ax.set_title(f"$\\sigma = {sigma}$")

    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    fig.suptitle("Effect of adding noise to a point distribution", y=1.02)
    plt.tight_layout()
    fig.savefig(Path(__file__).parent.parent / "media/smoothing_2d.png", bbox_inches="tight", dpi=150)

if __name__ == "__main__":
    gaussian_mixture_score_image()
    smoothing_point_distribution_in_2d()