from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(42)


def sample_faces(
    n_total: int = 12000, mouth_frac: float = 0.6, eyebrow_frac: float = 0.08
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two corresponding point clouds:
      X0 = smiley distribution, X1 = frowny distribution
    """
    n_brow_each = max(1, int(n_total * eyebrow_frac) // 2)
    n_brow_total = 2 * n_brow_each

    n_remaining = n_total - n_brow_total
    n_mouth = int(n_remaining * mouth_frac)
    n_eye_total = n_remaining - n_mouth
    n_eye_each = n_eye_total // 2
    n_mouth += n_eye_total - (2 * n_eye_each)

    # Eyes
    eye_sigma = 0.09
    left_eye_center = np.array([-0.45, 0.45])
    right_eye_center = np.array([0.45, 0.45])

    left_eye = left_eye_center + rng.normal(scale=eye_sigma, size=(n_eye_each, 2))
    right_eye = right_eye_center + rng.normal(scale=eye_sigma, size=(n_eye_each, 2))
    left_brow_src = left_eye_center + rng.normal(scale=eye_sigma, size=(n_brow_each, 2))
    right_brow_src = right_eye_center + rng.normal(scale=eye_sigma, size=(n_brow_each, 2))

    # Frown eyes are slightly lower than smile eyes
    frown_eye_shift_y = -0.1
    left_eye_frown_center = left_eye_center + np.array([0.0, frown_eye_shift_y])
    right_eye_frown_center = right_eye_center + np.array([0.0, frown_eye_shift_y])
    left_eye_frown = left_eye_frown_center + rng.normal(scale=eye_sigma, size=(n_eye_each, 2))
    right_eye_frown = right_eye_frown_center + rng.normal(scale=eye_sigma, size=(n_eye_each, 2))

    # Mouth latent parameters (shared between smile/frown)
    theta = rng.uniform(np.deg2rad(210), np.deg2rad(330), size=n_mouth)
    r = 0.75 + rng.normal(scale=0.04, size=n_mouth)
    noise = rng.normal(scale=0.035, size=(n_mouth, 2))

    # Smiley lower arc
    mouth_smile = np.column_stack([r * np.cos(theta), r * np.sin(theta)]) + noise

    # Frowny upper arc (reflect y)
    mouth_frown = np.column_stack([r * np.cos(theta), -r * np.sin(theta)]) + noise
    mouth_frown_y_shift = np.mean(mouth_frown[:, 1]) - np.mean(mouth_smile[:, 1])
    mouth_frown[:, 1] -= mouth_frown_y_shift

    # Frowny eyebrows: short noisy line segments with lower inner corners.
    brow_u_left = rng.uniform(-1.0, 1.0, size=n_brow_each)
    brow_u_right = rng.uniform(-1.0, 1.0, size=n_brow_each)

    left_brow = np.column_stack(
        [
            left_eye_frown_center[0] + 0.21 * brow_u_left,
            left_eye_center[1] + 0.32 - 0.06 * brow_u_left,
        ]
    ) + rng.normal(scale=(0.012, 0.012), size=(n_brow_each, 2))

    right_brow = np.column_stack(
        [
            right_eye_frown_center[0] + 0.21 * brow_u_right,
            right_eye_center[1] + 0.32 + 0.06 * brow_u_right,
        ]
    ) + rng.normal(scale=(0.012, 0.012), size=(n_brow_each, 2))

    X0 = np.vstack([left_eye, right_eye, mouth_smile, left_brow_src, right_brow_src])
    X1 = np.vstack([left_eye_frown, right_eye_frown, mouth_frown, left_brow, right_brow])

    # Appears as if samples from a distribution while maintaining correspondence for interpolation
    perm = rng.permutation(X0.shape[0])
    return X0[perm], X1[perm]


def point_cloud_to_density(
    points: np.ndarray,
    xlim: tuple[float, float] = (-1.3, 1.3),
    ylim: tuple[float, float] = (-1.3, 1.3),
    bins: int = 220,
    sigma: float = 2.0,
) -> np.ndarray:
    # histogram2d returns [x_bin, y_bin]; transpose for imshow
    density = np.histogram2d(
        points[:, 0], points[:, 1], bins=bins, range=[xlim, ylim], density=True
    )[0].T
    return gaussian_filter(density, sigma=sigma)


def create_animation_axes(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    figsize: tuple[float, float] = (6.4, 6.4),
    facecolor: str = "black",
):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    ax.set_aspect("equal", "box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def render_animation(
    fig, update_fn, n_frames: int, output_path: str, interval: int = 45, fps: int = 24, dpi: int = 160
) -> None:
    animation = FuncAnimation(fig, update_fn, frames=n_frames, interval=interval, blit=True)
    animation.save(output_path, dpi=dpi, fps=fps)
    plt.close(fig)


def main() -> None:
    # Figure params
    xlim = (-1.3, 1.3)
    ylim = (-1.3, 1.3)
    bins = 220
    sigma = 2.2
    n_frames = 90

    X0, X1 = sample_faces()
    H0 = point_cloud_to_density(X0, xlim, ylim, bins, sigma)
    vmax = H0.max() * 1.05  # stable color scale
    noise_half = rng.normal(scale=1 / n_frames, size=(n_frames // 2, *X0.shape))
    noise_half = np.cumsum(noise_half, axis=0)
    noise = np.concatenate([noise_half, noise_half[::-1]], axis=0)
    if n_frames % 2:
        noise = np.concatenate([noise, np.zeros((1, *X0.shape))], axis=0)

    def interpolate_points(frame: int) -> np.ndarray:
        t = frame / (n_frames - 1)
        points = (1 - t) * X0 + t * X1 + noise[frame]
        return points

    media_dir = Path(__file__).resolve().parent / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Density animation
    fig_density, ax_density = create_animation_axes(xlim, ylim)
    img = ax_density.imshow(
        H0,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=vmax,
        interpolation="bilinear",
        animated=True,
    )

    def update_density(frame: int):
        points = interpolate_points(frame)
        Ht = point_cloud_to_density(points, xlim, ylim, bins, sigma)
        img.set_array(Ht)
        return (img,)

    render_animation(
        fig_density,
        update_density,
        n_frames,
        str(media_dir / "smiley_to_frowny_density.mp4"),
    )

    # Point-only animation
    fig_points, ax_points = create_animation_axes(xlim, ylim)
    Xt0 = interpolate_points(0)
    pts = ax_points.scatter(
        Xt0[:, 0],
        Xt0[:, 1],
        s=3,
        c="white",
        alpha=0.85,
        edgecolors="none",
        animated=True,
    )

    def update_points(frame: int):
        points = interpolate_points(frame)
        pts.set_offsets(points)
        return (pts,)

    render_animation(
        fig_points,
        update_points,
        n_frames,
        str(media_dir / "smiley_to_frowny_points.mp4"),
    )


if __name__ == "__main__":
    main()
