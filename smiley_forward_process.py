"""
Animate the forward noising process on the smiley point cloud:
  q(x_{t+1} | x_t) = N(sqrt(alpha_t) * x_t, (1 - alpha_t) * I_d * 0.25)
i.e. x_{t+1} = sqrt(alpha_t) * x_t + sqrt(1 - alpha_t) * eps,  eps ~ N(0, I*0.25).
"""
from pathlib import Path

import numpy as np

from distribution_transport import create_animation_axes, render_animation, sample_faces

rng = np.random.default_rng(42)
_ROOT = Path(__file__).resolve().parent


def forward_trajectory(
    x0: np.ndarray, n_frames: int, alpha_start: float = 0.999, alpha_end: float = 0.92
) -> np.ndarray:
    """Sequential samples x_0, ..., x_{n_frames-1} under the Markov kernel above."""
    traj = np.empty((n_frames,) + x0.shape)
    traj[0] = x0
    alphas = np.linspace(alpha_start, alpha_end, n_frames - 1)
    x = x0.astype(np.float64, copy=True)
    for k in range(n_frames - 1):
        eps = rng.standard_normal(size=x.shape)*0.25
        a = alphas[k]
        x = np.sqrt(a) * x + np.sqrt(1.0 - a) * eps
        traj[k + 1] = x
    return traj


def main() -> None:
    xlim = (-1.3, 1.3)
    ylim = (-1.3, 1.3)
    n_frames = 90

    X0, _ = sample_faces()
    traj = forward_trajectory(X0, n_frames)

    fig_points, ax_points = create_animation_axes(xlim, ylim)
    pts = ax_points.scatter(
        traj[0][:, 0],
        traj[0][:, 1],
        s=3,
        c="white",
        alpha=0.85,
        edgecolors="none",
        animated=True,
    )

    def update_points(frame: int):
        pts.set_offsets(traj[frame])
        return (pts,)

    render_animation(
        fig_points,
        update_points,
        n_frames,
        str(_ROOT / "media" / "smiley_forward_process.mp4"),
        fps=12,
    )


if __name__ == "__main__":
    main()
