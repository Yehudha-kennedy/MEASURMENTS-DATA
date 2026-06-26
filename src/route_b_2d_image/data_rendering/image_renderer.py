"""
Route B: Image Renderer
=======================
Converts 1-D S11 spectral vectors into 224×224 2-D images.

Encodings implemented:
  B1 – Single-channel line trace (Real part only)        → (N, 224, 224)
  B2 – Dual-channel rasterised maps (Re + Im)            → (N, 224, 224, 2)
  B3 – Gramian Angular Field (GAF) from the Real channel → (N, 224, 224)
"""
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt


# ======================================================================
# B1: Single-Channel Line-Trace Image
# ======================================================================

def render_traces_to_images(X: np.ndarray, output_shape: tuple = (224, 224)) -> np.ndarray:
    """Render Re channel as a line-trace grayscale image (B1).

    Parameters
    ----------
    X : ndarray, shape (N, 4001, 2)
    output_shape : tuple
        (height, width) of the output image.

    Returns
    -------
    images : ndarray, shape (N, 224, 224), dtype uint8
    """
    N, bins, _ = X.shape
    h, w = output_shape
    images = np.zeros((N, h, w), dtype=np.uint8)

    global_min = float(np.min(X[:, :, 0]))
    global_max = float(np.max(X[:, :, 0]))

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    x_axis = np.arange(bins)

    print("Rendering B1 (single-channel line trace)...")
    for i in range(N):
        if i % 100 == 0:
            print(f"  B1: {i}/{N}")

        ax.clear()
        ax.set_ylim(global_min, global_max)
        ax.set_xlim(0, bins - 1)
        ax.set_axis_off()

        ax.plot(x_axis, X[i, :, 0], color="black", linewidth=1.0, antialiased=False)

        fig.canvas.draw()

        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        r, g, b = img[:, :, 1], img[:, :, 2], img[:, :, 3]
        img_gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)

        images[i] = img_gray

    plt.close(fig)
    return images


# ======================================================================
# B2: Dual-Channel Rasterised Maps
# ======================================================================

def render_dual_channel_images(
    X: np.ndarray, output_shape: tuple = (224, 224)
) -> np.ndarray:
    """Render Re and Im channels as two separate grayscale images (B2).

    Both channels share the same standardised axis ranges so the visual
    scale is consistent.

    Parameters
    ----------
    X : ndarray, shape (N, 4001, 2)
    output_shape : tuple
        (height, width) of each output image.

    Returns
    -------
    images : ndarray, shape (N, 224, 224, 2), dtype uint8
        Channel 0 → Re image, channel 1 → Im image.
    """
    N, bins, _ = X.shape
    h, w = output_shape
    images = np.zeros((N, h, w, 2), dtype=np.uint8)

    # Shared axis ranges across BOTH channels for consistent scaling
    global_min = float(min(np.min(X[:, :, 0]), np.min(X[:, :, 1])))
    global_max = float(max(np.max(X[:, :, 0]), np.max(X[:, :, 1])))

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    x_axis = np.arange(bins)

    print("Rendering B2 (dual-channel rasterised maps)...")
    for i in range(N):
        if i % 100 == 0:
            print(f"  B2: {i}/{N}")

        for ch in range(2):
            ax.clear()
            ax.set_ylim(global_min, global_max)
            ax.set_xlim(0, bins - 1)
            ax.set_axis_off()

            ax.plot(
                x_axis,
                X[i, :, ch],
                color="black",
                linewidth=1.0,
                antialiased=False,
            )

            fig.canvas.draw()

            img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            r, g, b = img[:, :, 1], img[:, :, 2], img[:, :, 3]
            img_gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)

            images[i, :, :, ch] = img_gray

    plt.close(fig)
    return images


# ======================================================================
# B3: Gramian Angular Field (GAF)
# ======================================================================

def render_gaf_images(
    X: np.ndarray, output_size: int = 224
) -> np.ndarray:
    """Compute Gramian Angular Summation Field from Re channel (B3).

    Steps per sample:
      1. Take Re channel (4001 values).
      2. Downsample to *output_size* bins via linear interpolation.
      3. Rescale to [−1, 1].
      4. GAF[i, j] = cos(arccos(x_i) + arccos(x_j)).
      5. Scale to [0, 255] uint8.

    Parameters
    ----------
    X : ndarray, shape (N, 4001, 2)
    output_size : int
        Side length of the square GAF image (default 224).

    Returns
    -------
    images : ndarray, shape (N, 224, 224), dtype uint8
    """
    N, bins, _ = X.shape
    images = np.zeros((N, output_size, output_size), dtype=np.uint8)

    x_orig = np.linspace(0, 1, bins)
    x_new = np.linspace(0, 1, output_size)

    print("Rendering B3 (Gramian Angular Field)...")
    for i in range(N):
        if i % 100 == 0:
            print(f"  B3: {i}/{N}")

        # Downsample Re channel
        series = np.interp(x_new, x_orig, X[i, :, 0])

        # Rescale to [-1, 1]
        s_min = series.min()
        s_max = series.max()
        denom = s_max - s_min
        if denom == 0:
            denom = 1e-8
        series_scaled = 2.0 * (series - s_min) / denom - 1.0
        # Clip to handle floating-point rounding
        series_scaled = np.clip(series_scaled, -1.0, 1.0)

        # GAF: cos(arccos(x_i) + arccos(x_j))
        phi = np.arccos(series_scaled)  # (output_size,)
        gaf = np.cos(phi[:, None] + phi[None, :])  # (output_size, output_size)

        # Scale [-1, 1] → [0, 255]
        gaf_uint8 = ((gaf + 1.0) / 2.0 * 255.0).astype(np.uint8)
        images[i] = gaf_uint8

    return images


# ======================================================================
# CLI / main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Route B image encodings (B1, B2, B3)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing dataset_grouped.npz",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/route_b",
        help="Output directory for rendered images",
    )
    parser.add_argument(
        "--skip_b1", action="store_true", help="Skip B1 rendering"
    )
    parser.add_argument(
        "--skip_b2", action="store_true", help="Skip B2 rendering"
    )
    parser.add_argument(
        "--skip_b3", action="store_true", help="Skip B3 rendering"
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent.parent.parent
    data_path = base / args.data_dir / "dataset_grouped.npz"
    out_path = base / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        # Fall back to smoke test with dummy data
        print(f"Warning: {data_path} not found. Running smoke test with dummy data.")
        rng = np.random.default_rng(0)
        N_SAMPLES = 6
        X = rng.standard_normal((N_SAMPLES, 4001, 2)).astype(np.float32)
        y = rng.integers(0, 3, size=N_SAMPLES)
        groups = np.arange(N_SAMPLES)
    else:
        data = np.load(data_path)
        X, y, groups = data["X"], data["y"], data["groups"]

    # ----- B1 -----
    if not args.skip_b1:
        images_b1 = render_traces_to_images(X)
        np.savez_compressed(
            out_path / "dataset_images_b1.npz", X=images_b1, y=y, groups=groups
        )
        print(f"Saved B1: {images_b1.shape} -> {out_path / 'dataset_images_b1.npz'}")

    # ----- B2 -----
    if not args.skip_b2:
        images_b2 = render_dual_channel_images(X)
        np.savez_compressed(
            out_path / "dataset_images_b2.npz", X=images_b2, y=y, groups=groups
        )
        print(f"Saved B2: {images_b2.shape} -> {out_path / 'dataset_images_b2.npz'}")

    # ----- B3 -----
    if not args.skip_b3:
        images_b3 = render_gaf_images(X)
        np.savez_compressed(
            out_path / "dataset_images_b3.npz", X=images_b3, y=y, groups=groups
        )
        print(f"Saved B3: {images_b3.shape} -> {out_path / 'dataset_images_b3.npz'}")

    print("=== image_renderer.py complete ===")
