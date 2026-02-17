"""Render trajectory video from a checkpoint without running eval."""

import argparse
import os

import imageio
import numpy as np
import torch
import tqdm
from PIL import Image
from gsplat.rendering import rasterization

from scipy.spatial.transform import Rotation

from examples.datasets.colmap import Parser
from examples.datasets.traj import generate_interpolated_path


def apply_swerve(camtoworlds, amplitude, periods):
    """Apply sinusoidal lateral swerve to camera trajectory using Frenet frame.

    Args:
        camtoworlds: (N, 4, 4) camera-to-world matrices.
        amplitude: Lateral offset in meters (scene units).
        periods: Number of full sine wave cycles over the trajectory.
    Returns:
        Modified (N, 4, 4) camera-to-world matrices.
    """
    N = len(camtoworlds)
    positions = camtoworlds[:, :3, 3].copy()  # (N, 3)

    # Compute tangent vectors (forward difference, smoothed)
    tangents = np.zeros_like(positions)
    tangents[:-1] = positions[1:] - positions[:-1]
    tangents[-1] = tangents[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    tangents = tangents / norms

    # Use camera's local right vector as the lateral direction
    # Camera right = first column of rotation matrix (x-axis in camera frame)
    right_vectors = camtoworlds[:, :3, 0].copy()  # (N, 3)

    # Sinusoidal offset
    t = np.linspace(0, 2 * np.pi * periods, N)
    offsets = amplitude * np.sin(t)  # (N,)

    result = camtoworlds.copy()
    result[:, :3, 3] = positions + right_vectors * offsets[:, None]
    return result


def render_one(means, quats, scales, opacities, colors, sh_degree,
               camtoworlds, K, width, height, output_path, fps, interp, device,
               image_paths=None):
    """Render a single video from the given camera poses.

    Args:
        image_paths: If provided, render side-by-side (3DGS left, GT right).
    """
    if interp:
        c2w = camtoworlds[5:-5]
        c2w_interp = generate_interpolated_path(c2w[:, :3, :4], 1)
        c2w_all = np.concatenate([
            c2w_interp,
            np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(c2w_interp), axis=0),
        ], axis=1)
        # Interpolation changes frame count, so GT comparison not available
        image_paths = None
    else:
        c2w_all = camtoworlds
    c2w_all = torch.from_numpy(c2w_all).float().to(device)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)

    for i in tqdm.trange(len(c2w_all), desc=f"Rendering {os.path.basename(output_path)}"):
        viewmat = torch.linalg.inv(c2w_all[i:i+1])
        renders, _, _ = rasterization(
            means, quats, scales, opacities, colors,
            viewmat, K[None], width, height,
            sh_degree=sh_degree, render_mode="RGB", packed=False,
        )
        rgb = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)
        rendered = (rgb.cpu().numpy() * 255).astype(np.uint8)

        if image_paths is not None:
            gt = np.array(Image.open(image_paths[i]).resize((width, height)))
            if gt.ndim == 2:
                gt = np.stack([gt] * 3, axis=-1)
            canvas = np.concatenate([rendered, gt], axis=1)
        else:
            canvas = rendered

        writer.append_data(canvas)

    writer.close()
    print(f"Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint file")
    parser.add_argument("--data_dir", required=True, help="Path to COLMAP dataset")
    parser.add_argument("--output", default=None, help="Output video path (or directory for --camera all)")
    parser.add_argument("--camera", default=None,
                        help="Camera name prefix to filter (e.g. camera1), or 'all' for each camera separately")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-interp", action="store_true", help="Use raw poses without interpolation")
    parser.add_argument("--no-gt", action="store_true", help="Skip GT side-by-side comparison")
    parser.add_argument("--swerve", type=float, default=None,
                        help="Lateral swerve amplitude in scene units (e.g. 1.0 for ±1m)")
    parser.add_argument("--swerve-periods", type=float, default=3.0,
                        help="Number of sine wave cycles over the trajectory (default: 3)")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load dataset
    colmap_parser = Parser(data_dir=args.data_dir, factor=1, normalize=True, test_every=8)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)["splats"]
    means = ckpt["means"]
    quats = torch.nn.functional.normalize(ckpt["quats"], p=2, dim=-1)
    scales = torch.exp(ckpt["scales"])
    opacities = torch.sigmoid(ckpt["opacities"])
    sh0 = ckpt["sh0"]
    shN = ckpt["shN"]
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(colors.shape[-2] ** 0.5) - 1
    print(f"Loaded {len(means)} gaussians, sh_degree={sh_degree}")

    K = torch.from_numpy(list(colmap_parser.Ks_dict.values())[0]).float().to(device)
    width, height = list(colmap_parser.imsize_dict.values())[0]
    interp = not args.no_interp

    image_names = list(colmap_parser.image_names)
    image_paths = list(colmap_parser.image_paths)
    camtoworlds = colmap_parser.camtoworlds

    use_gt = not args.no_gt and not interp

    # Scene scale info for swerve
    scale_factor = np.linalg.norm(colmap_parser.transform[:3, :3], ord=2)
    if args.swerve is not None:
        scene_swerve = args.swerve * scale_factor
        print(f"Swerve: ±{args.swerve:.1f}m real -> ±{scene_swerve:.2f} scene units (scale={scale_factor:.3f})")

    def maybe_swerve(c2w):
        if args.swerve is not None:
            return apply_swerve(c2w, args.swerve * scale_factor, args.swerve_periods)
        return c2w

    if args.camera == "all":
        prefixes = sorted(set(n.split("_")[0] for n in image_names))
        print(f"Found {len(prefixes)} cameras: {', '.join(prefixes)}")
        output_dir = args.output or "videos"
        os.makedirs(output_dir, exist_ok=True)
        for prefix in prefixes:
            mask = np.array([n.startswith(prefix + "_") or n.startswith(prefix + ".") or n == prefix
                             for n in image_names])
            c2w = maybe_swerve(camtoworlds[mask])
            paths = [p for p, m in zip(image_paths, mask) if m] if use_gt else None
            print(f"\n{prefix}: {len(c2w)} poses")
            out_path = os.path.join(output_dir, f"{prefix}.mp4")
            render_one(means, quats, scales, opacities, colors, sh_degree,
                       c2w, K, width, height, out_path, args.fps, interp, device,
                       image_paths=paths)
    else:
        if args.camera:
            mask = np.array([n.startswith(args.camera) for n in image_names])
            camtoworlds = camtoworlds[mask]
            paths = [p for p, m in zip(image_paths, mask) if m] if use_gt else None
            print(f"Filtered to {args.camera}: {len(camtoworlds)} poses")
        else:
            paths = image_paths if use_gt else None
        camtoworlds = maybe_swerve(camtoworlds)
        output_path = args.output or "traj_video.mp4"
        render_one(means, quats, scales, opacities, colors, sh_degree,
                   camtoworlds, K, width, height, output_path, args.fps, interp, device,
                   image_paths=paths)


if __name__ == "__main__":
    main()
