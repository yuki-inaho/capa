#!/usr/bin/env python3
"""Export raw aligned RGB-D and CAPA-refined depth as separate colored PLYs.

Input data and processing contract:

* ``--rgb`` is an RGB image at the aligned output resolution.
* ``--raw-depth`` is a dilation-free, RGB-FOV-aligned single-channel uint16
  depth PNG. Its values are converted to metres using ``--depth-scale``.
* ``--refined-depth`` is a CAPA ``depth_pred_nhw`` tensor. CAPA outputs metres
  and is kept separate from the raw observation data; no fusion or overwrite
  is performed here.
* ``--camera`` supplies the RGB pinhole intrinsics used for both point clouds.

The raw PLY contains only positive finite sensor observations. The refined PLY
contains every positive finite CAPA prediction. Both use the OpenCV camera
coordinate convention: x right, y down, z forward, with RGB vertex colors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import yaml


def load_rgb(path: Path) -> np.ndarray:
    # RGB is the color source for both point clouds and defines the target HxW.
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_uint16_depth_m(path: Path, depth_scale: float) -> np.ndarray:
    # Sensor depth is stored as uint16 pixels. The default scale assumes mm,
    # so e.g. a pixel value of 750 becomes 0.750 m before backprojection.
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
        raise ValueError(
            f"Expected a single-channel uint16 depth PNG, got shape={depth_raw.shape}, "
            f"dtype={depth_raw.dtype}"
        )
    return depth_raw.astype(np.float32) / depth_scale


def load_refined_depth(path: Path) -> np.ndarray:
    # CAPA writes a dense HxW prediction in metres under depth_pred_nhw.
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "depth_pred_nhw" not in data:
        raise KeyError(f"{path} does not contain 'depth_pred_nhw'")
    depth = data["depth_pred_nhw"]
    if depth.ndim == 3:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW or 1xHxW refined depth, got {tuple(depth.shape)}")
    return depth.detach().cpu().numpy().astype(np.float32)


def load_rgb_intrinsics(path: Path) -> tuple[float, float, float, float]:
    # Use calibrated RGB intrinsics; synthetic/default intrinsics would distort
    # the 3-D positions even when the depth values themselves are correct.
    camera = yaml.safe_load(path.read_text())
    k = camera.get("K")
    if not isinstance(k, list) or len(k) != 9:
        raise ValueError(f"Expected a 3x3 camera matrix K in {path}")
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


def depth_to_points(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    # Backproject z-depth with x=(u-cx)z/fx and y=(v-cy)z/fy. The valid mask
    # deliberately differs by input: raw observations stay sparse, while CAPA
    # predictions normally produce a dense cloud.
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: rgb={rgb.shape[:2]}, depth={depth_m.shape}")
    fx, fy, cx, cy = intrinsics
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    ys, xs = np.nonzero(valid)
    z = depth_m[ys, xs]
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    vertices = np.stack((x, y, z), axis=1)
    colors = np.ascontiguousarray(rgb[ys, xs], dtype=np.uint8)
    return vertices, colors


def save_colored_ply(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
    cloud.export(path)
    print(f"saved {path}: points={len(vertices)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--raw-depth", type=Path, required=True)
    parser.add_argument("--refined-depth", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True, help="RGB camera YAML")
    parser.add_argument("--raw-out", type=Path, required=True)
    parser.add_argument("--refined-out", type=Path, required=True)
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="uint16 depth units per metre; default=1000 for millimetres",
    )
    args = parser.parse_args()

    rgb = load_rgb(args.rgb)
    raw_depth_m = load_uint16_depth_m(args.raw_depth, args.depth_scale)
    refined_depth_m = load_refined_depth(args.refined_depth)
    intrinsics = load_rgb_intrinsics(args.camera)

    # Export two independent products so the effect of CAPA can be inspected
    # against the original sensor observations without a fusion policy.
    raw_vertices, raw_colors = depth_to_points(raw_depth_m, rgb, intrinsics)
    refined_vertices, refined_colors = depth_to_points(refined_depth_m, rgb, intrinsics)
    save_colored_ply(args.raw_out, raw_vertices, raw_colors)
    save_colored_ply(args.refined_out, refined_vertices, refined_colors)


if __name__ == "__main__":
    main()
