#!/usr/bin/env python3
"""
Offline reconstruction pipeline using Depth Anything 3 Large.

Pipeline:
    DA3-LARGE -> depth + camera poses -> point clouds -> optional ICP refinement -> TSDF

DA3-LARGE-1.1 does not expose the 3D Gaussian Splatting branch, but it does provide
relative depth and camera poses. This script uses those outputs as the modern V3
front-end and fuses them into a preview reconstruction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from main import RuntimeConfig, depth_to_pointcloud, load_calibration_intrinsics, preprocess_frame, save_pointcloud_ply


def parse_args():
    parser = argparse.ArgumentParser(description="DA3 Large depth/pose to point cloud and TSDF pipeline.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_dir", default="./da3_large_fusion")
    parser.add_argument("--model", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--process_res", type=int, default=384)
    parser.add_argument("--process_res_method", default="upper_bound_resize")
    parser.add_argument("--da3_batch_size", type=int, default=0, help="Process frames in chunks to reduce VRAM. 0 means all frames at once.")
    parser.add_argument("--export_da3_outputs", action="store_true", help="Save DA3 npy outputs. Disabled by default to reduce RAM/disk.")
    parser.add_argument("--fps_sample", type=float, default=1.0)
    parser.add_argument("--frame_step", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=48)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--rotate", choices=("none", "90cw", "90ccw", "180"), default="90cw")
    parser.add_argument("--wb", choices=("none", "grayworld", "clahe"), default="grayworld")
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--resolution_scale", type=float, default=1.0)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Manual scale factor for DA3 relative depth before fusion.")
    parser.add_argument("--pointcloud_stride", type=int, default=1)
    parser.add_argument("--save_pointclouds", action="store_true")
    parser.add_argument("--dense_cloud", action="store_true", help="Export an unvoxelized accumulated dense point cloud.")
    parser.add_argument("--dense_max_points", type=int, default=0, help="Optional random cap for dense cloud points. 0 keeps all points.")
    parser.add_argument("--preview_voxel", type=float, default=0.03, help="Voxel size only for fused_da3_cloud.ply preview. 0 disables preview downsample.")
    parser.add_argument("--export_format", default="mini_npz", help="Optional DA3 export format: mini_npz, glb, ply, depth_vis, etc.")
    parser.add_argument("--use_ray_pose", action="store_true")
    parser.add_argument("--ref_view_strategy", default="saddle_balanced")
    parser.add_argument("--icp_refine", action="store_true")
    parser.add_argument("--icp_voxel", type=float, default=0.06)
    parser.add_argument("--icp_distance", type=float, default=0.08)
    parser.add_argument("--icp_strict", action="store_true", help="Use stricter multi-scale ICP defaults.")
    parser.add_argument("--icp_voxels", default="", help="Comma-separated multi-scale voxels, e.g. 0.08,0.04,0.02")
    parser.add_argument("--icp_distances", default="", help="Comma-separated ICP distances, e.g. 0.10,0.05,0.025")
    parser.add_argument("--icp_max_rmse", type=float, default=0.0, help="Reject refined pose if final ICP RMSE is above this value. 0 disables.")
    parser.add_argument("--icp_min_fitness", type=float, default=0.0, help="Reject refined pose if final ICP fitness is below this value. 0 disables.")
    parser.add_argument("--remove_outliers", action="store_true", help="Apply statistical outlier removal before ICP.")
    parser.add_argument("--outlier_nb_neighbors", type=int, default=24)
    parser.add_argument("--outlier_std_ratio", type=float, default=1.5)
    parser.add_argument("--tsdf", action="store_true")
    parser.add_argument("--tsdf_voxel", type=float, default=0.03)
    parser.add_argument("--tsdf_sdf_trunc", type=float, default=0.12)
    return parser.parse_args()


def make_config(args) -> RuntimeConfig:
    return RuntimeConfig(
        ip="",
        video=args.video,
        loop_video=False,
        realtime_video=False,
        port=8080,
        stream_path="/video",
        use_gpu=args.device == "cuda",
        checkpoint=None,
        repo_path=None,
        model_backend="da3",
        encoder="vitl",
        max_depth=args.max_depth,
        input_size=args.process_res,
        resolution_scale=args.resolution_scale,
        rotate=args.rotate,
        wb=args.wb,
        brightness=args.brightness,
        contrast=max(args.contrast, 0.0),
        saturation=max(args.saturation, 0.0),
        gamma=max(args.gamma, 0.05),
        width=0,
        height=0,
        frame_skip=0,
        fps_limit=0.0,
        display_scale=1.0,
        display_every=1,
        print_every=30,
        fast_vis=True,
        profile_interval=0.0,
        torch_compile=False,
        low_latency=False,
        capture_fps=0.0,
        depth_output_scale=1.0,
        pointcloud=False,
        save_dir=None,
        save_every=0,
        pointcloud_save_dir=None,
        pointcloud_save_every=0,
        pointcloud_stride=max(args.pointcloud_stride, 1),
        pointcloud_max_depth=args.max_depth,
        live_registration=False,
        registration_every=1,
        registration_stride=4,
        registration_voxel=0.08,
        registration_feature_radius=0.35,
        registration_ransac_distance=0.16,
        registration_icp_distance=args.icp_distance,
        registration_max_points=60000,
        registration_map_voxel=0.04,
        registration_device="cpu",
        registration_backend="teaser",
        teaser_noise_bound=0.08,
        teaser_max_correspondences=5000,
        teaser_mutual_filter=False,
        calibration=args.calibration,
        window_width=1280,
        window_height=720,
        camera_fx=None,
        camera_fy=None,
        camera_cx=None,
        camera_cy=None,
        reconnect_delay=1.0,
        fp16=True,
    )


def extract_frames(args, frames_dir: Path, cfg: RuntimeConfig) -> list[str]:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = args.frame_step if args.frame_step > 0 else max(1, int(round(fps / max(args.fps_sample, 0.001))))
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    paths = []
    frame_index = args.start_frame
    while len(paths) < args.max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (frame_index - args.start_frame) % step == 0:
            frame = preprocess_frame(frame, cfg)
            path = frames_dir / f"frame_{len(paths):04d}_{frame_index:06d}.png"
            cv2.imwrite(str(path), frame)
            paths.append(str(path))
        frame_index += 1
    cap.release()
    return paths


def scale_intrinsics(intrinsics: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scaled = intrinsics.copy().astype(np.float64)
    scaled[0, :] *= dst_w / float(src_w)
    scaled[1, :] *= dst_h / float(src_h)
    return scaled


def normalize_extrinsics(extrinsics, n: int) -> np.ndarray:
    if extrinsics is None:
        print("[da3] no extrinsics returned; using identity poses")
        return np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
    ext = np.asarray(extrinsics, dtype=np.float64)
    print(f"[da3] extrinsics shape={ext.shape}")
    if ext.shape == (n, 4, 4):
        return ext
    if ext.shape == (n, 3, 4):
        out = np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
        out[:, :3, :4] = ext
        return out
    if ext.shape == (1, n, 4, 4):
        return ext[0]
    if ext.shape == (1, n, 3, 4):
        out = np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
        out[:, :3, :4] = ext[0]
        return out
    if ext.shape == (4, 4):
        return np.repeat(ext[None], n, axis=0)
    print("[da3] unsupported extrinsics format; using identity poses")
    return np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)


def normalize_intrinsics(intrinsics, n: int, width: int, height: int) -> np.ndarray:
    fallback = np.array(
        [[0.9 * width, 0.0, (width - 1) * 0.5], [0.0, 0.9 * width, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if intrinsics is None:
        print("[da3] no intrinsics returned; using fallback intrinsics")
        return np.repeat(fallback[None], n, axis=0)
    ixt = np.asarray(intrinsics, dtype=np.float64)
    print(f"[da3] intrinsics shape={ixt.shape}")
    if ixt.shape == (n, 3, 3):
        return ixt
    if ixt.shape == (1, n, 3, 3):
        return ixt[0]
    if ixt.shape == (3, 3):
        return np.repeat(ixt[None], n, axis=0)
    print("[da3] unsupported intrinsics format; using fallback intrinsics")
    return np.repeat(fallback[None], n, axis=0)


def rgbd_to_pcd(color_rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, max_depth: float):
    h, w = depth.shape[:2]
    color = cv2.resize(color_rgb, (w, h), interpolation=cv2.INTER_AREA if w < color_rgb.shape[1] else cv2.INTER_LINEAR)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color.astype(np.uint8)),
        o3d.geometry.Image(depth.astype(np.float32)),
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False,
    )
    ixt = o3d.camera.PinholeCameraIntrinsic(w, h, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, ixt), rgbd, ixt


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def icp_schedule(args) -> tuple[list[float], list[float]]:
    if args.icp_voxels:
        voxels = parse_float_list(args.icp_voxels)
    elif args.icp_strict:
        voxels = [0.05, 0.025, 0.0125]
    else:
        voxels = [args.icp_voxel]

    if args.icp_distances:
        distances = parse_float_list(args.icp_distances)
    elif args.icp_strict:
        distances = [0.07, 0.035, 0.018]
    else:
        distances = [args.icp_distance]

    if len(distances) == 1 and len(voxels) > 1:
        distances = distances * len(voxels)
    if len(voxels) != len(distances):
        raise ValueError("--icp_voxels and --icp_distances must have the same number of entries")
    return voxels, distances


def prepare_icp_cloud(pcd, voxel: float, args):
    cloud = pcd.voxel_down_sample(voxel)
    if args.remove_outliers and len(cloud.points) > args.outlier_nb_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )
    if len(cloud.points) > 30:
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=40))
    return cloud


def refine_pose_icp(current_pcd, previous_pcd, init_world_current: np.ndarray, previous_world_pcd, args) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
    transform = init_world_current.copy()
    final_result = None
    for voxel, distance in zip(*icp_schedule(args)):
        source = prepare_icp_cloud(current_pcd, voxel, args)
        target = prepare_icp_cloud(previous_world_pcd, voxel, args)
        if len(source.points) < 50 or len(target.points) < 50:
            continue
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7,
            relative_rmse=1e-7,
            max_iteration=80 if args.icp_strict else 40,
        )
        final_result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )
        transform = final_result.transformation

    if final_result is None:
        return init_world_current, current_pcd.transform(init_world_current.copy())

    rejected = False
    if args.icp_max_rmse > 0.0 and final_result.inlier_rmse > args.icp_max_rmse:
        rejected = True
    if args.icp_min_fitness > 0.0 and final_result.fitness < args.icp_min_fitness:
        rejected = True

    if rejected:
        print(
            f"[icp] rejected fitness={final_result.fitness:.3f} rmse={final_result.inlier_rmse:.4f}; "
            "using initial DA3 pose"
        )
        world_current = init_world_current
    else:
        world_current = transform
        print(f"[icp] fitness={final_result.fitness:.3f} rmse={final_result.inlier_rmse:.4f}")
    return world_current, current_pcd.transform(world_current.copy())


def save_trajectory(path: Path, poses: np.ndarray):
    with path.open("w", encoding="utf-8") as handle:
        for idx, pose in enumerate(poses):
            handle.write(f"{idx} " + " ".join(f"{v:.9g}" for v in pose.reshape(-1)) + "\n")


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def run_da3_inference(model, image_paths: list[str], args, out: Path):
    return model.inference(
        image_paths,
        infer_gs=False,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        export_dir=str(out) if args.export_da3_outputs else None,
        export_format=args.export_format,
    )


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    frames_dir = out / "input_frames"
    pointcloud_dir = out / "pointclouds"
    out.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pointclouds:
        pointcloud_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_config(args)
    load_calibration_intrinsics(cfg)
    image_paths = extract_frames(args, frames_dir, cfg)
    if len(image_paths) < 2:
        print("[fatal] Need at least two frames for DA3 pose/reconstruction", file=sys.stderr)
        return 2

    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model).to(device=device).eval()
    print(f"[da3] frames={len(image_paths)} model={args.model}")
    accumulated = o3d.geometry.PointCloud()
    dense_parts = []
    previous_world = None
    refined_poses = []
    volume = None
    if args.tsdf:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=args.tsdf_voxel,
            sdf_trunc=args.tsdf_sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    start = time.perf_counter()
    batch_size = args.da3_batch_size if args.da3_batch_size > 0 else len(image_paths)
    metadata_written = False
    global_idx = 0
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        print(f"[da3] batch {batch_start // batch_size + 1}: frames {batch_start}-{batch_start + len(batch_paths) - 1}")
        pred = run_da3_inference(model, batch_paths, args, out)

        depths = pred.depth.astype(np.float32) * args.depth_scale
        images = pred.processed_images
        if images is None:
            images = np.stack([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in batch_paths], axis=0)
        extrinsics = normalize_extrinsics(pred.extrinsics, len(depths))
        intrinsics = normalize_intrinsics(pred.intrinsics, len(depths), images.shape[2], images.shape[1])

        if not metadata_written:
            with (out / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "is_metric": json_safe(pred.is_metric),
                        "scale_factor": json_safe(pred.scale_factor),
                        "model": args.model,
                        "depth_scale": args.depth_scale,
                        "da3_batch_size": batch_size,
                    },
                    handle,
                    indent=2,
                )
            print(f"[da3] is_metric={pred.is_metric} scale_factor={pred.scale_factor}")
            metadata_written = True

        if args.export_da3_outputs:
            np.save(out / f"depth_batch_{batch_start:05d}.npy", depths)
            np.save(out / f"extrinsics_batch_{batch_start:05d}.npy", extrinsics)
            np.save(out / f"intrinsics_batch_{batch_start:05d}.npy", intrinsics)

        for local_idx, (color_rgb, depth, ext, ixt) in enumerate(zip(images, depths, extrinsics, intrinsics)):
            ixt_depth = scale_intrinsics(ixt, color_rgb.shape[:2], depth.shape[:2])
            pcd, rgbd, intrinsic = rgbd_to_pcd(color_rgb, depth, ixt_depth, args.max_depth)
            world_cam = np.linalg.inv(ext)
            if args.icp_refine and previous_world is not None:
                world_cam, pcd_world = refine_pose_icp(pcd, previous_world, world_cam, previous_world, args)
            else:
                pcd_world = pcd.transform(world_cam.copy())
            previous_world = pcd_world
            refined_poses.append(world_cam.copy())
            if args.dense_cloud:
                dense_parts.append(pcd_world)
            preview_piece = pcd_world if args.preview_voxel <= 0 else pcd_world.voxel_down_sample(args.preview_voxel)
            accumulated += preview_piece
            if args.save_pointclouds:
                pts = np.asarray(pcd_world.points)
                cols = (np.asarray(pcd_world.colors) * 255.0).astype(np.uint8)
                save_pointcloud_ply(pts, cols, str(pointcloud_dir), global_idx)
            if volume is not None:
                volume.integrate(rgbd, intrinsic, ext)
            print(f"[frame {global_idx:04d}] points={len(pcd_world.points)}")
            global_idx += 1

        del pred, depths, images, extrinsics, intrinsics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.preview_voxel > 0:
        accumulated = accumulated.voxel_down_sample(max(args.preview_voxel * 0.5, 0.005))
    o3d.io.write_point_cloud(str(out / "fused_da3_cloud.ply"), accumulated)
    if args.dense_cloud:
        dense = o3d.geometry.PointCloud()
        for part in dense_parts:
            dense += part
        if args.dense_max_points > 0 and len(dense.points) > args.dense_max_points:
            indices = np.random.choice(len(dense.points), args.dense_max_points, replace=False)
            dense = dense.select_by_index(indices.tolist())
        o3d.io.write_point_cloud(str(out / "dense_da3_cloud.ply"), dense)
        print(f"[done] wrote {out / 'dense_da3_cloud.ply'} points={len(dense.points)}")
    save_trajectory(out / "trajectory_da3.txt", np.asarray(refined_poses))
    if volume is not None:
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out / "tsdf_mesh_da3.ply"), mesh)
        o3d.io.write_point_cloud(str(out / "tsdf_cloud_da3.ply"), volume.extract_point_cloud())

    elapsed = time.perf_counter() - start
    print(f"[done] wrote {out / 'fused_da3_cloud.ply'}")
    print(f"[done] wrote {out / 'trajectory_da3.txt'}")
    print(f"[done] elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
