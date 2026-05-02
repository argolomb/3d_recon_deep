#!/usr/bin/env python3
"""
Offline reconstruction pipeline:

Depth Anything -> point clouds -> TEASER++ alignment -> ICP refinement -> TSDF fusion.

This is intentionally offline. It favors stable pose estimation and clean TSDF output
over real-time display.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from main import (
    RuntimeConfig,
    depth_to_pointcloud,
    infer_depth,
    load_calibration_intrinsics,
    load_model,
    preprocess_frame,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline TEASER++/ICP pose estimation and TSDF fusion.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="./fusion_out")
    parser.add_argument("--model_backend", choices=("v2_metric", "v1_zoe", "auto"), default="v2_metric")
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vitl")
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--resolution_scale", type=float, default=0.75)
    parser.add_argument("--rotate", choices=("none", "90cw", "90ccw", "180"), default="90cw")
    parser.add_argument("--wb", choices=("none", "grayworld", "clahe"), default="grayworld")
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--keyframe_step", type=int, default=5)
    parser.add_argument("--pointcloud_stride", type=int, default=2)
    parser.add_argument("--registration_voxel", type=float, default=0.08)
    parser.add_argument("--feature_radius", type=float, default=0.35)
    parser.add_argument("--teaser_noise_bound", type=float, default=0.08)
    parser.add_argument("--teaser_max_correspondences", type=int, default=5000)
    parser.add_argument("--teaser_mutual_filter", action="store_true")
    parser.add_argument("--icp_distance", type=float, default=0.05)
    parser.add_argument("--ransac_fallback", action="store_true")
    parser.add_argument("--tsdf_voxel", type=float, default=0.02)
    parser.add_argument("--tsdf_sdf_trunc", type=float, default=0.08)
    parser.add_argument("--save_keyframes", action="store_true")
    return parser.parse_args()


def make_runtime_config(args) -> RuntimeConfig:
    return RuntimeConfig(
        ip="",
        video=args.video,
        loop_video=False,
        realtime_video=False,
        port=8080,
        stream_path="/video",
        use_gpu=args.use_gpu,
        checkpoint=args.checkpoint,
        repo_path=args.repo_path,
        model_backend=args.model_backend,
        encoder=args.encoder,
        max_depth=args.max_depth,
        input_size=args.input_size,
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
        registration_stride=max(args.pointcloud_stride, 1),
        registration_voxel=max(args.registration_voxel, 0.001),
        registration_feature_radius=max(args.feature_radius, 0.001),
        registration_ransac_distance=max(args.teaser_noise_bound * 2.0, 0.001),
        registration_icp_distance=max(args.icp_distance, 0.001),
        registration_max_points=60000,
        registration_map_voxel=0.04,
        registration_device="cpu",
        registration_backend="teaser",
        teaser_noise_bound=max(args.teaser_noise_bound, 0.001),
        teaser_max_correspondences=max(args.teaser_max_correspondences, 100),
        teaser_mutual_filter=args.teaser_mutual_filter,
        calibration=args.calibration,
        window_width=1280,
        window_height=720,
        camera_fx=None,
        camera_fy=None,
        camera_cx=None,
        camera_cy=None,
        reconnect_delay=1.0,
        fp16=not args.no_fp16,
    )


def scaled_intrinsic(cfg: RuntimeConfig, color_width: int, color_height: int, depth_width: int, depth_height: int):
    fx0 = cfg.camera_fx if cfg.camera_fx is not None else 0.9 * color_width
    fy0 = cfg.camera_fy if cfg.camera_fy is not None else fx0
    cx0 = cfg.camera_cx if cfg.camera_cx is not None else (color_width - 1) * 0.5
    cy0 = cfg.camera_cy if cfg.camera_cy is not None else (color_height - 1) * 0.5
    sx = depth_width / float(color_width)
    sy = depth_height / float(color_height)
    return o3d.camera.PinholeCameraIntrinsic(
        depth_width,
        depth_height,
        fx0 * sx,
        fy0 * sy,
        cx0 * sx,
        cy0 * sy,
    )


class RobustPoseEstimator:
    def __init__(self, args):
        self.args = args
        self.teaserpp = None
        try:
            import teaserpp_python

            self.teaserpp = teaserpp_python
            print("[pose] TEASER++ enabled")
        except Exception as exc:
            if not args.ransac_fallback:
                raise RuntimeError(
                    "TEASER++ Python binding teaserpp_python is not importable in this environment. "
                    "Run inside the venv where TEASER++ was installed, or use --ransac_fallback."
                ) from exc
            print(f"[pose] TEASER++ unavailable ({exc}); using Open3D RANSAC fallback")

        self.prev_down = None
        self.prev_fpfh = None
        self.pose_world_cam = np.eye(4, dtype=np.float64)

    def prepare(self, points: np.ndarray, colors_rgb: np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64) / 255.0)
        down = pcd.voxel_down_sample(self.args.registration_voxel)
        if len(down.points) < 50:
            return pcd, None, None
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.args.registration_voxel * 2.5, max_nn=30)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.args.feature_radius, max_nn=100),
        )
        return pcd, down, fpfh

    def feature_correspondences(self, source_fpfh, target_fpfh) -> Tuple[np.ndarray, np.ndarray]:
        src_feat = np.asarray(source_fpfh.data, dtype=np.float32).T
        tgt_feat = np.asarray(target_fpfh.data, dtype=np.float32).T
        if len(src_feat) == 0 or len(tgt_feat) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        batch = 512
        src_to_tgt = np.empty(len(src_feat), dtype=np.int64)
        src_to_tgt_dist = np.empty(len(src_feat), dtype=np.float32)
        tgt_sq = np.sum(tgt_feat * tgt_feat, axis=1)[None, :]
        for start in range(0, len(src_feat), batch):
            block = src_feat[start : start + batch]
            dist = np.sum(block * block, axis=1)[:, None] + tgt_sq - 2.0 * block @ tgt_feat.T
            src_to_tgt[start : start + len(block)] = np.argmin(dist, axis=1)
            src_to_tgt_dist[start : start + len(block)] = np.min(dist, axis=1)

        src_idx = np.arange(len(src_feat), dtype=np.int64)
        tgt_idx = src_to_tgt
        if self.args.teaser_mutual_filter:
            tgt_to_src = np.empty(len(tgt_feat), dtype=np.int64)
            src_sq = np.sum(src_feat * src_feat, axis=1)[None, :]
            for start in range(0, len(tgt_feat), batch):
                block = tgt_feat[start : start + batch]
                dist = np.sum(block * block, axis=1)[:, None] + src_sq - 2.0 * block @ src_feat.T
                tgt_to_src[start : start + len(block)] = np.argmin(dist, axis=1)
            keep = tgt_to_src[tgt_idx] == src_idx
            src_idx = src_idx[keep]
            tgt_idx = tgt_idx[keep]
            src_to_tgt_dist = src_to_tgt_dist[keep]

        if len(src_idx) > self.args.teaser_max_correspondences:
            keep = np.argsort(src_to_tgt_dist)[: self.args.teaser_max_correspondences]
            src_idx = src_idx[keep]
            tgt_idx = tgt_idx[keep]
        return src_idx, tgt_idx

    def teaser_transform(self, source_down, target_down, source_fpfh, target_fpfh) -> np.ndarray:
        src_idx, tgt_idx = self.feature_correspondences(source_fpfh, target_fpfh)
        if len(src_idx) < 6:
            raise RuntimeError(f"too few TEASER++ correspondences: {len(src_idx)}")

        src_pts = np.asarray(source_down.points)[src_idx].T.astype(np.float64)
        dst_pts = np.asarray(target_down.points)[tgt_idx].T.astype(np.float64)
        params = self.teaserpp.RobustRegistrationSolver.Params()
        params.cbar2 = 1.0
        params.noise_bound = self.args.teaser_noise_bound
        params.estimate_scaling = False
        params.rotation_estimation_algorithm = (
            self.teaserpp.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        params.rotation_gnc_factor = 1.4
        params.rotation_max_iterations = 100
        params.rotation_cost_threshold = 1e-12
        solver = self.teaserpp.RobustRegistrationSolver(params)
        solver.solve(src_pts, dst_pts)
        solution = solver.getSolution()

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(solution.rotation, dtype=np.float64)
        transform[:3, 3] = np.asarray(solution.translation, dtype=np.float64).reshape(3)
        print(f"[pose] TEASER++ correspondences={len(src_idx)}")
        return transform

    def ransac_transform(self, source_down, target_down, source_fpfh, target_fpfh) -> np.ndarray:
        reg = o3d.pipelines.registration
        result = reg.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.args.teaser_noise_bound * 2.0,
            estimation_method=reg.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            criteria=reg.RANSACConvergenceCriteria(400000, 500),
        )
        return result.transformation

    def estimate(self, points: np.ndarray, colors_rgb: np.ndarray) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        pcd, down, fpfh = self.prepare(points, colors_rgb)
        if down is None:
            raise RuntimeError("keyframe too sparse after voxel downsample")

        if self.prev_down is None:
            self.prev_down = down
            self.prev_fpfh = fpfh
            return self.pose_world_cam.copy(), pcd

        if self.teaserpp is not None:
            init = self.teaser_transform(down, self.prev_down, fpfh, self.prev_fpfh)
        else:
            init = self.ransac_transform(down, self.prev_down, fpfh, self.prev_fpfh)

        reg = o3d.pipelines.registration
        icp = reg.registration_icp(
            down,
            self.prev_down,
            self.args.icp_distance,
            init,
            reg.TransformationEstimationPointToPlane(),
        )
        t_world_cur = icp.transformation
        self.pose_world_cam = t_world_cur
        self.prev_down = down.transform(t_world_cur)
        self.prev_fpfh = fpfh
        print(f"[pose] ICP fitness={icp.fitness:.3f} rmse={icp.inlier_rmse:.4f}")
        return self.pose_world_cam.copy(), pcd


def make_rgbd(frame_bgr: np.ndarray, depth_m: np.ndarray, cfg: RuntimeConfig):
    h, w = depth_m.shape[:2]
    color_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA if w < frame_bgr.shape[1] else cv2.INTER_LINEAR)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb.astype(np.uint8)),
        o3d.geometry.Image(depth_m.astype(np.float32)),
        depth_scale=1.0,
        depth_trunc=cfg.max_depth,
        convert_rgb_to_intensity=False,
    )
    intrinsic = scaled_intrinsic(cfg, frame_bgr.shape[1], frame_bgr.shape[0], w, h)
    return rgbd, intrinsic


def save_trajectory(path: Path, poses):
    with path.open("w", encoding="utf-8") as handle:
        for index, pose in poses:
            flat = " ".join(f"{v:.9g}" for v in pose.reshape(-1))
            handle.write(f"{index} {flat}\n")


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = make_runtime_config(args)
    load_calibration_intrinsics(cfg)
    model = load_model(cfg)
    pose_estimator = RobustPoseEstimator(args)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.tsdf_voxel,
        sdf_trunc=args.tsdf_sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[fatal] could not open video: {args.video}", file=sys.stderr)
        return 2
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    poses = []
    processed = 0
    keyframes = 0
    frame_index = args.start_frame
    start = time.perf_counter()

    try:
        while True:
            ok, raw = cap.read()
            if not ok or raw is None:
                break
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            if (frame_index - args.start_frame) % args.keyframe_step != 0:
                frame_index += 1
                processed += 1
                continue

            frame = preprocess_frame(raw, cfg)
            depth = infer_depth(model, frame, cfg.input_size, output_scale=1.0)
            points, colors = depth_to_pointcloud(frame, depth, cfg, stride=cfg.pointcloud_stride)
            if len(points) < 100:
                print(f"[frame {frame_index}] skipped sparse point cloud")
                frame_index += 1
                processed += 1
                continue

            pose_world_cam, pcd = pose_estimator.estimate(points, colors)
            rgbd, intrinsic = make_rgbd(frame, depth, cfg)
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose_world_cam))
            poses.append((frame_index, pose_world_cam.copy()))
            keyframes += 1

            if args.save_keyframes:
                o3d.io.write_point_cloud(str(out / f"keyframe_{keyframes:05d}.ply"), pcd)
            print(f"[frame {frame_index}] integrated keyframe={keyframes}")

            frame_index += 1
            processed += 1
    finally:
        cap.release()

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cloud = volume.extract_point_cloud()
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(out / "fused_cloud.ply"), cloud)
    save_trajectory(out / "trajectory.txt", poses)
    elapsed = time.perf_counter() - start
    print(f"[done] keyframes={keyframes} elapsed={elapsed:.1f}s")
    print(f"[done] wrote {out / 'mesh.ply'}")
    print(f"[done] wrote {out / 'fused_cloud.ply'}")
    print(f"[done] wrote {out / 'trajectory.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
