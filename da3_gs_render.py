#!/usr/bin/env python3
"""
Render 3D Gaussian Splatting with Depth Anything 3.

Pipeline:
    video/images -> corrected keyframes -> DA3 infer_gs -> gs_ply + gs_video

This is separate from main.py because DA3 3DGS is a batch multi-view export,
not a per-frame real-time metric-depth operation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

from main import RuntimeConfig, preprocess_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Depth Anything 3 Gaussian Splatting renderer.")
    parser.add_argument("--video", default=None, help="Input video, e.g. ./spectacular-rec/data.mp4")
    parser.add_argument("--images_dir", default=None, help="Directory of input images instead of video.")
    parser.add_argument("--output_dir", default="./da3_gs_out")
    parser.add_argument("--model", default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--process_res_method", default="upper_bound_resize")
    parser.add_argument("--fps_sample", type=float, default=1.0, help="Sample this many frames per second from video.")
    parser.add_argument("--frame_step", type=int, default=0, help="Alternative fixed video frame step. Overrides --fps_sample if > 0.")
    parser.add_argument("--max_frames", type=int, default=24)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--rotate", choices=("none", "90cw", "90ccw", "180"), default="90cw")
    parser.add_argument("--wb", choices=("none", "grayworld", "clahe"), default="grayworld")
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--resolution_scale", type=float, default=1.0, help="Scale extracted keyframes before DA3.")
    parser.add_argument("--export_format", default="gs_ply-gs_video", help="DA3 export format, e.g. gs_ply-gs_video or gs_video.")
    parser.add_argument("--gs_views_interval", type=int, default=1)
    parser.add_argument("--gs_video_quality", choices=("low", "medium", "high"), default="medium")
    parser.add_argument(
        "--gs_trj_mode",
        choices=("original", "smooth", "interpolate", "interpolate_smooth", "wander", "dolly_zoom", "extend", "wobble_inter"),
        default="extend",
    )
    parser.add_argument("--gs_chunk_size", type=int, default=2, help="Lower this if CUDA runs out of memory.")
    parser.add_argument("--render_height", type=int, default=0)
    parser.add_argument("--render_width", type=int, default=0)
    parser.add_argument("--no_depth_vis", action="store_true", help="Render RGB-only video instead of RGB+depth panel.")
    parser.add_argument("--use_ray_pose", action="store_true")
    parser.add_argument("--ref_view_strategy", default="saddle_balanced")
    args = parser.parse_args()
    if not args.video and not args.images_dir:
        parser.error("Provide --video or --images_dir")
    return args


def has_gaussian_branch(model) -> bool:
    candidates = [getattr(model, "model", None)]
    if candidates[0] is not None:
        candidates.extend(
            [
                getattr(candidates[0], "da3", None),
                getattr(candidates[0], "module", None),
            ]
        )
    for candidate in candidates:
        if candidate is None:
            continue
        if getattr(candidate, "gs_head", None) is not None and getattr(candidate, "gs_adapter", None) is not None:
            return True
    return False


def make_preprocess_config(args) -> RuntimeConfig:
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
        max_depth=80.0,
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
        pointcloud_stride=1,
        pointcloud_max_depth=80.0,
        live_registration=False,
        registration_every=1,
        registration_stride=4,
        registration_voxel=0.08,
        registration_feature_radius=0.35,
        registration_ransac_distance=0.16,
        registration_icp_distance=0.05,
        registration_max_points=60000,
        registration_map_voxel=0.04,
        registration_device="cpu",
        registration_backend="open3d_ransac",
        teaser_noise_bound=0.08,
        teaser_max_correspondences=5000,
        teaser_mutual_filter=False,
        calibration=None,
        window_width=1280,
        window_height=720,
        camera_fx=None,
        camera_fy=None,
        camera_cx=None,
        camera_cy=None,
        reconnect_delay=1.0,
        fp16=True,
    )


def extract_video_frames(args, frames_dir: Path) -> list[str]:
    cfg = make_preprocess_config(args)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.frame_step > 0:
        step = args.frame_step
    else:
        step = max(1, int(round(source_fps / max(args.fps_sample, 0.001))))

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    paths: list[str] = []
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


def collect_image_paths(images_dir: Path, max_frames: int) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts]
    if max_frames > 0:
        paths = paths[:max_frames]
    return [str(p) for p in paths]


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    frames_dir = out / "input_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        image_paths = extract_video_frames(args, frames_dir)
    else:
        image_paths = collect_image_paths(Path(args.images_dir), args.max_frames)

    if len(image_paths) < 2:
        print("[fatal] DA3 3DGS needs multiple views; extracted fewer than 2 frames", file=sys.stderr)
        return 2

    print(f"[da3-gs] frames={len(image_paths)}")
    print(f"[da3-gs] output={out}")

    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[da3-gs] CUDA requested but unavailable; using CPU")

    model = DepthAnything3.from_pretrained(args.model).to(device=device).eval()
    if "gs" in args.export_format and not has_gaussian_branch(model):
        print(
            "[fatal] This DA3 model does not expose the 3D Gaussian branch "
            "(gs_head/gs_adapter). Use depth-anything/DA3-GIANT-1.1 for gs_ply/gs_video; "
            "DA3-LARGE-1.1 only produced depth/pose here.",
            file=sys.stderr,
        )
        return 2
    render_hw = None
    if args.render_height > 0 and args.render_width > 0:
        render_hw = (args.render_height, args.render_width)

    export_kwargs = {
        "gs_ply": {"gs_views_interval": args.gs_views_interval},
        "gs_video": {
            "chunk_size": args.gs_chunk_size,
            "trj_mode": args.gs_trj_mode,
            "vis_depth": None if args.no_depth_vis else "hcat",
            "video_quality": args.gs_video_quality,
        },
    }

    start = time.perf_counter()
    model.inference(
        image_paths,
        infer_gs=True,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        render_hw=render_hw,
        export_dir=str(out),
        export_format=args.export_format,
        export_kwargs=export_kwargs,
    )
    elapsed = time.perf_counter() - start
    print(f"[da3-gs] done in {elapsed:.1f}s")
    print(f"[da3-gs] gaussian ply: {out / 'gs_ply'}")
    print(f"[da3-gs] rendered video: {out / 'gs_video'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
