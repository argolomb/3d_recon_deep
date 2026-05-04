#!/usr/bin/env python3
"""
Real-time metric depth from Android IP Webcam using Depth Anything.

Expected input stream:
    http://<IP>:8080/video

Recommended setup for Depth Anything V1 metric checkpoints:
    git clone https://github.com/LiheYoung/Depth-Anything
    cd Depth-Anything/metric_depth
    # install the official repo requirements/environment
    # download one checkpoint from:
    # https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth

Run:
    python main.py --ip 192.168.0.10 --repo_path ./Depth-Anything/metric_depth \
        --checkpoint ./models/depth_anything_metric_depth_indoor.pt

Notes:
    - Metric checkpoints output depth in meters.
    - Use indoor checkpoint for rooms and outdoor checkpoint for street-scale scenes.
    - Saving writes RGB PNG plus uint16 depth PNG in millimeters, suitable for SLAM logs.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import socket
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RuntimeConfig:
    ip: str
    video: Optional[str]
    loop_video: bool
    realtime_video: bool
    port: int
    stream_path: str
    ws_media_format: str
    ws_mpeg_format: str
    use_gpu: bool
    checkpoint: Optional[str]
    repo_path: Optional[str]
    model_backend: str
    encoder: str
    max_depth: float
    input_size: int
    resolution_scale: float
    rotate: str
    wb: str
    brightness: float
    contrast: float
    saturation: float
    gamma: float
    clahe_clip: float
    clahe_tile: int
    sharpen: float
    denoise: int
    width: int
    height: int
    frame_skip: int
    fps_limit: float
    display_scale: float
    display_every: int
    print_every: int
    fast_vis: bool
    profile_interval: float
    torch_compile: bool
    cuda_empty_cache_every: int
    da3_process_res_method: str
    da3_metric_focal_scale: bool
    da3_metric_scale: float
    low_latency: bool
    capture_fps: float
    depth_output_scale: float
    pointcloud: bool
    save_dir: Optional[str]
    save_every: int
    pointcloud_save_dir: Optional[str]
    pointcloud_save_every: int
    pointcloud_stride: int
    pointcloud_max_depth: float
    pointcloud_compute_device: str
    pointcloud_web: bool
    pointcloud_web_host: str
    pointcloud_web_port: int
    pointcloud_web_every: int
    pointcloud_web_stride: int
    pointcloud_web_max_points: int
    dataset_save_dir: Optional[str]
    dataset_save_every: int
    dataset_save_stride: int
    dataset_save_max_depth: float
    live_registration: bool
    registration_every: int
    registration_stride: int
    registration_depth_scale: float
    registration_voxel: float
    registration_feature_radius: float
    registration_ransac_distance: float
    registration_icp_distance: float
    registration_icp_fine_distance: float
    registration_min_fitness: float
    registration_max_rmse: float
    registration_max_points: int
    registration_max_map_points: int
    registration_map_voxel: float
    registration_device: str
    registration_backend: str
    registration_pose_icp_refine: bool
    teaser_noise_bound: float
    teaser_max_correspondences: int
    teaser_mutual_filter: bool
    calibration: Optional[str]
    window_width: int
    window_height: int
    camera_fx: Optional[float]
    camera_fy: Optional[float]
    camera_cx: Optional[float]
    camera_cy: Optional[float]
    reconnect_delay: float
    fp16: bool


class FPSMeter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.last = time.perf_counter()
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        dt = max(now - self.last, 1e-6)
        instant = 1.0 / dt
        self.fps = instant if self.fps == 0.0 else self.smoothing * self.fps + (1.0 - self.smoothing) * instant
        self.last = now
        return self.fps


def load_simple_yaml(path: str) -> dict:
    config_path = Path(path).expanduser()
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping")
        return data
    except ModuleNotFoundError:
        data = {}
        with config_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value.lower() in ("true", "yes", "on"):
                    data[key] = True
                elif value.lower() in ("false", "no", "off"):
                    data[key] = False
                elif value.lower() in ("null", "none", "~", ""):
                    data[key] = None
                else:
                    try:
                        data[key] = int(value)
                    except ValueError:
                        try:
                            data[key] = float(value)
                        except ValueError:
                            data[key] = value.strip("\"'")
        return data


def add_config_argument(parser: argparse.ArgumentParser) -> Optional[str]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="YAML config file. CLI arguments override YAML values.")
    known, _ = config_parser.parse_known_args()
    parser.add_argument("--config", default=known.config, help="YAML config file. CLI arguments override YAML values.")
    return known.config


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Real-time metric depth with Depth Anything and Android IP Webcam.")
    config_path = add_config_argument(parser)
    parser.add_argument("--ip", default=None, help="Android IP Webcam address or WebSocket URL, for example 192.168.0.10 or ws://host:port/path")
    parser.add_argument("--video", default=None, help="Local video file. Example: ./spectacular-rec/data.mp4")
    parser.add_argument("--loop_video", action="store_true", help="Loop local video at EOF.")
    parser.add_argument("--realtime_video", action="store_true", help="Throttle local video playback to the file FPS.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--stream_path", default="/video")
    parser.add_argument("--ws_media_format", choices=("image", "mpeg1video"), default="image", help="WebSocket payload format. image expects JPEG/PNG/MP4 messages; mpeg1video expects a continuous MPEG-1 stream.")
    parser.add_argument("--ws_mpeg_format", choices=("auto", "mpegts", "mpegvideo"), default="mpegvideo", help="Container/bitstream hint for --ws_media_format mpeg1video. Use mpegvideo for raw MPEG-1 video, mpegts for JSMpeg/ffmpeg MPEG-TS.")
    parser.add_argument("--use_gpu", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--checkpoint", default=None, help="Metric checkpoint path, local model dir, or Hugging Face model id for --model_backend da3.")
    parser.add_argument("--repo_path", default=None, help="Path to official Depth-Anything metric_depth repo or Depth-Anything-V2 repo.")
    parser.add_argument(
        "--model_backend",
        choices=("auto", "v1_zoe", "v2_metric", "da3"),
        default="auto",
        help="auto tries V2 metric first if available, then V1 ZoeDepth metric. da3 is experimental relative depth.",
    )
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vitl")
    parser.add_argument("--max_depth", type=float, default=20.0, help="Meters. Use 20 indoor, 80 outdoor.")
    parser.add_argument("--input_size", type=int, default=518, help="Network inference size, commonly 384 or 518.")
    parser.add_argument("--resolution_scale", type=float, default=1.0, help="Scale captured frames before inference/display.")
    parser.add_argument("--rotate", choices=("none", "90cw", "90ccw", "180"), default="none", help="Rotate frames before inference.")
    parser.add_argument("--wb", choices=("none", "grayworld", "clahe"), default="none", help="Simple color correction / white balance.")
    parser.add_argument("--brightness", type=float, default=0.0, help="Additive brightness adjustment after rotation.")
    parser.add_argument("--contrast", type=float, default=1.0, help="Multiplicative contrast adjustment after rotation.")
    parser.add_argument("--saturation", type=float, default=1.0, help="HSV saturation multiplier.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction. Values below 1 brighten shadows.")
    parser.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clip limit when --wb clahe.")
    parser.add_argument("--clahe_tile", type=int, default=8, help="CLAHE tile grid size when --wb clahe.")
    parser.add_argument("--sharpen", type=float, default=0.0, help="Unsharp mask amount. Try 0.3 to 0.8.")
    parser.add_argument("--denoise", type=int, default=0, help="Median blur kernel size before contrast, odd values like 3 or 5.")
    parser.add_argument("--width", type=int, default=0, help="Requested stream width; 0 keeps camera default.")
    parser.add_argument("--height", type=int, default=0, help="Requested stream height; 0 keeps camera default.")
    parser.add_argument("--frame_skip", type=int, default=0, help="Run inference every N+1 frames; reuse previous depth between runs.")
    parser.add_argument("--fps_limit", type=float, default=30.0)
    parser.add_argument("--display_scale", type=float, default=1.0, help="Scale only the OpenCV visualization. Values like 0.5 reduce CPU display cost.")
    parser.add_argument("--display_every", type=int, default=1, help="Render the OpenCV window every N frames.")
    parser.add_argument("--print_every", type=int, default=15, help="Print FPS every N frames instead of every frame.")
    parser.add_argument("--fast_vis", action="store_true", help="Use fixed max-depth visualization and skip per-frame percentile/median stats.")
    parser.add_argument("--profile_interval", type=float, default=0.0, help="Print rough capture/inference/visualization timings every N seconds.")
    parser.add_argument("--torch_compile", action="store_true", help="Try torch.compile(model). First frames are slower while compiling.")
    parser.add_argument("--cuda_empty_cache_every", type=int, default=0, help="Call torch.cuda.empty_cache() every N inferences. 0 disables.")
    parser.add_argument("--da3_process_res_method", default="upper_bound_resize", help="DA3 process_res_method passed to model.inference.")
    parser.add_argument("--da3_metric_focal_scale", action="store_true", help="For DA3METRIC-LARGE: multiply canonical depth by focal length in pixels.")
    parser.add_argument("--da3_metric_scale", type=float, default=1.0, help="Additional scale multiplier for DA3 depth. Use to calibrate/diagnose metric output.")
    parser.add_argument("--low_latency", action="store_true", help="Read camera frames in a background thread and always process the newest frame.")
    parser.add_argument("--capture_fps", type=float, default=0.0, help="Optional sleep limit for threaded capture. 0 reads as fast as possible.")
    parser.add_argument("--depth_output_scale", type=float, default=1.0, help="Scale metric depth output before CPU transfer. 0.5 is faster but saves lower-res depth.")
    parser.add_argument("--pointcloud", action="store_true", help="Open optional Open3D point-cloud viewer.")
    parser.add_argument("--save_dir", default=None, help="If set, save synchronized RGB and uint16 depth PNG files.")
    parser.add_argument("--save_every", type=int, default=0, help="Save every N inferred frames. 0 disables automatic saving; press s to save.")
    parser.add_argument("--pointcloud_save_dir", default=None, help="If set, save dense colored PLY point clouds. Press p to save current point cloud.")
    parser.add_argument("--pointcloud_save_every", type=int, default=0, help="Save one PLY every N inferred frames. 0 disables automatic point-cloud saving.")
    parser.add_argument("--pointcloud_stride", type=int, default=1, help="Point cloud pixel stride. 1 is dense, 2/4 are faster and smaller.")
    parser.add_argument("--pointcloud_max_depth", type=float, default=0.0, help="Optional max depth filter in meters for point clouds. 0 uses --max_depth.")
    parser.add_argument("--pointcloud_compute_device", choices=("auto", "cpu", "cuda"), default="auto", help="Compute depth-to-pointcloud projection on CUDA when possible.")
    parser.add_argument("--pointcloud_web", action="store_true", help="Stream live point clouds to browser clients over WebSocket.")
    parser.add_argument("--pointcloud_web_host", default="0.0.0.0", help="Host/interface for the point-cloud WebSocket server.")
    parser.add_argument("--pointcloud_web_port", type=int, default=8765, help="Port for the point-cloud WebSocket server.")
    parser.add_argument("--pointcloud_web_every", type=int, default=1, help="Publish one point cloud every N inferred frames.")
    parser.add_argument("--pointcloud_web_stride", type=int, default=4, help="Point-cloud pixel stride used for browser streaming.")
    parser.add_argument("--pointcloud_web_max_points", type=int, default=120000, help="Max points per browser frame. 0 disables the cap.")
    parser.add_argument("--dataset_save_dir", default=None, help="Save synchronized RGB, depth_mm, partial PLY and metadata for offline reconstruction.")
    parser.add_argument("--dataset_save_every", type=int, default=0, help="Save dataset sample every N inferred frames. 0 disables automatic saving; press d to save.")
    parser.add_argument("--dataset_save_stride", type=int, default=2, help="Point-cloud stride for dataset partial PLY.")
    parser.add_argument("--dataset_save_max_depth", type=float, default=0.0, help="Optional max depth filter for dataset partial PLY. 0 uses pointcloud/max_depth.")
    parser.add_argument("--live_registration", action="store_true", help="Open a live preview map using Open3D RANSAC+ICP registration.")
    parser.add_argument("--registration_every", type=int, default=10, help="Submit one keyframe to live registration every N inferred frames.")
    parser.add_argument("--registration_stride", type=int, default=4, help="Pixel stride for live registration input. Higher is faster.")
    parser.add_argument("--registration_depth_scale", type=float, default=1.0, help="Resize depth only for live-registration point clouds. >1 upsamples for denser preview, <1 reduces memory.")
    parser.add_argument("--registration_voxel", type=float, default=0.08, help="Voxel size in meters before FPFH/RANSAC.")
    parser.add_argument("--registration_feature_radius", type=float, default=0.35, help="FPFH search radius in meters.")
    parser.add_argument("--registration_ransac_distance", type=float, default=0.16, help="RANSAC max correspondence distance in meters.")
    parser.add_argument("--registration_icp_distance", type=float, default=0.05, help="Coarse ICP max correspondence distance in meters.")
    parser.add_argument("--registration_icp_fine_distance", type=float, default=0.02, help="Fine ICP max correspondence distance in meters for robust backend.")
    parser.add_argument("--registration_min_fitness", type=float, default=0.05, help="Reject live registration if ICP fitness is below this. 0 disables.")
    parser.add_argument("--registration_max_rmse", type=float, default=0.0, help="Reject live registration if ICP RMSE is above this. 0 disables.")
    parser.add_argument("--registration_max_points", type=int, default=60000, help="Max raw points per submitted registration keyframe.")
    parser.add_argument("--registration_max_map_points", type=int, default=300000, help="Max points retained in live registration preview map. 0 disables cap.")
    parser.add_argument("--registration_map_voxel", type=float, default=0.04, help="Voxel size for accumulated preview map.")
    parser.add_argument("--registration_device", choices=("auto", "cpu", "cuda"), default="auto", help="Requested Open3D device for preview where supported.")
    parser.add_argument("--registration_backend", choices=("open3d_ransac", "teaser", "robust", "da3_pose"), default="open3d_ransac", help="Global registration backend for live preview.")
    parser.add_argument("--registration_pose_icp_refine", action="store_true", help="Refine DA3 pose backend with ICP.")
    parser.add_argument("--teaser_noise_bound", type=float, default=0.08, help="TEASER++ noise bound in meters.")
    parser.add_argument("--teaser_max_correspondences", type=int, default=5000, help="Max FPFH correspondences sent to TEASER++.")
    parser.add_argument("--teaser_mutual_filter", action="store_true", help="Keep only mutual nearest FPFH correspondences for TEASER++.")
    parser.add_argument("--calibration", default=None, help="Camera calibration JSON. If omitted with --video, tries <video_dir>/calibration.json.")
    parser.add_argument("--window_width", type=int, default=1280, help="Initial display window width; the window remains manually resizable.")
    parser.add_argument("--window_height", type=int, default=720, help="Initial display window height; the window remains manually resizable.")
    parser.add_argument("--camera_fx", type=float, default=None)
    parser.add_argument("--camera_fy", type=float, default=None)
    parser.add_argument("--camera_cx", type=float, default=None)
    parser.add_argument("--camera_cy", type=float, default=None)
    parser.add_argument("--reconnect_delay", type=float, default=1.0)
    parser.add_argument("--no_fp16", action="store_true", help="Disable CUDA fp16 inference.")
    if config_path:
        config = load_simple_yaml(config_path)
        valid_dests = {action.dest for action in parser._actions}
        unknown = sorted(key for key in config if key not in valid_dests)
        if unknown:
            parser.error(f"Unknown config key(s) in {config_path}: {', '.join(unknown)}")
        parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.ip and not args.video:
        parser.error("Provide --ip for IP Webcam or --video for a local recording.")

    return RuntimeConfig(
        ip=args.ip or "",
        video=args.video,
        loop_video=args.loop_video,
        realtime_video=args.realtime_video,
        port=args.port,
        stream_path=args.stream_path,
        ws_media_format=args.ws_media_format,
        ws_mpeg_format=args.ws_mpeg_format,
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
        clahe_clip=max(args.clahe_clip, 0.1),
        clahe_tile=max(args.clahe_tile, 2),
        sharpen=max(args.sharpen, 0.0),
        denoise=max(args.denoise, 0),
        width=args.width,
        height=args.height,
        frame_skip=max(args.frame_skip, 0),
        fps_limit=max(args.fps_limit, 0.0),
        display_scale=max(args.display_scale, 0.1),
        display_every=max(args.display_every, 1),
        print_every=max(args.print_every, 1),
        fast_vis=args.fast_vis,
        profile_interval=max(args.profile_interval, 0.0),
        torch_compile=args.torch_compile,
        cuda_empty_cache_every=max(args.cuda_empty_cache_every, 0),
        da3_process_res_method=args.da3_process_res_method,
        da3_metric_focal_scale=args.da3_metric_focal_scale,
        da3_metric_scale=args.da3_metric_scale,
        low_latency=args.low_latency,
        capture_fps=max(args.capture_fps, 0.0),
        depth_output_scale=min(max(args.depth_output_scale, 0.1), 1.0),
        pointcloud=args.pointcloud,
        save_dir=args.save_dir,
        save_every=max(args.save_every, 0),
        pointcloud_save_dir=args.pointcloud_save_dir,
        pointcloud_save_every=max(args.pointcloud_save_every, 0),
        pointcloud_stride=max(args.pointcloud_stride, 1),
        pointcloud_max_depth=max(args.pointcloud_max_depth, 0.0),
        pointcloud_compute_device=args.pointcloud_compute_device,
        pointcloud_web=args.pointcloud_web,
        pointcloud_web_host=args.pointcloud_web_host,
        pointcloud_web_port=max(args.pointcloud_web_port, 1),
        pointcloud_web_every=max(args.pointcloud_web_every, 1),
        pointcloud_web_stride=max(args.pointcloud_web_stride, 1),
        pointcloud_web_max_points=max(args.pointcloud_web_max_points, 0),
        dataset_save_dir=args.dataset_save_dir,
        dataset_save_every=max(args.dataset_save_every, 0),
        dataset_save_stride=max(args.dataset_save_stride, 1),
        dataset_save_max_depth=max(args.dataset_save_max_depth, 0.0),
        live_registration=args.live_registration,
        registration_every=max(args.registration_every, 1),
        registration_stride=max(args.registration_stride, 1),
        registration_depth_scale=max(args.registration_depth_scale, 0.1),
        registration_voxel=max(args.registration_voxel, 0.001),
        registration_feature_radius=max(args.registration_feature_radius, 0.001),
        registration_ransac_distance=max(args.registration_ransac_distance, 0.001),
        registration_icp_distance=max(args.registration_icp_distance, 0.001),
        registration_icp_fine_distance=max(args.registration_icp_fine_distance, 0.001),
        registration_min_fitness=max(args.registration_min_fitness, 0.0),
        registration_max_rmse=max(args.registration_max_rmse, 0.0),
        registration_max_points=max(args.registration_max_points, 1000),
        registration_max_map_points=max(args.registration_max_map_points, 0),
        registration_map_voxel=max(args.registration_map_voxel, 0.001),
        registration_device=args.registration_device,
        registration_backend=args.registration_backend,
        registration_pose_icp_refine=args.registration_pose_icp_refine,
        teaser_noise_bound=max(args.teaser_noise_bound, 0.001),
        teaser_max_correspondences=max(args.teaser_max_correspondences, 100),
        teaser_mutual_filter=args.teaser_mutual_filter,
        calibration=args.calibration,
        window_width=max(args.window_width, 320),
        window_height=max(args.window_height, 240),
        camera_fx=args.camera_fx,
        camera_fy=args.camera_fy,
        camera_cx=args.camera_cx,
        camera_cy=args.camera_cy,
        reconnect_delay=max(args.reconnect_delay, 0.1),
        fp16=not args.no_fp16,
    )


def stream_url(cfg: RuntimeConfig) -> str:
    if cfg.ip.startswith(("http://", "https://", "ws://", "wss://")):
        return cfg.ip
    if "/" in cfg.ip:
        return f"http://{cfg.ip}"
    if ":" in cfg.ip:
        path = cfg.stream_path if cfg.stream_path.startswith("/") else f"/{cfg.stream_path}"
        return f"http://{cfg.ip}{path}"
    path = cfg.stream_path if cfg.stream_path.startswith("/") else f"/{cfg.stream_path}"
    return f"http://{cfg.ip}:{cfg.port}{path}"


def is_websocket_source(cfg: RuntimeConfig) -> bool:
    return bool(not cfg.video and stream_url(cfg).startswith(("ws://", "wss://")))


def open_stream(cfg: RuntimeConfig) -> cv2.VideoCapture:
    source = cfg.video if cfg.video else stream_url(cfg)
    if source.startswith(("ws://", "wss://")):
        raise ValueError("WebSocket sources are handled by WebSocketFrameStream, not cv2.VideoCapture")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if cfg.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    if cfg.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def correct_colors(frame_bgr: np.ndarray, cfg: RuntimeConfig) -> np.ndarray:
    if cfg.denoise >= 3:
        kernel = cfg.denoise if cfg.denoise % 2 == 1 else cfg.denoise + 1
        frame_bgr = cv2.medianBlur(frame_bgr, kernel)

    if cfg.wb == "grayworld":
        image = frame_bgr.astype(np.float32)
        means = image.reshape(-1, 3).mean(axis=0)
        gray = float(means.mean())
        gains = gray / np.maximum(means, 1.0)
        frame_bgr = np.clip(image * gains, 0, 255).astype(np.uint8)
    elif cfg.wb == "clahe":
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
        l_chan = clahe.apply(l_chan)
        frame_bgr = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    if cfg.saturation != 1.0:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * cfg.saturation, 0, 255)
        frame_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if cfg.contrast != 1.0 or cfg.brightness != 0.0:
        frame_bgr = cv2.convertScaleAbs(frame_bgr, alpha=cfg.contrast, beta=cfg.brightness)

    if cfg.gamma != 1.0:
        inv_gamma = 1.0 / cfg.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        frame_bgr = cv2.LUT(frame_bgr, table)

    if cfg.sharpen > 0.0:
        blur = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=1.2)
        frame_bgr = cv2.addWeighted(frame_bgr, 1.0 + cfg.sharpen, blur, -cfg.sharpen, 0)

    return frame_bgr


def preprocess_frame(frame_bgr: np.ndarray, cfg: RuntimeConfig) -> np.ndarray:
    if cfg.rotate == "90cw":
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif cfg.rotate == "90ccw":
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif cfg.rotate == "180":
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)

    if cfg.resolution_scale != 1.0:
        frame_bgr = cv2.resize(
            frame_bgr,
            None,
            fx=cfg.resolution_scale,
            fy=cfg.resolution_scale,
            interpolation=cv2.INTER_AREA if cfg.resolution_scale < 1.0 else cv2.INTER_LINEAR,
        )
    return correct_colors(frame_bgr, cfg)


def get_frame(cap: cv2.VideoCapture, cfg: RuntimeConfig) -> Tuple[Optional[np.ndarray], cv2.VideoCapture]:
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        if cfg.video and cfg.loop_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                return preprocess_frame(frame_bgr, cfg), cap
        cap.release()
        if cfg.video:
            return None, cap
        print(f"[stream] connection lost; reconnecting in {cfg.reconnect_delay:.1f}s")
        time.sleep(cfg.reconnect_delay)
        return None, open_stream(cfg)
    return preprocess_frame(frame_bgr, cfg), cap


class LatestFrameStream:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.cap = open_stream(cfg)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        min_dt = 1.0 / self.cfg.capture_fps if self.cfg.capture_fps > 0 else 0.0
        while not self.stopped:
            start = time.perf_counter()
            ok, frame_bgr = self.cap.read()
            if not ok or frame_bgr is None:
                self.cap.release()
                print(f"[stream] connection lost; reconnecting in {self.cfg.reconnect_delay:.1f}s")
                time.sleep(self.cfg.reconnect_delay)
                self.cap = open_stream(self.cfg)
                continue

            frame_bgr = preprocess_frame(frame_bgr, self.cfg)

            with self.lock:
                self.frame = frame_bgr

            if min_dt > 0:
                elapsed = time.perf_counter() - start
                if elapsed < min_dt:
                    time.sleep(min_dt - elapsed)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def release(self) -> None:
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


class WebSocketImageClient:
    """Minimal RFC6455 media receiver for binary JPEG/PNG/MP4 or base64 text frames."""

    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in ("ws", "wss"):
            raise ValueError(f"Unsupported WebSocket scheme: {parsed.scheme}")
        host = parsed.hostname
        if not host:
            raise ValueError(f"Invalid WebSocket URL: {self.url}")
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        raw_sock = socket.create_connection((host, port), timeout=self.timeout)
        if parsed.scheme == "wss":
            raw_sock = ssl.create_default_context().wrap_socket(raw_sock, server_hostname=host)
        raw_sock.settimeout(self.timeout)

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        host_header = host if parsed.port is None else f"{host}:{port}"
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        raw_sock.sendall(request.encode("ascii"))
        response = self._recv_until(raw_sock, b"\r\n\r\n", max_bytes=16384).decode("iso-8859-1", errors="replace")
        status = response.split("\r\n", 1)[0]
        if " 101 " not in f" {status} ":
            raw_sock.close()
            raise ConnectionError(f"WebSocket handshake failed: {status}")

        headers = {}
        for line in response.split("\r\n")[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        expected = base64.b64encode(hashlib.sha1((key + self.GUID).encode("ascii")).digest()).decode("ascii")
        if headers.get("sec-websocket-accept") != expected:
            raw_sock.close()
            raise ConnectionError("WebSocket handshake failed: invalid Sec-WebSocket-Accept")
        self.sock = raw_sock

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    @staticmethod
    def _recv_until(sock_obj: socket.socket, marker: bytes, max_bytes: int) -> bytes:
        data = bytearray()
        while marker not in data:
            chunk = sock_obj.recv(1)
            if not chunk:
                raise ConnectionError("socket closed during handshake")
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ConnectionError("handshake response too large")
        return bytes(data)

    def _recv_exact(self, size: int) -> bytes:
        if self.sock is None:
            raise ConnectionError("WebSocket is not connected")
        data = bytearray()
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("WebSocket closed")
            data.extend(chunk)
        return bytes(data)

    def _send_control(self, opcode: int, payload: bytes = b"") -> None:
        if self.sock is None:
            return
        payload = payload[:125]
        mask = os.urandom(4)
        header = bytes((0x80 | opcode, 0x80 | len(payload)))
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def recv_message(self) -> Tuple[bytes, int]:
        fragments = []
        first_opcode = None
        while True:
            header = self._recv_exact(2)
            first_byte, second_byte = header
            fin = bool(first_byte & 0x80)
            opcode = first_byte & 0x0F
            masked = bool(second_byte & 0x80)
            payload_len = second_byte & 0x7F
            if payload_len == 126:
                payload_len = struct.unpack("!H", self._recv_exact(2))[0]
            elif payload_len == 127:
                payload_len = struct.unpack("!Q", self._recv_exact(8))[0]

            mask_key = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(payload_len) if payload_len else b""
            if masked:
                payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))

            if opcode == 8:
                raise ConnectionError("WebSocket close frame received")
            if opcode == 9:
                self._send_control(10, payload)
                continue
            if opcode == 10:
                continue
            if opcode in (1, 2):
                first_opcode = opcode
                fragments = [payload]
            elif opcode == 0 and first_opcode is not None:
                fragments.append(payload)
            else:
                continue

            if fin:
                return b"".join(fragments), first_opcode

    def recv_image(self) -> np.ndarray:
        payload, opcode = self.recv_message()
        return decode_websocket_image(payload, opcode)


def decode_websocket_text_payload(payload: bytes) -> bytes:
    text = payload.decode("utf-8", errors="ignore").strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
            for key in ("image", "frame", "data", "video"):
                if key in data:
                    text = str(data[key]).strip()
                    break
        except json.JSONDecodeError:
            pass
    if "," in text and text.lower().startswith(("data:image", "data:video")):
        text = text.split(",", 1)[1]
    return base64.b64decode(text, validate=False)


def decode_websocket_image(payload: bytes, opcode: int) -> np.ndarray:
    if opcode == 1:
        payload = decode_websocket_text_payload(payload)

    image_bytes = np.frombuffer(payload, dtype=np.uint8)
    frame_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is not None:
        return frame_bgr
    
    # Try to decode as MP4 video
    if len(payload) >= 8 and payload[4:8] == b"ftyp":
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(payload)
            tmp_path = tmp_file.name
        try:
            cap = cv2.VideoCapture(tmp_path, cv2.CAP_FFMPEG)
            ret, frame_bgr = cap.read()
            cap.release()
            os.unlink(tmp_path)
            if ret and frame_bgr is not None:
                return frame_bgr
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    
    raise ValueError("WebSocket payload is not a valid JPEG/PNG/MP4 media")


class FfmpegMpeg1Decoder:
    def __init__(self, cfg: RuntimeConfig, on_frame) -> None:
        self.cfg = cfg
        self.on_frame = on_frame
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.stopped = False

    def start(self) -> None:
        if self.proc is not None:
            return
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-flags",
            "low_delay",
            "-probesize",
            "32768",
            "-analyzeduration",
            "0",
        ]
        if self.cfg.ws_mpeg_format != "auto":
            command.extend(["-f", self.cfg.ws_mpeg_format])
        command.extend(
            [
                "-i",
                "pipe:0",
                "-an",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-q:v",
                "5",
                "pipe:1",
            ]
        )
        try:
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for ws_media_format=mpeg1video") from exc
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def feed(self, payload: bytes) -> None:
        if self.proc is None or self.proc.stdin is None:
            self.start()
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ConnectionError("ffmpeg decoder stopped accepting MPEG-1 data") from exc

    def _reader(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        buffer = bytearray()
        while not self.stopped:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                if self.proc.poll() is not None:
                    break
                time.sleep(0.001)
                continue
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    if len(buffer) > 2:
                        del buffer[:-2]
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    if start > 0:
                        del buffer[:start]
                    break
                jpeg = bytes(buffer[start : end + 2])
                del buffer[: end + 2]
                frame_bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    self.on_frame(frame_bgr)

    def close(self) -> None:
        self.stopped = True
        if self.proc is not None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.proc = None


class WebSocketFrameStream:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.url = stream_url(cfg)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.client: Optional[WebSocketImageClient] = None
        self.decoder: Optional[FfmpegMpeg1Decoder] = None
        if cfg.ws_media_format == "mpeg1video":
            self.decoder = FfmpegMpeg1Decoder(cfg, self._set_decoded_frame)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _set_decoded_frame(self, frame_bgr: np.ndarray) -> None:
        try:
            frame_bgr = preprocess_frame(frame_bgr, self.cfg)
        except Exception as exc:
            print(f"[ws] decoded frame preprocessing failed: {exc}")
            return
        with self.lock:
            self.frame = frame_bgr

    def _reader(self) -> None:
        min_dt = 1.0 / self.cfg.capture_fps if self.cfg.capture_fps > 0 else 0.0
        while not self.stopped:
            try:
                if self.client is None:
                    print(f"[ws] connecting {self.url}")
                    self.client = WebSocketImageClient(self.url)
                    self.client.connect()
                    print("[ws] connected")
                start = time.perf_counter()
                if self.decoder is not None:
                    payload, opcode = self.client.recv_message()
                    if opcode == 1:
                        payload = decode_websocket_text_payload(payload)
                    self.decoder.feed(payload)
                else:
                    frame_bgr = self.client.recv_image()
                    frame_bgr = preprocess_frame(frame_bgr, self.cfg)
                    with self.lock:
                        self.frame = frame_bgr
                if min_dt > 0:
                    elapsed = time.perf_counter() - start
                    if elapsed < min_dt:
                        time.sleep(min_dt - elapsed)
            except Exception as exc:
                if not self.stopped:
                    print(f"[ws] connection lost ({exc}); reconnecting in {self.cfg.reconnect_delay:.1f}s")
                    if self.client is not None:
                        self.client.close()
                        self.client = None
                    if self.decoder is not None:
                        self.decoder.close()
                        self.decoder = FfmpegMpeg1Decoder(self.cfg, self._set_decoded_frame)
                    time.sleep(self.cfg.reconnect_delay)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def release(self) -> None:
        self.stopped = True
        if self.client is not None:
            self.client.close()
        if self.decoder is not None:
            self.decoder.close()
        self.thread.join(timeout=1.0)


def _append_repo_path(repo_path: Optional[str]) -> None:
    if not repo_path:
        return
    repo = Path(repo_path).expanduser().resolve()
    candidates = [repo, repo / "metric_depth"]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _default_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and not torch.cuda.is_available():
        print("[model] --use_gpu was set, but CUDA is unavailable; falling back to CPU")
    return torch.device("cpu")


def _load_v1_zoe_metric(cfg: RuntimeConfig, device: torch.device):
    if not cfg.checkpoint:
        raise ValueError("V1 metric mode requires --checkpoint pointing to depth_anything_metric_depth_indoor.pt or outdoor.pt")

    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config

    conf = get_config("zoedepth", "infer")
    conf.pretrained_resource = f"local::{Path(cfg.checkpoint).expanduser().resolve()}"
    model = build_model(conf).to(device).eval()
    print(f"[model] loaded Depth Anything V1 metric/ZoeDepth checkpoint: {cfg.checkpoint}")
    return {"backend": "v1_zoe", "model": model, "device": device}


def _load_v2_metric(cfg: RuntimeConfig, device: torch.device):
    if not cfg.checkpoint:
        raise ValueError("V2 metric mode requires --checkpoint pointing to a metric Depth Anything V2 checkpoint")
    checkpoint = Path(cfg.checkpoint).expanduser()
    if not checkpoint.exists():
        model_files = []
        for directory in (Path("model"), Path("models")):
            if directory.exists():
                model_files.extend(str(path) for path in directory.glob("*.pth"))
        model_files = sorted(model_files)
        available = f" Available checkpoint files: {model_files}" if model_files else " No model/*.pth or models/*.pth files were found."
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            "For V2 metric, expected a file named like "
            "depth_anything_v2_metric_hypersim_vitl.pth or depth_anything_v2_metric_vkitti_vitl.pth."
            f"{available}"
        )

    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    if cfg.encoder not in model_configs:
        raise ValueError(f"Unsupported encoder for V2 metric: {cfg.encoder}")

    model = DepthAnythingV2(**model_configs[cfg.encoder], max_depth=cfg.max_depth)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model = model.to(device).eval()
    print(f"[model] loaded Depth Anything V2 metric checkpoint: {cfg.checkpoint}")
    return {"backend": "v2_metric", "model": model, "device": device}


def _load_da3(cfg: RuntimeConfig, device: torch.device):
    model_id = cfg.checkpoint or "depth-anything/DA3-LARGE-1.1"
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device).eval()
    print(f"[model] loaded Depth Anything 3 model: {model_id}")
    print("[model] warning: DA3-LARGE-1.1 outputs relative depth, not zero-shot metric meters")
    return {"backend": "da3", "model": model, "device": device}


def load_model(cfg: RuntimeConfig):
    _append_repo_path(cfg.repo_path)
    device = _default_device(cfg.use_gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    errors = []
    backends = ("v2_metric", "v1_zoe") if cfg.model_backend == "auto" else (cfg.model_backend,)
    for backend in backends:
        try:
            if backend == "v2_metric":
                handle = _load_v2_metric(cfg, device)
            elif backend == "v1_zoe":
                handle = _load_v1_zoe_metric(cfg, device)
            elif backend == "da3":
                handle = _load_da3(cfg, device)
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            handle["fp16"] = bool(cfg.fp16 and device.type == "cuda")
            if handle["fp16"] and backend != "da3":
                handle["model"].half()
                print("[model] CUDA fp16 enabled")
            if cfg.torch_compile and backend != "da3" and hasattr(torch, "compile"):
                print("[model] compiling model with torch.compile; expect slower warmup")
                handle["model"] = torch.compile(handle["model"], mode="reduce-overhead")
            return handle
        except Exception as exc:
            errors.append(f"{backend}: {exc}")

    details = "\n  ".join(errors)
    raise RuntimeError(
        "Could not load a metric Depth Anything model.\n"
        "Check --repo_path and --checkpoint. Errors:\n  "
        f"{details}"
    )


def _resize_for_model(rgb: np.ndarray, input_size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = input_size / max(h, w)
    new_w = max(14, int(round(w * scale / 14) * 14))
    new_h = max(14, int(round(h * scale / 14) * 14))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _rgb_to_tensor(rgb: np.ndarray, device: torch.device, fp16: bool, normalize_imagenet: bool) -> torch.Tensor:
    image = rgb.astype(np.float32) / 255.0
    if normalize_imagenet:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor.half() if fp16 else tensor.float()


def infer_depth(
    model_handle,
    frame_bgr: np.ndarray,
    input_size: int,
    output_scale: float = 1.0,
    da3_process_res_method: str = "upper_bound_resize",
) -> np.ndarray:
    """Return metric depth in meters, resized to the original frame resolution."""
    backend = model_handle["backend"]
    model = model_handle["model"]
    device = model_handle["device"]
    fp16 = model_handle.get("fp16", False)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb.shape[:2]
    out_h = max(1, int(round(original_h * output_scale)))
    out_w = max(1, int(round(original_w * output_scale)))

    with torch.inference_mode():
        if backend == "da3":
            prediction = model.inference(
                [rgb],
                process_res=input_size,
                process_res_method=da3_process_res_method,
                export_dir=None,
            )
            depth = prediction.depth
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().float().cpu().numpy()
            depth_m = np.asarray(depth[0] if np.asarray(depth).ndim == 3 else depth, dtype=np.float32)
            if depth_m.shape[:2] != (out_h, out_w):
                depth_m = cv2.resize(depth_m, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            del prediction
        elif backend == "v2_metric":
            # Metric V2 checkpoints output meters. We do preprocessing here instead of
            # calling infer_image() so the tensor stays on the user-selected device.
            resized = _resize_for_model(rgb, input_size)
            tensor = _rgb_to_tensor(resized, device, fp16, normalize_imagenet=True)
            with torch.cuda.amp.autocast(enabled=fp16):
                pred = model(tensor)
            if pred.ndim == 4:
                pred = pred[:, 0]
            pred = F.interpolate(pred.unsqueeze(1), size=(out_h, out_w), mode="bilinear", align_corners=True)
            depth_m = pred.squeeze().detach().float().cpu().numpy()
        else:
            resized = _resize_for_model(rgb, input_size)
            tensor = _rgb_to_tensor(resized, device, fp16, normalize_imagenet=False)
            with torch.cuda.amp.autocast(enabled=fp16):
                if hasattr(model, "infer"):
                    pred = model.infer(tensor)
                else:
                    pred = model(tensor)
            if isinstance(pred, dict):
                pred = pred.get("metric_depth", pred.get("out", next(iter(pred.values()))))
            if pred.ndim == 4:
                pred = pred[:, 0]
            pred = F.interpolate(pred.unsqueeze(1), size=(out_h, out_w), mode="bilinear", align_corners=False)
            depth_m = pred.squeeze().detach().float().cpu().numpy()

    if depth_m.shape[:2] != (out_h, out_w):
        depth_m = cv2.resize(depth_m, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    depth_m = np.nan_to_num(depth_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_m[depth_m < 0.0] = 0.0
    return depth_m


def normalize_pose_matrix(extrinsics) -> Optional[np.ndarray]:
    if extrinsics is None:
        return None
    ext = np.asarray(extrinsics, dtype=np.float64)
    if ext.shape == (1, 4, 4):
        ext = ext[0]
    elif ext.shape == (1, 3, 4):
        tmp = np.eye(4, dtype=np.float64)
        tmp[:3, :4] = ext[0]
        ext = tmp
    elif ext.shape == (3, 4):
        tmp = np.eye(4, dtype=np.float64)
        tmp[:3, :4] = ext
        ext = tmp
    if ext.shape != (4, 4):
        return None
    try:
        return np.linalg.inv(ext)
    except np.linalg.LinAlgError:
        return None


def apply_da3_metric_scale(depth_m: np.ndarray, frame_bgr: np.ndarray, cfg: RuntimeConfig) -> np.ndarray:
    depth_m = depth_m * float(cfg.da3_metric_scale)
    if not cfg.da3_metric_focal_scale:
        return depth_m
    fx, fy, _, _ = camera_intrinsics(frame_bgr.shape[1], frame_bgr.shape[0], cfg)
    focal = 0.5 * (fx + fy)
    return depth_m * float(focal)


def infer_depth_pose(
    model_handle,
    frame_bgr: np.ndarray,
    input_size: int,
    output_scale: float = 1.0,
    da3_process_res_method: str = "upper_bound_resize",
    cfg: Optional[RuntimeConfig] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if model_handle["backend"] != "da3":
        return infer_depth(model_handle, frame_bgr, input_size, output_scale, da3_process_res_method), None

    model = model_handle["model"]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb.shape[:2]
    out_h = max(1, int(round(original_h * output_scale)))
    out_w = max(1, int(round(original_w * output_scale)))
    with torch.inference_mode():
        prediction = model.inference(
            [rgb],
            process_res=input_size,
            process_res_method=da3_process_res_method,
            export_dir=None,
        )
    depth = prediction.depth
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().float().cpu().numpy()
    depth_m = np.asarray(depth[0] if np.asarray(depth).ndim == 3 else depth, dtype=np.float32)
    if depth_m.shape[:2] != (out_h, out_w):
        depth_m = cv2.resize(depth_m, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    depth_m = np.nan_to_num(depth_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_m[depth_m < 0.0] = 0.0
    if cfg is not None:
        depth_m = apply_da3_metric_scale(depth_m, frame_bgr, cfg)
    pose_world_cam = normalize_pose_matrix(getattr(prediction, "extrinsics", None))
    del prediction
    return depth_m, pose_world_cam


def depth_colormap(depth_m: np.ndarray, max_depth: float, fast_vis: bool) -> np.ndarray:
    valid = depth_m > 0
    if fast_vis:
        upper = max(max_depth, 0.1)
    elif np.any(valid):
        upper = min(max_depth, float(np.percentile(depth_m[valid], 98)))
        upper = max(upper, 0.1)
    else:
        upper = max_depth
    norm = np.clip(depth_m / upper, 0.0, 1.0)
    inv = (255.0 * (1.0 - norm)).astype(np.uint8)
    colored = cv2.applyColorMap(inv, cv2.COLORMAP_INFERNO)
    colored[~valid] = 0
    return colored


def create_display_window(cfg: RuntimeConfig) -> None:
    cv2.namedWindow("RGB | Metric Depth", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("RGB | Metric Depth", cfg.window_width, cfg.window_height)


def visualize(frame_bgr: np.ndarray, depth_m: Optional[np.ndarray], fps: float, cfg: RuntimeConfig) -> None:
    if cfg.display_scale != 1.0:
        frame_show = cv2.resize(
            frame_bgr,
            None,
            fx=cfg.display_scale,
            fy=cfg.display_scale,
            interpolation=cv2.INTER_AREA if cfg.display_scale < 1.0 else cv2.INTER_LINEAR,
        )
    else:
        frame_show = frame_bgr

    if depth_m is None:
        depth_vis = np.zeros_like(frame_show)
    else:
        if cfg.display_scale != 1.0:
            depth_show_m = cv2.resize(
                depth_m,
                (frame_show.shape[1], frame_show.shape[0]),
                interpolation=cv2.INTER_NEAREST if cfg.display_scale < 1.0 else cv2.INTER_LINEAR,
            )
        else:
            depth_show_m = depth_m
        depth_vis = depth_colormap(depth_show_m, cfg.max_depth, cfg.fast_vis)

    h = min(frame_show.shape[0], depth_vis.shape[0])
    rgb_show = cv2.resize(frame_show, (frame_show.shape[1], h))
    depth_show = cv2.resize(depth_vis, (frame_show.shape[1], h))
    canvas = np.hstack((rgb_show, depth_show))
    cv2.putText(canvas, f"FPS {fps:5.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 255, 40), 2, cv2.LINE_AA)
    if depth_m is not None and not cfg.fast_vis:
        valid = depth_m[depth_m > 0]
        if valid.size:
            cv2.putText(
                canvas,
                f"depth m: min {valid.min():.2f}  med {np.median(valid):.2f}  max {valid.max():.2f}",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 255, 40),
                2,
                cv2.LINE_AA,
            )
    cv2.imshow("RGB | Metric Depth", canvas)


def camera_intrinsics(width: int, height: int, cfg: RuntimeConfig) -> Tuple[float, float, float, float]:
    fx = cfg.camera_fx if cfg.camera_fx is not None else 0.9 * width
    fy = cfg.camera_fy if cfg.camera_fy is not None else fx
    cx = cfg.camera_cx if cfg.camera_cx is not None else (width - 1) * 0.5
    cy = cfg.camera_cy if cfg.camera_cy is not None else (height - 1) * 0.5
    return fx, fy, cx, cy


def load_calibration_intrinsics(cfg: RuntimeConfig) -> None:
    if cfg.camera_fx is not None and cfg.camera_fy is not None and cfg.camera_cx is not None and cfg.camera_cy is not None:
        return

    calibration = cfg.calibration
    if calibration is None and cfg.video:
        candidate = Path(cfg.video).expanduser().resolve().parent / "calibration.json"
        if candidate.exists():
            calibration = str(candidate)
    if calibration is None:
        return

    path = Path(calibration).expanduser()
    if not path.exists():
        print(f"[calibration] not found: {path}")
        return

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    camera = data["cameras"][0] if "cameras" in data else data
    fx = float(camera["focalLengthX"])
    fy = float(camera["focalLengthY"])
    cx = float(camera["principalPointX"])
    cy = float(camera["principalPointY"])
    width = float(camera["imageWidth"])
    height = float(camera["imageHeight"])

    if cfg.rotate == "90cw":
        fx, fy, cx, cy = fy, fx, height - 1.0 - cy, cx
        width, height = height, width
    elif cfg.rotate == "90ccw":
        fx, fy, cx, cy = fy, fx, cy, width - 1.0 - cx
        width, height = height, width
    elif cfg.rotate == "180":
        cx, cy = width - 1.0 - cx, height - 1.0 - cy

    scale = cfg.resolution_scale
    cfg.camera_fx = fx * scale
    cfg.camera_fy = fy * scale
    cfg.camera_cx = cx * scale
    cfg.camera_cy = cy * scale
    print(
        "[calibration] loaded "
        f"fx={cfg.camera_fx:.2f} fy={cfg.camera_fy:.2f} "
        f"cx={cfg.camera_cx:.2f} cy={cfg.camera_cy:.2f}"
    )


def depth_to_pointcloud(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    cfg: RuntimeConfig,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.pointcloud_compute_device in ("auto", "cuda") and torch.cuda.is_available():
        return depth_to_pointcloud_torch(rgb_bgr, depth_m, cfg, stride, torch.device("cuda"))

    h, w = depth_m.shape[:2]
    rgb_for_depth = cv2.resize(rgb_bgr, (w, h), interpolation=cv2.INTER_AREA if w < rgb_bgr.shape[1] else cv2.INTER_LINEAR)
    fx0, fy0, cx0, cy0 = camera_intrinsics(rgb_bgr.shape[1], rgb_bgr.shape[0], cfg)
    sx = w / float(rgb_bgr.shape[1])
    sy = h / float(rgb_bgr.shape[0])
    fx, fy, cx, cy = fx0 * sx, fy0 * sy, cx0 * sx, cy0 * sy
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[0:h:stride, 0:w:stride]
    max_depth = cfg.pointcloud_max_depth if cfg.pointcloud_max_depth > 0.0 else cfg.max_depth
    valid = (z > 0) & (z <= max_depth)
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack((x[valid], y[valid], z[valid]), axis=1)
    colors = cv2.cvtColor(rgb_for_depth[0:h:stride, 0:w:stride], cv2.COLOR_BGR2RGB)[valid]
    return points, colors


def depth_to_pointcloud_torch(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    cfg: RuntimeConfig,
    stride: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape[:2]
    rgb_for_depth = cv2.resize(
        rgb_bgr,
        (w, h),
        interpolation=cv2.INTER_AREA if w < rgb_bgr.shape[1] else cv2.INTER_LINEAR,
    )
    rgb_for_depth = cv2.cvtColor(rgb_for_depth, cv2.COLOR_BGR2RGB)

    fx0, fy0, cx0, cy0 = camera_intrinsics(rgb_bgr.shape[1], rgb_bgr.shape[0], cfg)
    sx = w / float(rgb_bgr.shape[1])
    sy = h / float(rgb_bgr.shape[0])
    fx, fy, cx, cy = fx0 * sx, fy0 * sy, cx0 * sx, cy0 * sy
    max_depth = cfg.pointcloud_max_depth if cfg.pointcloud_max_depth > 0.0 else cfg.max_depth

    z = torch.as_tensor(depth_m[0:h:stride, 0:w:stride], dtype=torch.float32, device=device)
    color = torch.as_tensor(rgb_for_depth[0:h:stride, 0:w:stride], dtype=torch.uint8, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(0, z.shape[0], device=device, dtype=torch.float32),
        torch.arange(0, z.shape[1], device=device, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx * stride
    yy = yy * stride
    valid = (z > 0) & (z <= max_depth)
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    points = torch.stack((x, y, z), dim=-1)[valid]
    colors = color[valid]
    return points.detach().cpu().numpy(), colors.detach().cpu().numpy()


def scale_depth_for_preview(depth_m: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return depth_m
    h, w = depth_m.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(depth_m, (out_w, out_h), interpolation=interpolation)


def save_pointcloud_ply(points: np.ndarray, colors_rgb: np.ndarray, save_dir: str, index: int) -> None:
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = root / f"cloud_{index:06d}_{stamp}.ply"
    colors_u8 = np.clip(colors_rgb, 0, 255).astype(np.uint8)
    vertices = np.empty(
        points.shape[0],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"] = points[:, 0].astype(np.float32)
    vertices["y"] = points[:, 1].astype(np.float32)
    vertices["z"] = points[:, 2].astype(np.float32)
    vertices["red"] = colors_u8[:, 0]
    vertices["green"] = colors_u8[:, 1]
    vertices["blue"] = colors_u8[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)
    print(f"\n[pointcloud] saved {path} ({len(vertices)} points)")


def write_pointcloud_ply(points: np.ndarray, colors_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = np.clip(colors_rgb, 0, 255).astype(np.uint8)
    vertices = np.empty(
        points.shape[0],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"] = points[:, 0].astype(np.float32)
    vertices["y"] = points[:, 1].astype(np.float32)
    vertices["z"] = points[:, 2].astype(np.float32)
    vertices["red"] = colors_u8[:, 0]
    vertices["green"] = colors_u8[:, 1]
    vertices["blue"] = colors_u8[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def pose_to_list(pose: Optional[np.ndarray]):
    return None if pose is None else np.asarray(pose, dtype=float).tolist()


def save_dataset_sample(
    frame_bgr: np.ndarray,
    depth_m: np.ndarray,
    cfg: RuntimeConfig,
    index: int,
    frame_index: int,
    infer_index: int,
    pose_world_cam: Optional[np.ndarray],
) -> None:
    if not cfg.dataset_save_dir:
        return
    root = Path(cfg.dataset_save_dir)
    stamp_ns = time.time_ns()
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{stamp_ns % 1_000_000_000:09d}"
    name = f"{index:06d}_{stamp}"

    rgb_dir = root / "rgb"
    depth_dir = root / "depth_mm"
    ply_dir = root / "ply"
    meta_dir = root / "meta"
    for directory in (rgb_dir, depth_dir, ply_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rgb_path = rgb_dir / f"{name}.png"
    depth_path = depth_dir / f"{name}.png"
    ply_path = ply_dir / f"{name}.ply"
    meta_path = meta_dir / f"{name}.json"

    depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(rgb_path), frame_bgr)
    cv2.imwrite(str(depth_path), depth_mm)

    original_max_depth = cfg.pointcloud_max_depth
    if cfg.dataset_save_max_depth > 0.0:
        cfg.pointcloud_max_depth = cfg.dataset_save_max_depth
    try:
        points, colors = depth_to_pointcloud(frame_bgr, depth_m, cfg, stride=cfg.dataset_save_stride)
    finally:
        cfg.pointcloud_max_depth = original_max_depth
    write_pointcloud_ply(points, colors, ply_path)

    fx, fy, cx, cy = camera_intrinsics(frame_bgr.shape[1], frame_bgr.shape[0], cfg)
    meta = {
        "timestamp_ns": stamp_ns,
        "timestamp_name": stamp,
        "frame_index": frame_index,
        "infer_index": infer_index,
        "rgb": str(rgb_path.relative_to(root)),
        "depth_mm": str(depth_path.relative_to(root)),
        "ply": str(ply_path.relative_to(root)),
        "depth_unit": "millimeters_uint16",
        "pointcloud_stride": cfg.dataset_save_stride,
        "pointcloud_points": int(len(points)),
        "camera": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": int(frame_bgr.shape[1]),
            "height": int(frame_bgr.shape[0]),
        },
        "pose_world_cam": pose_to_list(pose_world_cam),
        "model_backend": cfg.model_backend,
        "checkpoint": cfg.checkpoint,
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"\n[dataset] saved {name} points={len(points)}")


class PointCloudViewer:
    def __init__(self) -> None:
        import open3d as o3d

        self.o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Metric Point Cloud", width=960, height=720)
        self.pcd = o3d.geometry.PointCloud()
        self.added = False

    def update(self, points: np.ndarray, colors: np.ndarray) -> None:
        self.pcd.points = self.o3d.utility.Vector3dVector(points)
        self.pcd.colors = self.o3d.utility.Vector3dVector(colors)
        if not self.added:
            self.vis.add_geometry(self.pcd)
            self.added = True
        else:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self) -> None:
        self.vis.destroy_window()


class PointCloudWebSocketServer:
    def __init__(self, host: str, port: int, max_points: int = 120000) -> None:
        self.host = host
        self.port = port
        self.max_points = max_points
        self.sock: Optional[socket.socket] = None
        self.clients = []
        self.clients_lock = threading.Lock()
        self.pending_payload: Optional[bytes] = None
        self.event = threading.Event()
        self.stopped = False
        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)

    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(8)
        self.sock.settimeout(0.5)
        self.accept_thread.start()
        self.broadcast_thread.start()
        actual_host, actual_port = self.sock.getsockname()[:2]
        print(f"[pointcloud-web] ws://{actual_host}:{actual_port}")

    def publish(self, points: np.ndarray, colors_rgb: np.ndarray) -> None:
        if not self.has_clients() or points.size == 0:
            return
        payload = self._encode(points, colors_rgb)
        self.pending_payload = payload
        self.event.set()

    def has_clients(self) -> bool:
        with self.clients_lock:
            return bool(self.clients)

    def _accept_loop(self) -> None:
        while not self.stopped:
            try:
                if self.sock is None:
                    break
                client, address = self.sock.accept()
                client.settimeout(2.0)
                self._handshake(client)
                client.settimeout(0.2)
                with self.clients_lock:
                    self.clients.append(client)
                print(f"\n[pointcloud-web] client connected {address[0]}:{address[1]}")
            except socket.timeout:
                continue
            except Exception as exc:
                if not self.stopped:
                    print(f"\n[pointcloud-web] accept failed: {exc}")

    def _handshake(self, client: socket.socket) -> None:
        request = self._recv_until(client, b"\r\n\r\n", 16384).decode("iso-8859-1", errors="replace")
        headers = {}
        for line in request.split("\r\n")[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        key = headers.get("sec-websocket-key")
        if not key:
            raise ConnectionError("missing Sec-WebSocket-Key")
        accept = base64.b64encode(hashlib.sha1((key + WebSocketImageClient.GUID).encode("ascii")).digest()).decode("ascii")
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        )
        client.sendall(response.encode("ascii"))

    @staticmethod
    def _recv_until(sock_obj: socket.socket, marker: bytes, max_bytes: int) -> bytes:
        data = bytearray()
        while marker not in data:
            chunk = sock_obj.recv(1)
            if not chunk:
                raise ConnectionError("socket closed during handshake")
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ConnectionError("handshake request too large")
        return bytes(data)

    def _broadcast_loop(self) -> None:
        while not self.stopped:
            self.event.wait(timeout=0.2)
            if self.stopped:
                break
            payload = self.pending_payload
            self.pending_payload = None
            self.event.clear()
            if payload is None:
                continue
            frame = self._websocket_binary_frame(payload)
            dead_clients = []
            with self.clients_lock:
                clients = list(self.clients)
            for client in clients:
                try:
                    client.sendall(frame)
                except Exception:
                    dead_clients.append(client)
            if dead_clients:
                with self.clients_lock:
                    self.clients = [client for client in self.clients if client not in dead_clients]
                for client in dead_clients:
                    try:
                        client.close()
                    except OSError:
                        pass

    def _encode(self, points: np.ndarray, colors_rgb: np.ndarray) -> bytes:
        if self.max_points > 0 and len(points) > self.max_points:
            indices = np.linspace(0, len(points) - 1, self.max_points, dtype=np.int64)
            points = points[indices]
            colors_rgb = colors_rgb[indices]
        colors = colors_rgb.astype(np.float32)
        if colors.size and colors.max() > 1.0:
            colors *= 1.0 / 255.0
        packed = np.empty((len(points), 6), dtype=np.float32)
        packed[:, :3] = points.astype(np.float32)
        packed[:, 3:] = np.clip(colors, 0.0, 1.0)
        return packed.tobytes()

    @staticmethod
    def _websocket_binary_frame(payload: bytes) -> bytes:
        length = len(payload)
        if length < 126:
            header = bytes((0x82, length))
        elif length < 65536:
            header = bytes((0x82, 126)) + struct.pack("!H", length)
        else:
            header = bytes((0x82, 127)) + struct.pack("!Q", length)
        return header + payload

    def close(self) -> None:
        self.stopped = True
        self.event.set()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        with self.clients_lock:
            clients = list(self.clients)
            self.clients = []
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        self.accept_thread.join(timeout=1.0)
        self.broadcast_thread.join(timeout=1.0)


class LiveRegistrationPreview:
    def __init__(self, cfg: RuntimeConfig) -> None:
        import open3d as o3d

        self.cfg = cfg
        self.o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Live Registration Preview", width=1200, height=800)
        self.display_pcd = o3d.geometry.PointCloud()
        self.added = False

        self.lock = threading.Lock()
        self.event = threading.Event()
        self.stop_event = threading.Event()
        self.pending = None
        self.latest_map = None
        self.map_pcd = None
        self.last_keyframe = None
        self.last_keyframe_fpfh = None
        self.last_pose_world_cam = None
        self.da3_pose_reference = None
        self.accumulated_pose = np.eye(4, dtype=np.float64)
        self.pose = np.eye(4, dtype=np.float64)
        self.keyframes = 0
        self.busy = False
        self.teaserpp = None

        cuda_available = False
        try:
            cuda_available = bool(o3d.core.cuda.is_available())
        except Exception:
            pass
        if cfg.registration_device == "cuda" and not cuda_available:
            print("[registration] Open3D CUDA is not available; RANSAC/FPFH preview will run on CPU")
        elif cfg.registration_device in ("cuda", "auto") and cuda_available:
            print("[registration] Open3D CUDA is available, but legacy FPFH/RANSAC is CPU-bound in this preview")
        if cfg.registration_backend in ("teaser", "robust"):
            try:
                import teaserpp_python

                self.teaserpp = teaserpp_python
                print(f"[registration] using {cfg.registration_backend}; FPFH matching and ICP remain Open3D")
            except Exception as exc:
                if cfg.registration_backend == "teaser":
                    print(f"[registration] TEASER++ Python binding unavailable: {exc}; falling back to Open3D RANSAC")
                    self.cfg.registration_backend = "open3d_ransac"
                else:
                    print(f"[registration] TEASER++ unavailable for robust backend: {exc}; robust will use RANSAC+ICP")

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _make_pcd(self, points: np.ndarray, colors_rgb: np.ndarray):
        if len(points) > self.cfg.registration_max_points:
            idx = np.random.choice(len(points), self.cfg.registration_max_points, replace=False)
            points = points[idx]
            colors_rgb = colors_rgb[idx]
        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = self.o3d.utility.Vector3dVector(colors_rgb.astype(np.float64) / 255.0)
        return pcd

    def _prepare_features(self, pcd):
        down = pcd.voxel_down_sample(self.cfg.registration_voxel)
        if len(down.points) < 50:
            return None, None
        normal_radius = self.cfg.registration_voxel * 2.5
        down.estimate_normals(self.o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        fpfh = self.o3d.pipelines.registration.compute_fpfh_feature(
            down,
            self.o3d.geometry.KDTreeSearchParamHybrid(radius=self.cfg.registration_feature_radius, max_nn=100),
        )
        return down, fpfh

    def _feature_correspondences(self, source_fpfh, target_fpfh) -> Tuple[np.ndarray, np.ndarray]:
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
        if self.cfg.teaser_mutual_filter:
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

        if len(src_idx) > self.cfg.teaser_max_correspondences:
            keep = np.argsort(src_to_tgt_dist)[: self.cfg.teaser_max_correspondences]
            src_idx = src_idx[keep]
            tgt_idx = tgt_idx[keep]
        return src_idx, tgt_idx

    def _register(self, source_down, target_down, source_fpfh, target_fpfh) -> np.ndarray:
        if self.cfg.registration_backend == "robust":
            return self._register_robust(source_down, target_down, source_fpfh, target_fpfh)
        if self.cfg.registration_backend == "teaser":
            return self._register_teaser(source_down, target_down, source_fpfh, target_fpfh)
        return self._register_open3d_ransac(source_down, target_down, source_fpfh, target_fpfh)

    def _cap_map_points(self) -> None:
        if self.cfg.registration_max_map_points <= 0 or self.map_pcd is None:
            return
        count = len(self.map_pcd.points)
        if count <= self.cfg.registration_max_map_points:
            return
        indices = np.random.choice(count, self.cfg.registration_max_map_points, replace=False)
        self.map_pcd = self.map_pcd.select_by_index(indices.tolist())

    def _da3_live_transform(self, pose_world_cam: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pose_world_cam is None:
            return None
        if self.da3_pose_reference is None:
            self.da3_pose_reference = pose_world_cam.copy()
            self.accumulated_pose = np.eye(4, dtype=np.float64)
            return self.accumulated_pose.copy()
        try:
            relative = np.linalg.inv(self.da3_pose_reference) @ pose_world_cam
        except np.linalg.LinAlgError:
            return None
        self.accumulated_pose = relative
        return self.accumulated_pose.copy()

    def _register_open3d_ransac(self, source_down, target_down, source_fpfh, target_fpfh) -> np.ndarray:
        reg = self.o3d.pipelines.registration
        result = reg.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.cfg.registration_ransac_distance,
            estimation_method=reg.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            criteria=reg.RANSACConvergenceCriteria(400000, 500),
        )
        try:
            result_icp = reg.registration_icp(
                source_down,
                target_down,
                self.cfg.registration_icp_distance,
                result.transformation,
                reg.TransformationEstimationPointToPlane(),
            )
            return result_icp.transformation
        except Exception:
            return result.transformation

    def _refine_and_score(self, source_down, target_down, init: np.ndarray):
        reg = self.o3d.pipelines.registration
        transform = init
        result = None
        for distance, iterations in (
            (self.cfg.registration_icp_distance, 50),
            (self.cfg.registration_icp_fine_distance, 30),
        ):
            result = reg.registration_icp(
                source_down,
                target_down,
                distance,
                transform,
                reg.TransformationEstimationPointToPlane(),
                reg.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=iterations),
            )
            transform = result.transformation
        if result is None:
            return transform, 0.0, float("inf")
        return transform, float(result.fitness), float(result.inlier_rmse)

    def _is_registration_acceptable(self, fitness: float, rmse: float) -> bool:
        if self.cfg.registration_min_fitness > 0.0 and fitness < self.cfg.registration_min_fitness:
            return False
        if self.cfg.registration_max_rmse > 0.0 and rmse > self.cfg.registration_max_rmse:
            return False
        return True

    def _register_robust(self, source_down, target_down, source_fpfh, target_fpfh) -> np.ndarray:
        candidates = []
        identity = np.eye(4, dtype=np.float64)

        if self.teaserpp is not None:
            try:
                teaser_init = self._register_teaser_global_only(source_down, target_down, source_fpfh, target_fpfh)
                transform, fitness, rmse = self._refine_and_score(source_down, target_down, teaser_init)
                candidates.append(("teaser", transform, fitness, rmse))
            except Exception as exc:
                print(f"\n[registration] robust TEASER candidate failed: {exc}")

        try:
            ransac_init = self._register_open3d_ransac_global_only(source_down, target_down, source_fpfh, target_fpfh)
            transform, fitness, rmse = self._refine_and_score(source_down, target_down, ransac_init)
            candidates.append(("ransac", transform, fitness, rmse))
        except Exception as exc:
            print(f"\n[registration] robust RANSAC candidate failed: {exc}")

        try:
            transform, fitness, rmse = self._refine_and_score(source_down, target_down, identity)
            candidates.append(("icp_identity", transform, fitness, rmse))
        except Exception:
            pass

        if not candidates:
            print("\n[registration] robust failed; using identity")
            return identity

        candidates.sort(key=lambda item: (item[2], -item[3]), reverse=True)
        name, transform, fitness, rmse = candidates[0]
        if not self._is_registration_acceptable(fitness, rmse):
            print(f"\n[registration] robust rejected {name}: fitness={fitness:.3f} rmse={rmse:.4f}; using identity")
            return identity
        print(f"\n[registration] robust selected {name}: fitness={fitness:.3f} rmse={rmse:.4f}")
        return transform

    def _register_open3d_ransac_global_only(self, source_down, target_down, source_fpfh, target_fpfh) -> np.ndarray:
        reg = self.o3d.pipelines.registration
        result = reg.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.cfg.registration_ransac_distance,
            estimation_method=reg.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            criteria=reg.RANSACConvergenceCriteria(800000, 1000),
        )
        return result.transformation

    def _register_teaser(self, source_down, target_down, source_fpfh, target_fpfh) -> np.ndarray:
        if self.teaserpp is None:
            return self._register_open3d_ransac(source_down, target_down, source_fpfh, target_fpfh)

        transform = self._register_teaser_global_only(source_down, target_down, source_fpfh, target_fpfh)
        try:
            transform, fitness, rmse = self._refine_and_score(source_down, target_down, transform)
            print(f"\n[registration] TEASER++ refined fitness={fitness:.3f} rmse={rmse:.4f}")
        except Exception:
            pass
        return transform

    def _register_teaser_global_only(self, source_down, target_down, source_fpfh, target_fpfh) -> np.ndarray:
        if self.teaserpp is None:
            raise RuntimeError("TEASER++ is not available")

        src_idx, tgt_idx = self._feature_correspondences(source_fpfh, target_fpfh)
        if len(src_idx) < 6:
            raise RuntimeError(f"TEASER++ got too few correspondences ({len(src_idx)})")

        src_pts_all = np.asarray(source_down.points)
        tgt_pts_all = np.asarray(target_down.points)
        src = src_pts_all[src_idx].T.astype(np.float64)
        dst = tgt_pts_all[tgt_idx].T.astype(np.float64)

        params = self.teaserpp.RobustRegistrationSolver.Params()
        params.cbar2 = 1.0
        params.noise_bound = self.cfg.teaser_noise_bound
        params.estimate_scaling = False
        params.rotation_estimation_algorithm = (
            self.teaserpp.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        params.rotation_gnc_factor = 1.4
        params.rotation_max_iterations = 100
        params.rotation_cost_threshold = 1e-12

        solver = self.teaserpp.RobustRegistrationSolver(params)
        solver.solve(src, dst)
        solution = solver.getSolution()

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(solution.rotation, dtype=np.float64)
        transform[:3, 3] = np.asarray(solution.translation, dtype=np.float64).reshape(3)

        print(f"\n[registration] TEASER++ correspondences={len(src_idx)}")
        return transform

    def submit(self, points: np.ndarray, colors_rgb: np.ndarray, pose_world_cam: Optional[np.ndarray] = None) -> None:
        if points.size == 0:
            return
        with self.lock:
            if self.busy:
                return
            self.pending = (points.copy(), colors_rgb.copy(), None if pose_world_cam is None else pose_world_cam.copy())
            self.busy = True
            self.event.set()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            self.event.wait(timeout=0.1)
            if self.stop_event.is_set():
                break
            with self.lock:
                item = self.pending
                self.pending = None
                self.event.clear()
            if item is None:
                continue

            try:
                points, colors, pose_world_cam = item
                pcd = self._make_pcd(points, colors)
                source_down, source_fpfh = self._prepare_features(pcd)
                if source_down is None:
                    with self.lock:
                        self.busy = False
                    print("\n[registration] skipped sparse keyframe")
                    continue

                if self.last_keyframe is None:
                    da3_transform = self._da3_live_transform(pose_world_cam) if self.cfg.registration_backend == "da3_pose" else None
                    transform = da3_transform if da3_transform is not None else self.pose
                    pcd_world = pcd.transform(transform.copy())
                    source_down.transform(transform.copy())
                    self.last_keyframe = source_down
                    self.last_keyframe_fpfh = source_fpfh
                    self.last_pose_world_cam = transform.copy()
                    self.map_pcd = pcd_world.voxel_down_sample(self.cfg.registration_map_voxel)
                    self.keyframes = 1
                else:
                    target_down = self.last_keyframe
                    target_fpfh = self.last_keyframe_fpfh
                    if self.cfg.registration_backend == "da3_pose" and pose_world_cam is not None:
                        transform = self._da3_live_transform(pose_world_cam)
                        if transform is None:
                            transform = self.last_pose_world_cam if self.last_pose_world_cam is not None else self.pose
                        if self.cfg.registration_pose_icp_refine:
                            transform, fitness, rmse = self._refine_and_score(source_down, target_down, transform)
                            print(f"\n[registration] da3_pose ICP fitness={fitness:.3f} rmse={rmse:.4f}")
                    else:
                        transform = self._register(source_down, target_down, source_fpfh, target_fpfh)
                    pcd_world = pcd.transform(transform)
                    source_down.transform(transform)
                    self.last_keyframe = source_down
                    self.last_keyframe_fpfh = source_fpfh
                    self.last_pose_world_cam = transform.copy()
                    self.map_pcd += pcd_world.voxel_down_sample(self.cfg.registration_map_voxel)
                    self.map_pcd = self.map_pcd.voxel_down_sample(self.cfg.registration_map_voxel)
                    self._cap_map_points()
                    self.keyframes += 1

                with self.lock:
                    self.latest_map = copy.deepcopy(self.map_pcd)
                    self.busy = False
                print(f"\n[registration] keyframes={self.keyframes} points={len(self.map_pcd.points)}")
                print(transform)
            except Exception as exc:
                with self.lock:
                    self.busy = False
                print(f"\n[registration] failed: {exc}")

    def poll(self) -> None:
        with self.lock:
            latest = self.latest_map
            self.latest_map = None
        if latest is not None:
            self.display_pcd.points = latest.points
            self.display_pcd.colors = latest.colors
            if not self.added:
                self.vis.add_geometry(self.display_pcd)
                self.added = True
            else:
                self.vis.update_geometry(self.display_pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self) -> None:
        self.stop_event.set()
        self.event.set()
        self.thread.join(timeout=1.0)
        self.vis.destroy_window()


def save_synchronized(rgb_bgr: np.ndarray, depth_m: np.ndarray, save_dir: str, index: int) -> None:
    root = Path(save_dir)
    rgb_dir = root / "rgb"
    depth_dir = root / "depth_mm"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{index:06d}_{stamp}"
    depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(rgb_dir / f"{name}.png"), rgb_bgr)
    cv2.imwrite(str(depth_dir / f"{name}.png"), depth_mm)
    print(f"[save] {name}.png")


def sleep_for_fps_limit(loop_start: float, fps_limit: float) -> None:
    if fps_limit <= 0:
        return
    target_dt = 1.0 / fps_limit
    elapsed = time.perf_counter() - loop_start
    if elapsed < target_dt:
        time.sleep(target_dt - elapsed)


def main() -> int:
    cfg = parse_args()
    load_calibration_intrinsics(cfg)
    source_label = cfg.video if cfg.video else stream_url(cfg)
    print(f"[stream] opening {source_label}")
    print("[keys] q/ESC quit | s save RGB+depth | p save point cloud | d save dataset sample")

    try:
        model_handle = load_model(cfg)
    except Exception as exc:
        print(f"[fatal] model load failed: {exc}", file=sys.stderr)
        return 2

    ws_source = is_websocket_source(cfg)
    cap = None if (ws_source or (cfg.low_latency and not cfg.video)) else open_stream(cfg)
    if ws_source:
        stream = WebSocketFrameStream(cfg)
    elif cfg.low_latency and not cfg.video:
        stream = LatestFrameStream(cfg)
    else:
        stream = None
    video_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None and cfg.video else 0.0
    create_display_window(cfg)
    viewer = None
    if cfg.pointcloud:
        try:
            viewer = PointCloudViewer()
        except Exception as exc:
            print(f"[pointcloud] Open3D unavailable or viewer failed: {exc}")
    pointcloud_web = None
    if cfg.pointcloud_web:
        try:
            pointcloud_web = PointCloudWebSocketServer(
                cfg.pointcloud_web_host,
                cfg.pointcloud_web_port,
                cfg.pointcloud_web_max_points,
            )
            pointcloud_web.start()
        except Exception as exc:
            print(f"[pointcloud-web] server failed: {exc}")
    registration_preview = None
    if cfg.live_registration:
        try:
            registration_preview = LiveRegistrationPreview(cfg)
        except Exception as exc:
            print(f"[registration] Open3D preview failed: {exc}")

    fps_meter = FPSMeter()
    frame_index = 0
    infer_index = 0
    saved_index = 0
    pointcloud_saved_index = 0
    dataset_saved_index = 0
    last_depth = None
    last_pose_world_cam = None
    last_infer_error = 0.0
    timings = {"capture": 0.0, "infer": 0.0, "visualize": 0.0, "pointcloud": 0.0}
    timing_count = 0
    profile_infer_count = 0
    last_profile = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            t0 = time.perf_counter()
            if stream is not None:
                frame_bgr = stream.read()
            else:
                frame_bgr, cap = get_frame(cap, cfg)
            timings["capture"] += time.perf_counter() - t0
            if frame_bgr is None:
                if cfg.video:
                    print("\n[video] end of file")
                    break
                time.sleep(0.001)
                continue

            run_infer = last_depth is None or frame_index % (cfg.frame_skip + 1) == 0
            if run_infer:
                try:
                    t0 = time.perf_counter()
                    last_depth, last_pose_world_cam = infer_depth_pose(
                        model_handle,
                        frame_bgr,
                        cfg.input_size,
                        cfg.depth_output_scale,
                        cfg.da3_process_res_method,
                        cfg,
                    )
                    timings["infer"] += time.perf_counter() - t0
                    infer_index += 1
                    profile_infer_count += 1
                    if (
                        cfg.cuda_empty_cache_every > 0
                        and model_handle["device"].type == "cuda"
                        and infer_index % cfg.cuda_empty_cache_every == 0
                    ):
                        torch.cuda.empty_cache()
                except Exception as exc:
                    now = time.perf_counter()
                    if now - last_infer_error > 2.0:
                        print(f"[infer] failed; keeping previous depth if available: {exc}")
                        last_infer_error = now

            fps = fps_meter.tick()
            if frame_index % cfg.display_every == 0:
                t0 = time.perf_counter()
                visualize(frame_bgr, last_depth, fps, cfg)
                timings["visualize"] += time.perf_counter() - t0
            if frame_index % cfg.print_every == 0:
                print(f"\rFPS {fps:5.1f} | frame {frame_index:06d} | infer {infer_index:06d}", end="", flush=True)

            if viewer is not None and last_depth is not None and run_infer:
                t0 = time.perf_counter()
                points, colors = depth_to_pointcloud(frame_bgr, last_depth, cfg, stride=max(cfg.pointcloud_stride, 2))
                if points.size:
                    viewer.update(points, colors.astype(np.float32) / 255.0)
                timings["pointcloud"] += time.perf_counter() - t0
            if pointcloud_web is not None and last_depth is not None and run_infer:
                if infer_index % cfg.pointcloud_web_every == 0 and pointcloud_web.has_clients():
                    t0 = time.perf_counter()
                    points, colors = depth_to_pointcloud(frame_bgr, last_depth, cfg, stride=cfg.pointcloud_web_stride)
                    if points.size:
                        pointcloud_web.publish(points, colors)
                    timings["pointcloud"] += time.perf_counter() - t0
            if registration_preview is not None and last_depth is not None and run_infer:
                if infer_index % cfg.registration_every == 0:
                    registration_depth = scale_depth_for_preview(last_depth, cfg.registration_depth_scale)
                    points, colors = depth_to_pointcloud(frame_bgr, registration_depth, cfg, stride=cfg.registration_stride)
                    registration_preview.submit(points, colors, last_pose_world_cam)
            if registration_preview is not None:
                registration_preview.poll()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s") and cfg.save_dir and last_depth is not None:
                save_synchronized(frame_bgr, last_depth, cfg.save_dir, saved_index)
                saved_index += 1
            if key == ord("p") and cfg.pointcloud_save_dir and last_depth is not None:
                points, colors = depth_to_pointcloud(frame_bgr, last_depth, cfg, stride=cfg.pointcloud_stride)
                if points.size:
                    save_pointcloud_ply(points, colors, cfg.pointcloud_save_dir, pointcloud_saved_index)
                    pointcloud_saved_index += 1
            if key == ord("d") and cfg.dataset_save_dir and last_depth is not None:
                save_dataset_sample(frame_bgr, last_depth, cfg, dataset_saved_index, frame_index, infer_index, last_pose_world_cam)
                dataset_saved_index += 1

            if cfg.save_dir and cfg.save_every > 0 and last_depth is not None and run_infer:
                if infer_index % cfg.save_every == 0:
                    save_synchronized(frame_bgr, last_depth, cfg.save_dir, saved_index)
                    saved_index += 1
            if cfg.dataset_save_dir and cfg.dataset_save_every > 0 and last_depth is not None and run_infer:
                if infer_index % cfg.dataset_save_every == 0:
                    save_dataset_sample(frame_bgr, last_depth, cfg, dataset_saved_index, frame_index, infer_index, last_pose_world_cam)
                    dataset_saved_index += 1
            if cfg.pointcloud_save_dir and cfg.pointcloud_save_every > 0 and last_depth is not None and run_infer:
                if infer_index % cfg.pointcloud_save_every == 0:
                    points, colors = depth_to_pointcloud(frame_bgr, last_depth, cfg, stride=cfg.pointcloud_stride)
                    if points.size:
                        save_pointcloud_ply(points, colors, cfg.pointcloud_save_dir, pointcloud_saved_index)
                        pointcloud_saved_index += 1

            frame_index += 1
            timing_count += 1
            if cfg.profile_interval > 0.0 and time.perf_counter() - last_profile >= cfg.profile_interval:
                denom = max(timing_count, 1)
                print(
                    "\n[profile] avg ms/frame "
                    f"capture={timings['capture'] / denom * 1000:.1f} "
                    f"infer={timings['infer'] / max(profile_infer_count, 1) * 1000:.1f} "
                    f"vis={timings['visualize'] / denom * 1000:.1f} "
                    f"pc={timings['pointcloud'] / denom * 1000:.1f}"
                )
                if model_handle["device"].type == "cuda":
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    print(f"[profile] cuda memory allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
                timings = {"capture": 0.0, "infer": 0.0, "visualize": 0.0, "pointcloud": 0.0}
                timing_count = 0
                profile_infer_count = 0
                last_profile = time.perf_counter()
            if cfg.video and cfg.realtime_video and video_fps > 0:
                sleep_for_fps_limit(loop_start, video_fps)
            sleep_for_fps_limit(loop_start, cfg.fps_limit)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if stream is not None:
            stream.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if viewer is not None:
            viewer.close()
        if pointcloud_web is not None:
            pointcloud_web.close()
        if registration_preview is not None:
            registration_preview.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
