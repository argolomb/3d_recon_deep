# 3D_RECON_DEEP

Real-time and offline depth/point-cloud reconstruction experiments with Android IP Webcam, Depth Anything V2 Metric, Depth Anything 3 Large/Metric, Open3D, TEASER++, ICP, TSDF and optional 3D Gaussian Splatting export.

The project is built around a practical pipeline:

```text
RGB video / IP Webcam
  -> Depth Anything
  -> depth map
  -> point cloud
  -> optional pose/registration
  -> live preview or offline fusion
```

## Features

- Live depth from Android IP Webcam (`http://<ip>:8080/video`)
- Local MP4/video playback for repeatable tests
- Depth Anything V2 Metric support
- Depth Anything 3 Large support
- Depth Anything 3 Metric Large support
- OpenCV visualization: RGB + depth
- Open3D point-cloud viewer
- Live registration preview with multiple backends:
  - `open3d_ransac`
  - `teaser`
  - `robust`
  - `da3_pose`
- CUDA inference and CUDA depth-to-point-cloud projection
- YAML configuration files to avoid long CLI commands
- RGB/depth logging for SLAM
- Dense PLY point-cloud export
- Offline TSDF fusion
- Optional DA3 3DGS export helper, limited to DA3 models that expose Gaussian branch

## Repository Layout

```text
main.py                  # live/video depth + visualization + point cloud + registration preview
offline_fusion.py        # V2 Metric / TEASER++ / ICP / TSDF offline pipeline
da3_large_pipeline.py    # DA3 Large depth/pose offline point-cloud/TSDF pipeline
da3_gs_render.py         # DA3 3D Gaussian Splatting export helper

config_da3_live.yaml     # DA3 Large live/video configuration
config_da3metric_live.yaml # DA3Metric Large live configuration

models/                  # downloaded model snapshots/checkpoints
Depth-Anything-V2/       # official Depth Anything V2 repo
depth-anything-3/        # official Depth Anything 3 repo
TEASER-plusplus/         # optional TEASER++ checkout/build
```

## Requirements

Tested with Python 3.10 in a `uv` virtual environment.

Core dependencies:

```text
opencv-python
numpy
torch
torchvision
open3d
depth-anything-3
depth-anything-v2 repo
teaserpp_python optional
gsplat optional
```

## Environment Setup

Create and activate a virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

Install base dependencies:

```bash
uv pip install opencv-python numpy torch torchvision open3d huggingface_hub pyyaml
```

Install Depth Anything 3:

```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3
cd depth-anything-3
uv pip install -e .
cd ..
```

Install Depth Anything V2:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2/metric_depth
uv pip install -r requirements.txt
cd ../..
```

Optional TEASER++:

Follow the official build instructions for your system:

```text
https://github.com/MIT-SPARK/TEASER-plusplus
```

The Python binding must be importable as:

```python
import teaserpp_python
```

Optional 3DGS rendering dependency:

```bash
uv pip install "git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"
```

## Download Models

Depth Anything 3 Large:

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='depth-anything/DA3-LARGE-1.1', local_dir='models/DA3-LARGE-1.1')"
```

Depth Anything 3 Metric Large:

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='depth-anything/DA3METRIC-LARGE', local_dir='models/DA3METRIC-LARGE')"
```

Depth Anything V2 Metric examples:

```text
models/depth_anything_v2_metric_hypersim_vitl.pth  # indoor
models/depth_anything_v2_metric_vkitti_vitl.pth    # outdoor
```

Download those from Hugging Face:

```text
https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large
https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large
```

## Android IP Webcam Input

Install the Android app “IP Webcam”, start the server, and use:

```text
http://<PHONE_IP>:8080/video
```

In YAML you can write:

```yaml
ip: 192.168.15.3:8080/video
```

or:

```yaml
ip: 192.168.15.3
port: 8080
stream_path: /video
```

## WebSocket Input

`main.py` also accepts image frames from `ws://` or `wss://` sources through the same `ip` field:

```bash
.venv/bin/python main.py --ip ws://192.168.15.2:8765/video --config config_da3_live.yaml
```

or in YAML:

```yaml
ip: ws://192.168.15.2:8765/video
```

The default WebSocket mode expects each RGB frame as a JPEG/PNG image. It also accepts complete MP4 payloads and extracts the first frame, but this is not recommended for low-latency live use. Supported payloads are binary JPEG/PNG/MP4 frames, text base64, data URLs such as `data:image/jpeg;base64,...` or `data:video/mp4;base64,...`, or JSON text with an `image`, `frame`, `video`, or `data` field containing base64 image/video data.

For a continuous MPEG-1 stream, enable the MPEG-1 WebSocket decoder:

```yaml
ip: ws://192.168.15.2:8765/video
ws_media_format: mpeg1video
ws_mpeg_format: mpegvideo   # raw MPEG-1 elementary stream
```

This mode requires `ffmpeg` available in `PATH`. If your server sends MPEG-TS containing MPEG-1 video, use `ws_mpeg_format: mpegts`; for a raw `mpeg1video` elementary stream, keep `mpegvideo`.

## YAML Configuration

`main.py` accepts:

```bash
.venv/bin/python main.py --config config_da3_live.yaml
```

CLI arguments override YAML values:

```bash
.venv/bin/python main.py --config config_da3_live.yaml --input_size 384
```

YAML values use the same names as CLI arguments without `--`.

## Browser Point Cloud Streaming

`main.py` can publish the live point cloud to browser clients over a separate WebSocket server:

```yaml
pointcloud_web: true
pointcloud_web_host: 0.0.0.0
pointcloud_web_port: 8765
pointcloud_web_every: 1
pointcloud_web_stride: 4
pointcloud_web_max_points: 120000
```

Run the pipeline normally:

```bash
.venv/bin/python main.py --config config_da3_live.yaml
```

Then open `web_pointcloud_viewer.html` in a browser and connect to:

```text
ws://127.0.0.1:8765
```

For another machine on the network, replace `127.0.0.1` with the IP of the machine running `main.py`.

The stream format is binary `Float32Array` data interleaved as:

```text
x, y, z, r, g, b, x, y, z, r, g, b, ...
```

Coordinates are in meters. Colors are normalized floats from `0.0` to `1.0`, ready for a Three.js `BufferGeometry` with `position` and `color` attributes.

## Live DA3 Large

DA3 Large provides strong relative depth and pose estimation. It is useful for live depth preview and pose-assisted point-cloud accumulation.

Run:

```bash
.venv/bin/python main.py --config config_da3_live.yaml
```

Important YAML fields:

```yaml
model_backend: da3
checkpoint: ./models/DA3-LARGE-1.1
use_gpu: true

input_size: 448
resolution_scale: 1.0
depth_output_scale: 1.0
pointcloud_compute_device: cuda

live_registration: false
registration_backend: da3_pose
```

Increase depth quality:

```yaml
input_size: 518
resolution_scale: 1.0
depth_output_scale: 1.0
```

Reduce VRAM/FPS cost:

```yaml
input_size: 384
resolution_scale: 0.75
depth_output_scale: 0.75
frame_skip: 1
```

## Live DA3Metric Large

DA3Metric Large is useful when you want to test a metric-depth DA3 variant without pose.

Run:

```bash
.venv/bin/python main.py --config config_da3metric_live.yaml
```

Current configuration opens:

- RGB/depth visualization
- Open3D point-cloud viewer
- no live registration

Relevant fields:

```yaml
model_backend: da3
checkpoint: ./models/DA3METRIC-LARGE
da3_metric_focal_scale: false
da3_metric_scale: 1.0

pointcloud: true
pointcloud_stride: 6
pointcloud_max_depth: 30
```

If the depth visualization is black, reduce scaling or keep:

```yaml
da3_metric_focal_scale: false
da3_metric_scale: 1.0
fast_vis: false
```

## Depth Anything V2 Metric

Use this when you need stable metric depth in meters.

Example command:

```bash
.venv/bin/python main.py \
  --ip 192.168.15.3:8080/video \
  --repo_path ./Depth-Anything-V2/metric_depth \
  --checkpoint ./models/depth_anything_v2_metric_vkitti_vitl.pth \
  --model_backend v2_metric \
  --encoder vitl \
  --max_depth 80 \
  --use_gpu
```

Indoor model:

```bash
--checkpoint ./models/depth_anything_v2_metric_hypersim_vitl.pth --max_depth 20
```

Outdoor model:

```bash
--checkpoint ./models/depth_anything_v2_metric_vkitti_vitl.pth --max_depth 80
```

## Video File Input

Instead of IP Webcam:

```yaml
video: ./spectacular-rec/data.mp4
# ip: 192.168.15.3:8080/video
```

Useful options:

```yaml
loop_video: true
realtime_video: false
```

Run:

```bash
.venv/bin/python main.py --config config_da3_live.yaml
```

## Image Preprocessing

The input image can be corrected before inference.

Available parameters:

```yaml
rotate: none        # none, 90cw, 90ccw, 180
wb: clahe           # none, grayworld, clahe
clahe_clip: 3.5
clahe_tile: 8
contrast: 1.12
brightness: 4
saturation: 1.08
gamma: 0.9
sharpen: 0.35
denoise: 3
```

More contrast:

```yaml
clahe_clip: 5.0
contrast: 1.2
sharpen: 0.5
gamma: 0.85
```

Less noisy:

```yaml
clahe_clip: 2.5
sharpen: 0.2
denoise: 3
```

## Point Cloud Viewer

Enable Open3D point-cloud viewer:

```yaml
pointcloud: true
pointcloud_stride: 4
pointcloud_max_depth: 30
pointcloud_compute_device: cuda
```

Dense but heavier:

```yaml
pointcloud_stride: 1
```

Light:

```yaml
pointcloud_stride: 6
```

## Live Registration Preview

Enable:

```yaml
live_registration: true
```

Available backends:

```text
open3d_ransac
teaser
robust
da3_pose
```

Backend meanings:

- `open3d_ransac`: FPFH + Open3D RANSAC + ICP
- `teaser`: FPFH correspondences + TEASER++ + ICP
- `robust`: tries TEASER++, RANSAC and ICP identity, then selects best by fitness/RMSE
- `da3_pose`: uses DA3 pose estimate for accumulation, optionally refined by ICP

DA3 pose backend:

```yaml
registration_backend: da3_pose
registration_pose_icp_refine: true
registration_every: 5
registration_stride: 1
registration_depth_scale: 1.0
registration_map_voxel: 0.01
registration_max_map_points: 1000000
```

Robust backend:

```yaml
registration_backend: robust
registration_every: 10
registration_stride: 2
registration_voxel: 0.04
registration_feature_radius: 0.20
registration_ransac_distance: 0.08
registration_icp_distance: 0.04
registration_icp_fine_distance: 0.018
registration_min_fitness: 0.08
registration_max_rmse: 0.03
```

If registration is too slow:

```yaml
registration_every: 15
registration_stride: 4
registration_map_voxel: 0.04
registration_max_map_points: 300000
```

If the preview is not dense enough:

```yaml
registration_stride: 1
registration_depth_scale: 1.0
registration_map_voxel: 0.01
registration_max_map_points: 1000000
```

## Saving RGB and Depth for SLAM

Enable synchronized save:

```yaml
save_dir: ./slam_log
save_every: 10
```

Or press `s` during live execution.

Outputs:

```text
slam_log/rgb/*.png
slam_log/depth_mm/*.png
```

Depth is saved as `uint16` millimeters.

## Saving Point Clouds

Enable:

```yaml
pointcloud_save_dir: ./pointclouds
pointcloud_save_every: 10
pointcloud_stride: 1
```

Or press `p` during live execution.

Output:

```text
pointclouds/cloud_*.ply
```

## Saving Reconstruction Dataset Samples

For offline reconstruction, `main.py` can save a synchronized package containing:

- RGB frame
- depth PNG as `uint16` millimeters
- partial point cloud as PLY
- JSON metadata with timestamp, intrinsics and optional DA3 pose

Enable automatic dataset capture:

```yaml
dataset_save_dir: ./dataset_capture
dataset_save_every: 5
dataset_save_stride: 4
dataset_save_max_depth: 30
```

Or press `d` during live execution to save one sample manually.

Output structure:

```text
dataset_capture/
  rgb/
    000000_YYYYMMDD_HHMMSS_xxxxxxxxx.png
  depth_mm/
    000000_YYYYMMDD_HHMMSS_xxxxxxxxx.png
  ply/
    000000_YYYYMMDD_HHMMSS_xxxxxxxxx.ply
  meta/
    000000_YYYYMMDD_HHMMSS_xxxxxxxxx.json
```

Metadata example:

```json
{
  "timestamp_ns": 1760000000000000000,
  "frame_index": 123,
  "infer_index": 45,
  "rgb": "rgb/000000_YYYYMMDD_HHMMSS_xxxxxxxxx.png",
  "depth_mm": "depth_mm/000000_YYYYMMDD_HHMMSS_xxxxxxxxx.png",
  "ply": "ply/000000_YYYYMMDD_HHMMSS_xxxxxxxxx.ply",
  "depth_unit": "millimeters_uint16",
  "pointcloud_stride": 4,
  "pointcloud_points": 123456,
  "camera": {
    "fx": 1450.81,
    "fy": 1450.81,
    "cx": 719.0,
    "cy": 960.0,
    "width": 1440,
    "height": 1920
  },
  "pose_world_cam": null,
  "model_backend": "da3",
  "checkpoint": "./models/DA3METRIC-LARGE"
}
```

Use a larger stride for lighter logs:

```yaml
dataset_save_stride: 6
```

Use a smaller stride for denser partial clouds:

```yaml
dataset_save_stride: 1
```

## Offline V2 Metric Fusion

Use [offline_fusion.py](offline_fusion.py) for:

```text
Depth Anything V2 Metric
-> point clouds
-> TEASER++
-> ICP
-> TSDF
```

Example:

```bash
.venv/bin/python offline_fusion.py \
  --video ./spectacular-rec/data.mp4 \
  --repo_path ./Depth-Anything-V2/metric_depth \
  --checkpoint ./models/depth_anything_v2_metric_vkitti_vitl.pth \
  --output_dir ./fusion_out \
  --model_backend v2_metric \
  --encoder vitl \
  --max_depth 80 \
  --use_gpu \
  --rotate 90cw \
  --wb grayworld \
  --input_size 384 \
  --resolution_scale 0.75 \
  --keyframe_step 5 \
  --pointcloud_stride 2 \
  --registration_voxel 0.08 \
  --feature_radius 0.35 \
  --teaser_noise_bound 0.08 \
  --teaser_max_correspondences 5000 \
  --icp_distance 0.05 \
  --tsdf_voxel 0.02 \
  --tsdf_sdf_trunc 0.08
```

Outputs:

```text
fusion_out/mesh.ply
fusion_out/fused_cloud.ply
fusion_out/trajectory.txt
```

## Offline DA3 Large Pipeline

Use [da3_large_pipeline.py](da3_large_pipeline.py) when you want DA3 Large depth/pose in batch.

Example:

```bash
.venv/bin/python da3_large_pipeline.py \
  --video ./spectacular-rec/data.mp4 \
  --output_dir ./da3_large_fusion \
  --model depth-anything/DA3-LARGE-1.1 \
  --device cuda \
  --rotate 90cw \
  --wb grayworld \
  --frame_step 1 \
  --max_frames 337 \
  --da3_batch_size 8 \
  --process_res 384 \
  --max_depth 80 \
  --pointcloud_stride 2 \
  --preview_voxel 0.04 \
  --icp_refine \
  --icp_strict \
  --tsdf
```

Memory-safe knobs:

```bash
--da3_batch_size 4
--process_res 336
--pointcloud_stride 4
```

Dense export:

```bash
--dense_cloud --dense_max_points 3000000
```

## DA3 3D Gaussian Splatting

[da3_gs_render.py](da3_gs_render.py) exports DA3 Gaussian Splatting when the model has a Gaussian branch.

Important: `DA3-LARGE-1.1` does not expose the 3DGS branch. Use a DA3 model with `gs_head/gs_adapter`, such as `DA3-GIANT-1.1`, if your GPU has enough memory.

Example:

```bash
.venv/bin/python da3_gs_render.py \
  --video ./spectacular-rec/data.mp4 \
  --output_dir ./da3_gs_out \
  --model depth-anything/DA3-GIANT-1.1 \
  --device cuda \
  --rotate 90cw \
  --wb grayworld \
  --fps_sample 1 \
  --max_frames 12 \
  --process_res 384 \
  --gs_video_quality low \
  --gs_chunk_size 1
```

Outputs:

```text
da3_gs_out/gs_ply/
da3_gs_out/gs_video/
```

## Runtime Keys

In `main.py`:

```text
q or ESC  quit
s         save synchronized RGB + depth, if save_dir is set
p         save current point cloud, if pointcloud_save_dir is set
d         save synchronized RGB + depth + partial PLY + metadata, if dataset_save_dir is set
```

## Performance Tips

Improve FPS:

```yaml
input_size: 384
resolution_scale: 0.75
depth_output_scale: 0.75
display_scale: 0.5
display_every: 2
frame_skip: 1
fast_vis: true
```

Improve quality:

```yaml
input_size: 518
resolution_scale: 1.0
depth_output_scale: 1.0
fast_vis: false
```

Reduce VRAM:

```yaml
input_size: 384
cuda_empty_cache_every: 20
registration_max_map_points: 300000
```

Move point-cloud projection to CUDA:

```yaml
pointcloud_compute_device: cuda
```

Note: Open3D legacy registration, FPFH, RANSAC, TEASER++ and the Open3D visualizer are still CPU-bound.

## Troubleshooting

### `Provide --ip for IP Webcam or --video`

Your YAML probably did not include either:

```yaml
ip: 192.168.15.3:8080/video
```

or:

```yaml
video: ./path/to/video.mp4
```

### Depth visualization is black

Depth values may be much larger than `max_depth` or scaling is wrong.

Try:

```yaml
fast_vis: false
max_depth: 80
da3_metric_focal_scale: false
da3_metric_scale: 1.0
```

### CUDA out of memory

Lower:

```yaml
input_size: 384
resolution_scale: 0.75
depth_output_scale: 0.75
registration_max_map_points: 300000
```

For offline DA3:

```bash
--da3_batch_size 4 --process_res 336
```

### Point cloud is sparse

Lower stride:

```yaml
pointcloud_stride: 1
registration_stride: 1
```

But this increases CPU/GPU memory.

### Registration is unstable

Use `da3_pose` if running DA3 Large:

```yaml
registration_backend: da3_pose
registration_pose_icp_refine: true
```

Or use robust:

```yaml
registration_backend: robust
registration_min_fitness: 0.04
registration_max_rmse: 0.05
```

### TEASER++ not found

Make sure this works in the active venv:

```bash
.venv/bin/python -c "import teaserpp_python; print('ok')"
```

If not, rebuild/install TEASER++ Python bindings.

## Notes on Metric Scale

- Depth Anything V2 Metric checkpoints are the safest option for metric depth in meters.
- DA3 Large is modern and provides pose/depth, but its depth is not necessarily absolute metric for all use cases.
- DA3Metric Large may require scale calibration depending on focal handling and camera setup.
- For SLAM/TSDF work, always validate scale against a known measurement.

## GitHub Hygiene

Do not commit large model weights or generated outputs. Recommended `.gitignore` entries:

```gitignore
.venv/
models/
*.mp4
fusion_out/
da3_large_fusion/
da3_gs_out/
pointclouds/
dataset_capture/
slam_log/
__pycache__/
```

## License

This repository combines local code with external models and third-party repositories. Check each upstream project/model license before commercial use:

- Depth Anything V2
- Depth Anything 3
- TEASER++
- Open3D
- gsplat

Some DA3 models are non-commercial. Review the Hugging Face model card for the exact model you use.
