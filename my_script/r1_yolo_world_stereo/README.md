# R1 YOLO-World stereo distance

This pipeline runs entirely on the PC receiving the R1 left/right RTP streams:

```text
UDP 5002/5003 -> rectify -> StereoSGBM -> metric 3D -> YOLO-World -> distance
```

## Starting the stereo service

Unitree officially documents starting Stereo Patch PC1 manually in the app.
The generic `robot_state` DDS API may expose it to the PC, depending on the R1
firmware. List services first (this makes no changes):

```bash
/usr/bin/python3 r1_service_control.py <network-interface> list
```

If the list contains an unprotected stereo/push service, use its exact name:

```bash
/usr/bin/python3 r1_service_control.py <network-interface> on '<exact-name>'
```

If the service is absent or protected, this firmware cannot switch it through
the public `robot_state` API; use the Unitree app or request an official PC1
control API from Unitree. Do not guess service names.

## Requirements

Confirm both streams reach the PC:

```bash
sudo tcpdump -ni any 'udp port 5002 or udp port 5003'
```

The programs automatically use OpenCV's GStreamer backend when available. On
systems where OpenCV reports `GStreamer: NO`, they run `gst-launch-1.0` as a
decoder and read its raw BGR output, so rebuilding OpenCV is unnecessary. Force
this fallback when diagnosing capture with `--capture-backend gstreamer-cli`.

Install YOLO-World support in the same Python environment. Do not replace a
working GStreamer-enabled OpenCV build with a PyPI OpenCV wheel:

```bash
/usr/bin/python3 -m pip install --user ultralytics
```

YOLO-World weights and its text encoder may be downloaded on first use. For an
offline PC, download/cache them in advance.

## Metric stereo calibration

Print a flat chessboard and accurately measure one square in metres. The
`--columns` and `--rows` values are counts of *inner corners*, not squares.
For example, for a 10 by 7 square board, use 9 by 6 inner corners:

```bash
cd /home/r1-edu/unitree_sdk2_python/my_script/r1_yolo_world_stereo
/usr/bin/python3 r1_stereo_calibrate.py \
  --columns 9 \
  --rows 6 \
  --square-size-m 0.024 \
  --output r1_stereo_calibration.npz
```

Show the chessboard in both cameras. Press Space to capture at least 20 varied
pairs: near/far, tilted, and across the centre and all image corners. Press `c`
to solve. Recalibrate if stereo RMS is above roughly 1 pixel. Never invent the
square size: its scale determines the reported distance and baseline.

If Unitree supplies factory `K1`, `D1`, `K2`, `D2`, `R`, and `T`, those are
preferable. Store `T` in metres and use the same NPZ keys generated above.

## Run detection and distance

```bash
cd /home/r1-edu/unitree_sdk2_python/my_script/r1_yolo_world_stereo
/usr/bin/python3 r1_yolo_world_stereo_distance.py \
  --calibration r1_stereo_calibration.npz \
  --classes 'person,chair,bottle,cup,box,door' \
  --model yolov8s-worldv2.pt
```

For a CPU-only PC, start with the faster profile below. YOLO runs every second
frame while stereo distance still updates on every frame:

```bash
python r1_yolo_world_stereo_distance.py \
  --calibration r1_stereo_calibration.npz \
  --classes 'person,chair,bottle,cup,box,door' \
  --model yolov8s-worldv2.pt \
  --device cpu \
  --image-size 416 \
  --num-disparities 64 \
  --block-size 3 \
  --detect-every 2
```

Use English noun prompts for the default YOLO-World text encoder. The default
distance is optical-axis `Z`; pass `--distance-mode euclidean` for straight-line
camera-to-object distance.

Important limitations:

- OpenCV pairs frames by decoder arrival time because it does not expose RTP
  timestamps through `VideoCapture`. Keep `pair delta` small and test first on
  stationary objects. A production moving-scene system should pair native RTP
  timestamps through GStreamer.
- Stereo depth can fail on shiny, transparent, textureless, occluded, or very
  distant surfaces. `no depth` is safer than displaying an unsupported value.
- The central 40 percent of each detection is sampled and trimmed before taking
  a median to reduce foreground/background mixing.
- This is a perception estimate, not a safety-rated distance measurement.
