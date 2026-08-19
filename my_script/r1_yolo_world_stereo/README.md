# R1 YOLO-World stereo distance

This pipeline runs entirely on the PC receiving the R1 left/right RTP streams:

```text
UDP 5002/5003 -> rectify -> StereoSGBM -> metric 3D -> YOLO-World -> distance
```

## Starting the stereo service

Unitree officially documents starting Stereo Patch PC1 manually in the app.
The generic `robot_state` DDS API may expose it to the PC, depending on the R1
firmware. `r1_service_control.py` is an experimental wrapper around that DDS
API; it does not modify startup files on PC1 and does not guess service names.

First identify the network interface connected to R1:

```bash
ip -br address
```

For example, this PC currently shows:

```text
enx00051b901cbf  UP  192.168.123.164/24
```

Run the tool from its directory. The `list` action is read-only:

```bash
cd /home/r1-edu/unitree_sdk2_python/my_script/r1_yolo_world_stereo
/usr/bin/python3 r1_service_control.py enx00051b901cbf list
```

Expected table format:

```text
SERVICE                                    STATUS   PROTECTED
some_service                               0        no
another_service                            1        yes
```

Tested on this R1 on 2026-08-18: the robot returned the following exact,
unprotected entry:

```text
stereo_patch_pc1                           0        no
```

Unitree does not document `ServiceList.status` as an on/off Boolean, so do not
interpret `0` by itself as stopped or started. Verify the resulting UDP stream.
For this robot, the concrete start command is:

```bash
/usr/bin/python3 r1_service_control.py \
  enx00051b901cbf on stereo_patch_pc1
```

Search the returned list for likely stereo/video services, but do not use a
guessed name:

```bash
/usr/bin/python3 r1_service_control.py enx00051b901cbf list \
  | grep -Ei 'stereo|video|push'
```

If the full list contains an unprotected stereo/push service, copy its exact
case-sensitive name and switch it on:

```bash
/usr/bin/python3 r1_service_control.py \
  enx00051b901cbf on '<exact-service-name>'
```

Verify success from the video packets rather than relying only on the numeric
service status:

```bash
sudo tcpdump -ni enx00051b901cbf -c 20 \
  'udp port 5001 or udp port 5002 or udp port 5003'
```

Stop the same service when finished:

```bash
/usr/bin/python3 r1_service_control.py \
  enx00051b901cbf off stereo_patch_pc1
```

Possible outcomes:

- `ServiceList failed` or timeout: wrong interface, DDS cannot reach R1, or the
  R1 `robot_state` service is unavailable.
- Service not returned: PC1 does not expose it through this API.
- `PROTECTED=yes`: firmware explicitly prevents remote switching.
- Switch succeeds but no UDP arrives: R1 Push Service or its target IP still
  needs configuration; Stereo Patch PC1 alone may not complete the data path.

If the service is absent or protected, this firmware cannot switch it through
the public `robot_state` API. Use Unitree App -> Settings -> Service Status, or
request an official PC1 control API from Unitree. Unitree also warns that
`video_hub` may conflict with the stereo push service, so do not try to run both
camera paths simultaneously.

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
