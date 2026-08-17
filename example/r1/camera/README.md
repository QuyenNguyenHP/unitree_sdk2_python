# R1 object detection with depth

This example combines the R1 EDU left RGB stream with the depth product made
available by Unitree's stereo service. It runs on the NX expansion dock and
shows the median object depth in metres.

## Robot requirements

- Patch >= `1.0.1.17`
- Stereo patch PC1 >= `0.4.0.1`
- System patch PC4 >= `2.0.0.0`
- R1 push service and Stereo patch PC1 service enabled in the Unitree app
- `video_hub` disabled in the app while the push service is active

Enable the depth receiver on the NX:

```bash
sudo unitree_depth_start
```

Check the RGB stream before running detection:

```bash
gst-launch-1.0 udpsrc port=5002 ! \
  application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! \
  rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

The Python OpenCV build must also have GStreamer enabled:

```bash
python3 -c "import cv2; print('GStreamer: YES' in cv2.getBuildInformation())"
```

If this prints `False`, use the GStreamer-enabled OpenCV supplied for the NX or
build OpenCV with GStreamer support. The standard PyPI OpenCV wheel may not
include it.

Install the detector and run the example:

```bash
python3 -m pip install ultralytics
python3 example/r1/camera/object_distance.py --model yolo11n.pt
```

The first Ultralytics run may download model weights. For an offline robot,
download the `.pt` file elsewhere, copy it to the NX, and pass its local path
with `--model`.

Press `q` or Escape to stop. Disable the depth receiver when finished:

```bash
sudo unitree_depth_stop
```

## Accuracy note

The script assumes the depth map is registered pixel-for-pixel to the left RGB
image because both Unitree products are documented at `544x448`. Verify this on
your robot by placing an object with clear edges at several distances. If its
depth silhouette is offset from the RGB image, obtain the stereo calibration
from Unitree and remap the depth image before indexing it with detector boxes.

Depth is sampled only from the centre 40% of each detection box and a median is
used to reduce background and edge noise. This reports optical-axis depth, not
the full Euclidean distance from the camera. It is appropriate for perception,
but not a safety-rated measurement.
