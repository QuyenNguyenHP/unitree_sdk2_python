# my_script

Custom helper scripts for working with the Unitree SDK in this repository.

## Audio Scripts

### `text_to_robot.py`

Use the robot's built-in TTS service through `TtsMaker(...)`.

Command:

```bash
python3 my_script/text_to_robot.py <network_interface> "Your text here"
```

Example:

```bash
python3 my_script/text_to_robot.py enxa0cec86d95d6 "Ni hao"
```

Optional flags:

```bash
python3 my_script/text_to_robot.py enxa0cec86d95d6 "Hello" --volume 85 --speaker-id 0 --wait 5
```

Notes:

- This depends on the robot's own `voice` service.
- In current testing, the robot TTS appears to work better for Chinese than English.

### `english_text_to_robot.py`

Use Ubuntu to generate English speech first, then stream the generated audio to the robot speaker.

Command:

```bash
python3 my_script/english_text_to_robot.py <network_interface> "Your English text here"
```

Example:

```bash
python3 my_script/english_text_to_robot.py enxa0cec86d95d6 "Hello, I am Unitree robot."
```

Ubuntu packages:

```bash
sudo apt update
sudo apt install libttspico-utils ffmpeg
```

Notes:

- The script prefers `pico2wave` first, then falls back to `espeak-ng` or `espeak`.
- The generated audio is normalized to `16kHz`, `mono`, `16-bit PCM` before streaming.
- The script waits for playback to finish before sending `PlayStop()`.

## Other Scripts

- `list_r1_topic.py`: inspect available topics
- `r1_wave_hand.py`: simple motion example
- `r1_face_recognition.py`: face recognition experiment

### `video_robot_face_detection.py`

Local face detection video test script.

What it does:

- Connects to the robot video service
- Reads camera frames from the robot
- Detects faces in each frame
- Draws face bounding boxes in an OpenCV window

Command:

```bash
python3 my_script/video_robot_face_detection.py <network_interface> [yunet_model.onnx]
```

Examples:

```bash
python3 my_script/video_robot_face_detection.py enxa0cec86d95d6
python3 my_script/video_robot_face_detection.py enxa0cec86d95d6 ./face_detection_yunet_2023mar.onnx
```

Notes:

- The script prefers YuNet if `face_detection_yunet_2023mar.onnx` is available.
- If YuNet is not available, it falls back to OpenCV Haar Cascade.
- Press `ESC` to close the video window.
