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
- `r1_mode_check.py`: monitor the R1 FSM, sport state, and low-level mode
- `r1_keyboard_control.py`: control R1 locomotion from the keyboard
- `r1_wave_hand.py`: simple motion example
- `r1_face_recognition.py`: face recognition experiment
- `r1_neck_keyboard_control.py`: control neck pitch/yaw through `rt/arm_sdk`
- `r1_arm_keyboard_control.py`: control the ten arm joints through `rt/arm_sdk`

## R1 `rt/arm_sdk` Control Scope

`rt/arm_sdk` can control 13 R1 upper-body joints:

```less
rt/arm_sdk
├── Tay trái: 5 joints
├── Tay phải: 5 joints
├── Eo: 1 joint
└── Cổ: 2 joints
```

The corresponding `LowCmd.motor_cmd` indices are:

- Left arm: `15`, `16`, `17`, `18`, `19`
- Right arm: `22`, `23`, `24`, `25`, `26`
- Waist yaw: `13`
- Head pitch/yaw: `29`, `30`

On R1, `LowCmd.mode_pr` is the shared `rt/arm_sdk` blend weight, represented as
an integer percentage from `0` (locomotion controller owns the upper body) to
`100` (`arm_sdk` owns the upper body). Because the weight applies to all 13
joints, a controller must initialize every joint from `rt/lowstate`, even when
it intends to move only the neck or one arm joint.

### `r1_neck_keyboard_control.py`

Control the R1 neck through `rt/arm_sdk` while the robot stays in locomotion
mode (FSM 811). The script checks the FSM before taking control and does not
change the locomotion mode.

```bash
python3 my_script/r1_neck_keyboard_control.py <network_interface>
```

Example:

```bash
python3 my_script/r1_neck_keyboard_control.py enxa0cec86d95d6
```

Use the arrow keys to move pitch/yaw, `C` or `0` to center, and `Q` or `ESC`
to quit. On exit, the script ramps the `arm_sdk` weight to zero so the
locomotion controller regains the neck smoothly.

The R1-specific `arm_sdk` weight is carried in `LowCmd.mode_pr` as an integer
percentage. The script also holds all 13 R1 upper-body SDK joints at their
measured startup positions while the neck is controlled.

### `r1_arm_keyboard_control.py`

Control the ten R1 arm joints individually through `rt/arm_sdk` while the robot
remains in locomotion mode (FSM 811):

```bash
/usr/bin/python3 my_script/r1_arm_keyboard_control.py <network_interface>
```

Keys `1`-`5` select a left-arm joint and `6`-`0` select a right-arm joint.
`Up`/`Down` changes the selected target, `Left`/`Right` changes the selected
joint, `C` restores the startup arm pose, and `Q`/`ESC` exits. For initial
testing, keep the robot standing still and keep obstacles away from both arms.

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
