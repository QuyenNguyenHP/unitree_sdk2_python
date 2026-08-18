# R1 Helper Scripts

Custom monitoring, motion, audio, and vision tools for the Unitree R1 in this
SDK checkout. This README covers every Python script currently present in
`my_script/`; generated files under `__pycache__/` are not user-facing tools.

## Before Running a Script

Run commands from the repository root unless a command says otherwise:

```bash
cd /home/r1-edu/unitree_sdk2_python
```

Use the network interface connected to the robot. The examples below use
`eth10`; replace it when your interface has another name:

```bash
ip -br addr
```

The interface must be `UP` and normally has an address in the robot's
`192.168.123.0/24` network. Use a Python environment with this repository and
CycloneDDS installed. `/usr/bin/python3` is used below because it is the known
working interpreter on this machine.

Motion scripts control real hardware. Secure the robot when appropriate, clear
the surrounding area, keep the remote ready for emergency damping/stop, and do
not run two motion controllers at the same time.

## Script Inventory

| Script | Purpose | Robot movement |
|---|---|---|
| `battery_check.py` | Monitor BMS charge, voltage, current, and temperature data | No |
| `list_r1_topic.py` | Discover visible `rt/...` DDS topics | No |
| `r1_mode_check.py` | Monitor FSM, sport state, velocity, and low-level mode | No |
| `r1_joint_positions.py` | Capture one read-only snapshot of all 26 joint positions | No |
| `r1_action.py` | List and execute firmware-provided arm gestures | Yes |
| `r1_keyboard_control.py` | Keyboard locomotion and lie/stand control | Yes |
| `r1_arm_keyboard_control.py` | Control ten individual arm joints through `rt/arm_sdk` | Yes |
| `r1_neck_keyboard_control.py` | Control neck pitch/yaw through `rt/arm_sdk` | Yes |
| `text_to_robot.py` | Speak text with the robot's built-in TTS service | No |
| `english_text_to_robot.py` | Generate English TTS locally and stream it to the robot | No |
| `led_control.py` | Set the robot LED strip to a chosen RGB color | No |
| `video_robot_face_detection.py` | Detect faces in the R1 camera stream | No |
| `r1_face_recognition.py` | Recognize known people in the R1 camera stream | No |

## Monitoring and Discovery

### `r1_joint_positions.py`

Capture one `rt/lowstate` snapshot and print all 26 physical joint positions in
radians and degrees:

```bash
/usr/bin/python3 my_script/r1_joint_positions.py eth10
```

The script only subscribes to robot state, sends no command, and works in any
FSM mode. Normal real-hardware safety precautions still apply.

### `battery_check.py`

Subscribe to the R1 BMS topic and print SOC, SOH, current, pack/cell voltages,
cell spread, cycle count, and raw temperature values:

```bash
/usr/bin/python3 my_script/battery_check.py eth10
```

Options:

```bash
/usr/bin/python3 my_script/battery_check.py eth10 \
  --topic rt/lf/bmsstate --interval 2
```

- Default topic: `rt/lf/bmsstate`
- Default display interval: `1` second
- Stop with `Ctrl+C`.
- Temperature scaling is firmware-dependent, so the script prints it raw.

### `r1_mode_check.py`

Monitor the R1 FSM plus `rt/sportmodestate` and `rt/lowstate`:

```bash
/usr/bin/python3 my_script/r1_mode_check.py eth10
/usr/bin/python3 my_script/r1_mode_check.py eth10 --interval 2
```

The report includes the FSM ID/name, sport mode, gait, velocity, yaw speed,
sport error code, `mode_machine`, and `mode_pr`. The default interval is five
seconds. Stop with `Ctrl+C`.

Known FSM names used by the script are:

- `0`: zero torque
- `1`: damping
- `4`: stance
- `701`: lie to stand
- `702`: stand to lie
- `811`: locomotion

### `list_r1_topic.py`

Run `cyclonedds ps` on a selected interface and extract visible `rt/...` topic
names:

```bash
/usr/bin/python3 my_script/list_r1_topic.py eth10
/usr/bin/python3 my_script/list_r1_topic.py eth10 0
```

The optional second argument is the DDS domain ID; its default is `0`. This
script requires the `cyclonedds` command-line program to be on `PATH`. It scans
for up to eight seconds and does not publish commands.

## High-Level Motion

### `r1_action.py`

Read the available actions directly from the connected R1 `arm` service and
start an interactive action menu:

```bash
/usr/bin/python3 my_script/r1_action.py eth10
```

Type `YES` once to enable execution. At the prompt, enter an action ID, exact
firmware name, or supported alias. Enter `list` to print the list again and `q`
to quit.

```text
Type YES to enable interactive action execution: YES
Current FSM: 811

Action ID or name [list/q]: 27
Executing arm action 27: shake_hand...
ExecuteAction(27) result: 0
Waiting 8 seconds before accepting another action...
Ready for the next action.

Action ID or name [list/q]: 18
```

The service does not expose an action-completion status API. The script waits
eight seconds before accepting the next action; adjust the delay for longer
motions:

```bash
/usr/bin/python3 my_script/r1_action.py eth10 --action-wait 12
```

List actions without moving the robot:

```bash
/usr/bin/python3 my_script/r1_action.py eth10 list
```

Execute one action and exit:

```bash
/usr/bin/python3 my_script/r1_action.py eth10 wave
/usr/bin/python3 my_script/r1_action.py eth10 shake_hand
/usr/bin/python3 my_script/r1_action.py eth10 clap
/usr/bin/python3 my_script/r1_action.py eth10 18
```

Useful options:

- `--action-wait SECONDS`: delay after an accepted action; default `8`.
- `--timeout SECONDS`: RPC timeout; default `10`.
- `--yes`: skip the safety confirmation.
- `--start-locomotion`: explicitly request FSM 811 if the secured robot is not
  already in locomotion.
- `--settle-time SECONDS`: wait after requesting locomotion; default `3`.

The action list is refreshed before execution. The tested R1 firmware
advertised:

- `11`: `blow_kiss_with_both_hands`
- `12`: `blow_kiss_with_left_hand`
- `13`: `blow_kiss_with_right_hand`
- `15`: `both_hands_up`
- `17`: `clamp` (`clap` alias)
- `18`: `high_five`
- `19`: `hug`
- `22`: `refuse`
- `23`: `right_hand_up`
- `24`: `ultraman_ray`
- `25`: `wave_under_head`
- `26`: `wave_above_head` (`wave` alias)
- `27`: `shake_hand` (`handshake` alias)
- `28`: `box_left_hand_win`
- `29`: `box_right_hand_win`
- `30`: `box_both_hand_win`
- `31`: `extend_right_arm_forward`
- `33`: `right_hand_on_heart`
- `34`: `both_hands_up_deviate_right`
- `35`: `emphasize`
- `36`: `forward_push`
- `99`: `release_arm` (`release` alias)

`release_arm` is not a second handshake stage. It releases or returns arm
action control and may not produce visible movement.

### `r1_keyboard_control.py`

Control locomotion from an interactive terminal:

```bash
/usr/bin/python3 my_script/r1_keyboard_control.py eth10
```

The script requires the confirmation text `MOVE`. Controls:

- `P`: enter locomotion mode.
- `W` / `S`: move forward/backward.
- `A` / `D`: move left/right.
- `R` / `E`: turn left/right.
- `L`: stance, then lie-to-stand after three seconds.
- `K`: stand-to-lie.
- `Space` or `X`: stop immediately.
- `Q` or `Esc`: stop and quit.

Each movement key requests one second of motion. Default speeds are `0.5 m/s`
linear, `0.4 m/s` lateral, and `0.8 rad/s` turning. Override them with
`--linear-speed`, `--lateral-speed`, and `--turn-speed`.

## Upper-Body `rt/arm_sdk` Control

On this R1, `rt/arm_sdk` controls 13 upper-body joints:

```text
Left arm:   5 joints, LowCmd indices 15-19
Right arm:  5 joints, LowCmd indices 22-26
Waist yaw:  1 joint,  LowCmd index 13
Head:       2 joints, LowCmd indices 29-30
```

`LowCmd.mode_pr` is a shared blend weight from `0` (locomotion owns the upper
body) to `100` (`arm_sdk` owns it). The keyboard controllers initialize all 13
joints from feedback before raising this weight and ramp it back to zero when
exiting. Both scripts require FSM 811 and do not change the locomotion FSM.

### `r1_neck_keyboard_control.py`

```bash
/usr/bin/python3 my_script/r1_neck_keyboard_control.py eth10
```

Type `NECK` to confirm. Controls:

- `Up` / `Down`: increase/decrease pitch.
- `Left` / `Right`: increase/decrease yaw.
- `C` or `0`: center both targets.
- `Q` or `Esc`: release control and quit.

Defaults are a `5°` key step, `±30°` pitch target limit, `±60°` yaw target
limit, and `45°/s` target slew rate. Configure them with `--step-deg`,
`--pitch-limit-deg`, `--yaw-limit-deg`, and `--max-speed-deg-s`.

### `r1_arm_keyboard_control.py`

```bash
/usr/bin/python3 my_script/r1_arm_keyboard_control.py eth10
```

Type `ARMS` to confirm. Controls:

- `1`-`5`: directly select a left-arm joint.
- `6`-`9`, `0`: directly select a right-arm joint.
- `Left` / `Right`: select the previous/next joint.
- `Up` / `Down`: change the selected target.
- `C`: return every arm joint to its startup angle.
- `Q` or `Esc`: release control and quit.

Defaults are a `3°` key step, maximum `±30°` travel from each startup angle,
and `30°/s` target slew rate. Configure them with `--step-deg`, `--travel-deg`,
and `--max-speed-deg-s`.

## Audio

The Unitree `AudioClient` talks to the robot `voice` service over DDS RPC. In
this repository it is used for:

- `TtsMaker(text, speaker_id)`: built-in robot text-to-speech. Use
  `speaker_id 0` for Chinese and `speaker_id 1` for English.
- `GetVolume()` / `SetVolume(volume)`: read or change the system speaker volume
  in the `0`-`100` range.
- `LedControl(R, G, B)`: set the LED strip color with `0`-`255` red, green,
  and blue values. Keep more than `200 ms` between calls.
- `PlayStream(app_name, stream_id, pcm_data)`: stream PCM audio to the robot.
  The required format is `16 kHz`, mono, `16-bit` PCM.
- `PlayStop(app_name)`: stop playback started by `PlayStream(...)`.

### `text_to_robot.py`

Send text to the robot's built-in `TtsMaker(...)` service:

```bash
/usr/bin/python3 my_script/text_to_robot.py eth10 "Ni hao"
/usr/bin/python3 my_script/text_to_robot.py eth10 "Hello" \
  --volume 85 --speaker-id 0 --wait 5
```

- `--volume` accepts `0`-`100`.
- `--speaker-id` defaults to `0`.
- `--wait` keeps the RPC client alive after submission; default `5` seconds.
- Built-in TTS has worked better for Chinese than English in local testing.

### `english_text_to_robot.py`

Generate English speech on Ubuntu, normalize it to robot-compatible audio, and
stream it to the speaker:

```bash
/usr/bin/python3 my_script/english_text_to_robot.py eth10 \
  "Hello, I am a Unitree robot."
```

Install at least one supported TTS backend plus `ffmpeg`:

```bash
sudo apt update
sudo apt install libttspico-utils ffmpeg
```

The script prefers `pico2wave`, then falls back to `espeak-ng` or `espeak` if
installed. It normalizes output to 16 kHz, mono, 16-bit PCM, waits for playback,
and calls `PlayStop()` before exiting.

### `led_control.py`

Set the robot LED strip to a specific RGB color:

```bash
/usr/bin/python3 my_script/led_control.py eth10
/usr/bin/python3 my_script/led_control.py eth10 --color red
/usr/bin/python3 my_script/led_control.py eth10 --r 255 --g 128 --b 0
```

When no color arguments are passed, the script shows a terminal menu and lets
the user choose a basic color with keys `1` to `9`.

Useful options:

- `--color`: one of `off`, `white`, `red`, `green`, `blue`, `yellow`, `cyan`,
  `magenta`, or `orange`.
- `--r`, `--g`, `--b`: manual RGB values from `0` to `255`. If any of these
  are set, all three must be provided.
- `--hold`: keep the script alive after sending the command; default `0.3`
  seconds.

Examples:

```bash
/usr/bin/python3 my_script/led_control.py eth10 --color cyan --hold 1
/usr/bin/python3 my_script/led_control.py eth10 --r 32 --g 32 --b 255
```

## Vision

Both vision scripts need a graphical session for `cv2.imshow`, OpenCV, NumPy,
and access to the robot's `videohub` service. Press `Esc` to close the window.

### `video_robot_face_detection.py`

Detect faces without identifying people:

```bash
/usr/bin/python3 my_script/video_robot_face_detection.py eth10
/usr/bin/python3 my_script/video_robot_face_detection.py eth10 \
  ./face_detection_yunet_2023mar.onnx
```

The script uses YuNet when OpenCV supports `FaceDetectorYN` and the supplied
ONNX model exists. Otherwise, it falls back to OpenCV's built-in Haar Cascade.

### `r1_face_recognition.py`

Recognize people using locally stored reference images:

```bash
/usr/bin/python3 my_script/r1_face_recognition.py eth10 ./faces
```

The default reference directory is `faces` in the current working directory.
Organize it by person:

```text
faces/
├── Alice/
│   ├── 1.jpg
│   └── 2.jpg
└── Bob/
    └── 1.jpg
```

Install the additional dependency in the same Python environment:

```bash
python3 -m pip install face_recognition
```

The script loads the first detected face encoding from each reference image,
uses a recognition tolerance of `0.45`, labels matches in the video window, and
prints recognized names in the terminal.
