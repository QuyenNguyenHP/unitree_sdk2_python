# Robot AI Voice (Jarvis)

A wake-word voice assistant for the Unitree G1 robot.

1. **Blue LED**: waiting for the `Jarvis` wake word.
2. **Red LED**: recording the user's question.
3. **Orange LED**: waiting for the AI server.
4. **Green LED**: playing the answer through the robot speaker.
5. On exit, the program stops audio playback and turns off the LED.

The colors can be changed in `config.json`. The program continuously reapplies
the active state's color so the robot cannot leave it overwritten by another
automatic LED state. `led_refresh_seconds` controls the refresh interval and
defaults to one second. It must be greater than 0.21 seconds because the robot
requires more than 200 ms between `LedControl` calls.

## Installation

The required `unitree_sdk2py` source and openWakeWord models are included in
this folder. You do not need to clone the Unitree SDK separately.

After copying this folder to the target computer, run:

```bash
cd /path/to/robot-Ai-voice
chmod +x install.sh
./install.sh
```

The installer deliberately installs `openwakeword==0.6.0` with `--no-deps`.
That release declares `tflite-runtime`, which has no Linux wheel for Python
3.12. This application uses ONNX Runtime, not TensorFlow Lite, and all packages
required by its ONNX path are listed in `requirements.txt`.

To install manually, use these two commands in this order:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
python3 -m pip install --user --break-system-packages --no-deps openwakeword==0.6.0
```

If pip cannot find the native CycloneDDS library, install CycloneDDS 0.10.x,
set `CYCLONEDDS_HOME` to its installation directory, and run the pip command
again. CycloneDDS is a native system dependency and cannot be bundled as pure
Python source.

## Wake-word models

The Jarvis models are stored inside the application folder:

```text
robot-Ai-voice/models
```

The folder already contains the models required for offline deployment. If
`hey_jarvis_v0.1.onnx`, `melspectrogram.onnx`, or `embedding_model.onnx` is
missing, the application automatically downloads the official openWakeWord
model bundle at startup. Internet access is required only for that download.

The `model_dir` setting is resolved relative to `config.json`, so it does not
depend on a username or installation path.

## Configuration

Edit `config.json` before starting the application:

- `network_interface`: network interface connected to the robot, such as
  `eth0` or `enp3s0`.
- `server`: URL of the voice AI server.
- `api_key`: leave it as an empty string when the server does not require a key.
- `device`: microphone exposed through PortAudio. The default is `pipewire`.
- `led_refresh_seconds`: how often the active LED color is reapplied. Keep this
  value greater than `0.21`; the recommended value is `1.0` second.

## Check and select a microphone

### 1. List microphones visible to the application

Run:

```bash
cd /path/to/robot-Ai-voice
python3 robot_ai_voice.py --list-devices
```

The command prints all PortAudio input and output devices. Look for a device
with at least one input channel. For example:

```text
  3 BY Y01: USB Audio (hw:2,0), ALSA (2 in, 0 out)
  8 pipewire, ALSA (64 in, 64 out)
```

Set `device` in `config.json` to either a device name or its numeric index:

```json
"device": "pipewire"
```

or:

```json
"device": 8
```

Device indexes can change after a reboot or after USB devices are reconnected.
A stable name such as `pipewire` is therefore recommended.

### 2. Select the default physical microphone with PipeWire

The program records at 16 kHz, mono. Some USB microphones only accept 44.1 or
48 kHz directly. Using the `pipewire` device allows PipeWire to perform the
sample-rate and channel conversion.

List PipeWire devices:

```bash
wpctl status
```

Under `Sources`, the device marked with `*` is the current default microphone:

```text
Sources:
  * 50. BY Y01 Analog Stereo
    53. Built-in Audio Analog Stereo
```

To select another microphone, use its source ID from the current output:

```bash
wpctl set-default 50
```

The number `50` is only an example. IDs may differ between computers and may
change after a reboot.

Verify the selected default microphone:

```bash
wpctl inspect @DEFAULT_AUDIO_SOURCE@ | grep -E 'node.description|node.name|alsa.card_name'
```

Then keep this setting in `config.json`:

```json
"device": "pipewire"
```

### 3. Test microphone recording

If PipeWire utilities are installed, record a short sample:

```bash
pw-record --rate 16000 --channels 1 /tmp/microphone-test.wav
```

Speak for a few seconds, press `Ctrl+C`, and play the recording:

```bash
pw-play /tmp/microphone-test.wav
```

If `pw-record` is unavailable, use ALSA tools instead:

```bash
arecord -D pipewire -f S16_LE -r 16000 -c 1 -d 5 /tmp/microphone-test.wav
aplay /tmp/microphone-test.wav
```

If the application reports `paInvalidSampleRate`, do not select the physical
ALSA USB device directly. Select it as the PipeWire default source and use
`"device": "pipewire"` in `config.json`.

## Run

```bash
cd /path/to/robot-Ai-voice
python3 robot_ai_voice.py
```

You can override the configured robot network interface on the command line:

```bash
python3 robot_ai_voice.py enp3s0
```

The server must return uncompressed PCM WAV audio. The application converts
the response to the format required by `PlayStream`: **16 kHz, mono, signed
16-bit PCM**. It then sends chunks with the same `stream_id` so the robot can
play them continuously from its cache.
