#!/usr/bin/env python3
"""Wake-word voice assistant that uses the Unitree speaker and RGB LED."""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import signal
import sys
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
from scipy.signal import resample_poly

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
CHANNELS = 1
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_MODEL_DIR = BASE_DIR / "models"


@dataclass(frozen=True)
class RecordingConfig:
    block_samples: int = 1_280  # 80 ms, openWakeWord's native frame size
    silence_seconds: float = 2.5
    max_seconds: float = 20.0
    min_seconds: float = 0.5
    pre_roll_seconds: float = 0.24
    silence_multiplier: float = 2.0
    min_silence_rms: float = 100.0


@dataclass(frozen=True)
class AudioResponse:
    data: bytes
    content_type: str


def rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0


def make_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm)
    return output.getvalue()


def parse_rgb(value: object, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be an array of three integers")
    rgb = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"{name} values must be between 0 and 255")
    return rgb  # type: ignore[return-value]


class RobotOutput:
    """Own the Unitree audio client and serialize LED/audio operations."""

    def __init__(
        self,
        client: AudioClient,
        colors: dict[str, tuple[int, int, int]],
        led_refresh_seconds: float = 1.0,
    ):
        if led_refresh_seconds <= 0.21:
            raise ValueError("led_refresh_seconds must be greater than 0.21")
        self.client = client
        self.colors = colors
        self.led_refresh_seconds = led_refresh_seconds
        self._last_led_call = 0.0
        self._current_color: tuple[int, int, int] | None = None
        self._current_state: str | None = None
        self._client_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._led_thread = threading.Thread(
            target=self._keep_led_state,
            name="robot-led-keepalive",
            daemon=True,
        )
        self.app_name = "robot_ai_voice"
        self._led_thread.start()

    def set_state(self, state: str, force: bool = False) -> None:
        color = self.colors[state]
        with self._client_lock:
            if not force and state == self._current_state and color == self._current_color:
                return

            # LedControl requires more than 200 ms between calls.
            elapsed = time.monotonic() - self._last_led_call
            if self._last_led_call and elapsed <= 0.21:
                time.sleep(0.21 - elapsed)
            result = self.client.LedControl(*color)
            self._last_led_call = time.monotonic()
            if result != 0:
                raise RuntimeError(f"LedControl failed for state '{state}', code={result}")
            self._current_state = state
            self._current_color = color

    def _keep_led_state(self) -> None:
        while not self._stop_event.wait(self.led_refresh_seconds):
            state = self._current_state
            if state is None:
                continue
            try:
                self.set_state(state, force=True)
            except Exception as exc:
                print(f"[WARN] LED keep-alive failed: {exc}", file=sys.stderr)

    def play_wav(self, wav_data: bytes, chunk_seconds: float = 1.0) -> None:
        pcm = wav_to_robot_pcm(wav_data)
        if not pcm:
            raise RuntimeError("The server returned an empty audio response")

        stream_id = str(uuid.uuid4())
        chunk_size = int(SAMPLE_RATE * SAMPLE_WIDTH * chunk_seconds)
        chunk_size -= chunk_size % SAMPLE_WIDTH
        for offset in range(0, len(pcm), chunk_size):
            chunk = pcm[offset : offset + chunk_size]
            with self._client_lock:
                result = self.client.PlayStream(self.app_name, stream_id, chunk)
            code = result[0] if isinstance(result, tuple) else result
            if code != 0:
                with self._client_lock:
                    self.client.PlayStop(self.app_name)
                raise RuntimeError(f"PlayStream failed at byte {offset}, code={code}")
            # Pace the stream so the robot cache is not flooded.
            if offset + len(chunk) < len(pcm):
                time.sleep(len(chunk) / (SAMPLE_RATE * SAMPLE_WIDTH))

        # Let the last cached block finish before changing the LED/state.
        time.sleep(len(pcm[offset:]) / (SAMPLE_RATE * SAMPLE_WIDTH) + 0.1)
        with self._client_lock:
            self.client.PlayStop(self.app_name)

    def stop(self) -> None:
        self._stop_event.set()
        try:
            with self._client_lock:
                self.client.PlayStop(self.app_name)
        finally:
            self.set_state("off")
            self._led_thread.join(timeout=self.led_refresh_seconds + 1.0)


class WakeWordDetector:
    def __init__(self, model: str, model_dir: Path, threshold: float):
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "openWakeWord is missing; run ./install.sh or install it with: "
                "python3 -m pip install --user --break-system-packages "
                "--no-deps openwakeword==0.6.0"
            ) from exc

        model_path = Path(model)
        if not model_path.is_file():
            exact = model_dir / f"{model}.onnx"
            candidates = sorted(glob.glob(str(model_dir / f"{model}_v*.onnx")))
            if exact.is_file():
                model_path = exact
            elif candidates:
                model_path = Path(candidates[-1])
        if not model_path.is_file():
            raise RuntimeError(f"Wake-word model '{model}' was not found in {model_dir}")

        self.model_name = model_path.stem
        self.threshold = threshold
        self._model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            melspec_model_path=str(model_dir / "melspectrogram.onnx"),
            embedding_model_path=str(model_dir / "embedding_model.onnx"),
        )

    def detected(self, pcm: bytes) -> tuple[bool, float]:
        prediction = self._model.predict(np.frombuffer(pcm, dtype="<i2"))
        score = float(prediction.get(self.model_name, max(prediction.values(), default=0.0)))
        return score >= self.threshold, score

    def reset(self) -> None:
        self._model.reset()


def ensure_wake_word_models(model: str, model_dir: Path) -> None:
    """Download the official openWakeWord model bundle when it is incomplete."""
    model_path = Path(model).expanduser()
    if model_path.is_file():
        return

    required_feature_models = (
        model_dir / "melspectrogram.onnx",
        model_dir / "embedding_model.onnx",
    )
    wake_model_exists = (model_dir / f"{model}.onnx").is_file() or any(
        model_dir.glob(f"{model}_v*.onnx")
    )
    if wake_model_exists and all(path.is_file() for path in required_feature_models):
        return

    print(
        f"[SETUP] Model '{model}' is missing. Downloading it to {model_dir}...",
        flush=True,
    )
    try:
        from openwakeword.utils import download_models

        model_dir.mkdir(parents=True, exist_ok=True)
        download_models([model], target_directory=str(model_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Could not download model '{model}' to '{model_dir}'. "
            "Check the Internet connection and try again."
        ) from exc

    wake_model_exists = (model_dir / f"{model}.onnx").is_file() or any(
        model_dir.glob(f"{model}_v*.onnx")
    )
    missing = [str(path.name) for path in required_feature_models if not path.is_file()]
    if not wake_model_exists or missing:
        details = ", ".join(missing) if missing else f"{model}.onnx"
        raise RuntimeError(f"Model download is incomplete; still missing: {details}")
    print("[SETUP] Model download completed.", flush=True)


class Microphone:
    def __init__(self, device: str | int | None, block_samples: int):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is missing; install requirements.txt") from exc
        self.sd = sd
        self.device = device
        self.block_samples = block_samples
        self.stream = None

    def __enter__(self) -> "Microphone":
        self.stream = self.sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=self.block_samples,
            device=self.device,
            channels=CHANNELS,
            dtype="int16",
        )
        self.stream.start()
        return self

    def __exit__(self, *_args: object) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def blocks(self) -> Iterator[bytes]:
        if self.stream is None:
            raise RuntimeError("Microphone is not open")
        while True:
            data, overflowed = self.stream.read(self.block_samples)
            if overflowed:
                print("[WARN] Microphone input overflow", file=sys.stderr)
            yield bytes(data)


def record_utterance(blocks: Iterator[bytes], noise_rms: float, cfg: RecordingConfig) -> bytes:
    frame_seconds = cfg.block_samples / SAMPLE_RATE
    silence_limit = max(1, round(cfg.silence_seconds / frame_seconds))
    max_blocks = max(1, round(cfg.max_seconds / frame_seconds))
    min_blocks = max(1, round(cfg.min_seconds / frame_seconds))
    pre_roll = deque(maxlen=max(1, round(cfg.pre_roll_seconds / frame_seconds)))
    threshold = max(cfg.min_silence_rms, noise_rms * cfg.silence_multiplier)
    captured: list[bytes] = []
    speech_started = False
    quiet_blocks = 0

    for total_blocks, block in enumerate(blocks, start=1):
        level = rms(block)
        if not speech_started:
            pre_roll.append(block)
            if level >= threshold:
                speech_started = True
                captured.extend(pre_roll)
                print("[RECORDING] Recording...", flush=True)
            elif total_blocks >= max_blocks:
                break
            continue

        captured.append(block)
        quiet_blocks = quiet_blocks + 1 if level < threshold else 0
        if len(captured) >= max_blocks:
            break
        if len(captured) >= min_blocks and quiet_blocks >= silence_limit:
            break
    return b"".join(captured)


def calibrate(blocks: Iterator[bytes], seconds: float, block_samples: int) -> float:
    count = max(1, round(seconds * SAMPLE_RATE / block_samples))
    print(f"[SETUP] Measuring ambient noise for {seconds:g} seconds. Please remain quiet...", flush=True)
    return float(np.percentile([rms(next(blocks)) for _ in range(count)], 75))


def send_voice(config: dict, audio: bytes, session_id: str) -> AudioResponse:
    url = str(config["server"]).rstrip("/") + "/v1/voice/chat"
    form = {
        "session_id": session_id,
        "response_format": "audio",
        "language": config.get("language", "en"),
    }
    api_key = str(config.get("api_key", ""))
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.post(
        url,
        data=form,
        files={"audio": ("recording.wav", make_wav(audio), "audio/wav")},
        headers=headers,
        timeout=float(config.get("timeout", 180)),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json().get("detail") or response.text
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Server HTTP {response.status_code}: {detail}") from exc
    if not response.content:
        raise RuntimeError("Server returned empty audio")
    return AudioResponse(response.content, response.headers.get("Content-Type", ""))


def wav_to_robot_pcm(data: bytes) -> bytes:
    """Convert an uncompressed PCM WAV to 16 kHz, mono, signed 16-bit PCM."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise RuntimeError(f"Unsupported WAV compression: {wav_file.getcomptype()}")
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            source_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, EOFError) as exc:
        raise RuntimeError("AI response must be an uncompressed PCM WAV file") from exc

    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.int32) - 128) << 8
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.int32)
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        samples = ((values ^ 0x800000) - 0x800000) >> 8
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4") >> 16
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels < 1 or samples.size % channels:
        raise RuntimeError("Invalid channel data in WAV response")
    if channels > 1:
        samples = np.rint(samples.reshape(-1, channels).mean(axis=1)).astype(np.int32)
    if source_rate != SAMPLE_RATE:
        divisor = math.gcd(source_rate, SAMPLE_RATE)
        samples = np.rint(resample_poly(samples.astype(np.float64), SAMPLE_RATE // divisor, source_rate // divisor))
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read config '{path}': {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError("config.json must contain a JSON object")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jarvis voice assistant for a Unitree G1 robot")
    parser.add_argument("network_interface", nargs="?", help="Network interface connected to the robot")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    try:
        config = load_config(args.config)
        network_interface = args.network_interface or config.get("network_interface")
        if not network_interface:
            raise RuntimeError("Set network_interface in config.json or pass it on the command line")

        configured_model_dir = Path(
            os.path.expanduser(str(config.get("model_dir", DEFAULT_MODEL_DIR)))
        )
        model_dir = (
            configured_model_dir
            if configured_model_dir.is_absolute()
            else args.config.resolve().parent / configured_model_dir
        )
        device: str | int | None = config.get("device", "pipewire")
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        recording = RecordingConfig(
            silence_seconds=float(config.get("silence_seconds", 2.5)),
            max_seconds=float(config.get("max_seconds", 20.0)),
            silence_multiplier=float(config.get("silence_multiplier", 2.0)),
            min_silence_rms=float(config.get("min_silence_rms", 100.0)),
        )
        colors_config = config.get("led_colors", {})
        colors = {
            "waiting": parse_rgb(colors_config.get("waiting", [0, 0, 255]), "waiting"),
            "recording": parse_rgb(colors_config.get("recording", [255, 0, 0]), "recording"),
            "processing": parse_rgb(colors_config.get("processing", [255, 128, 0]), "processing"),
            "speaking": parse_rgb(colors_config.get("speaking", [0, 255, 0]), "speaking"),
            "off": parse_rgb(colors_config.get("off", [0, 0, 0]), "off"),
        }

        ChannelFactoryInitialize(0, str(network_interface))
        audio_client = AudioClient()
        audio_client.SetTimeout(float(config.get("robot_timeout", 10.0)))
        audio_client.Init()
        robot = RobotOutput(
            audio_client,
            colors,
            float(config.get("led_refresh_seconds", 1.0)),
        )
        ensure_wake_word_models(str(config.get("wake_word", "hey_jarvis")), model_dir)
        detector = WakeWordDetector(
            str(config.get("wake_word", "hey_jarvis")),
            model_dir,
            float(config.get("threshold", 0.5)),
        )
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        print(f"[STARTUP ERROR] {exc}", file=sys.stderr)
        return 2

    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    session_id = str(config.get("session_id") or f"robot-{uuid.uuid4()}")

    try:
        with Microphone(device, recording.block_samples) as microphone:
            blocks = microphone.blocks()
            noise = calibrate(blocks, 1.5, recording.block_samples)
            robot.set_state("waiting")
            print("[WAITING] Ready. Say 'Jarvis' (Ctrl+C to exit).", flush=True)
            while running:
                found, _score = detector.detected(next(blocks))
                if not found:
                    continue

                detector.reset()
                robot.set_state("recording")
                print("[WAKE] Jarvis detected. Ask your question.", flush=True)
                audio = record_utterance(blocks, noise, recording)
                if not audio:
                    print("[WAITING] No question was detected.", flush=True)
                    robot.set_state("waiting")
                    continue

                try:
                    robot.set_state("processing")
                    print("[PROCESSING] Sending audio to the AI server...", flush=True)
                    response = send_voice(config, audio, session_id)
                    robot.set_state("speaking")
                    print("[SPEAKING] The robot is answering...", flush=True)
                    robot.play_wav(response.data, float(config.get("stream_chunk_seconds", 1.0)))
                except (requests.RequestException, RuntimeError, ValueError) as exc:
                    print(f"[ERROR] {exc}", file=sys.stderr)
                finally:
                    robot.set_state("waiting")
                    print("[WAITING] Say 'Hey Jarvis'.", flush=True)
    except (RuntimeError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    finally:
        try:
            robot.stop()
        except Exception as exc:
            print(f"[WARN] Could not stop audio or turn off the LED: {exc}", file=sys.stderr)
    print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
