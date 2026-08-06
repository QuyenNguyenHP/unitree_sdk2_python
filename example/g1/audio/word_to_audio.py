import argparse
import audioop
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from wav import read_wav, play_pcm_stream


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2


def synthesize_windows_tts(text: str, output_path: Path, voice_name: str | None = None) -> None:
    if os.name != "nt":
        raise RuntimeError("This example currently supports Windows only because it uses System.Speech.")

    script = r"""
Add-Type -AssemblyName System.Speech
$text = $env:UNITREE_TTS_TEXT
$output = $env:UNITREE_TTS_OUTPUT
$voice = $env:UNITREE_TTS_VOICE

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($voice) {
    $synth.SelectVoice($voice)
}

$synth.SetOutputToWaveFile($output)
$synth.Speak($text)
$synth.Dispose()
"""

    env = os.environ.copy()
    env["UNITREE_TTS_TEXT"] = text
    env["UNITREE_TTS_OUTPUT"] = str(output_path)
    if voice_name:
        env["UNITREE_TTS_VOICE"] = voice_name

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Windows TTS synthesis failed.")


def normalize_wav(input_path: Path, output_path: Path) -> None:
    with wave.open(str(input_path), "rb") as src:
        num_channels = src.getnchannels()
        sample_width = src.getsampwidth()
        sample_rate = src.getframerate()
        frames = src.readframes(src.getnframes())

    if sample_width != TARGET_SAMPLE_WIDTH:
        raise RuntimeError(f"Unsupported sample width: {sample_width * 8}-bit. Expected 16-bit PCM.")

    if num_channels == 2:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
    elif num_channels != 1:
        raise RuntimeError(f"Unsupported channel count: {num_channels}.")

    if sample_rate != TARGET_SAMPLE_RATE:
        frames, _ = audioop.ratecv(
            frames,
            sample_width,
            TARGET_CHANNELS,
            sample_rate,
            TARGET_SAMPLE_RATE,
            None,
        )

    with wave.open(str(output_path), "wb") as dst:
        dst.setnchannels(TARGET_CHANNELS)
        dst.setsampwidth(TARGET_SAMPLE_WIDTH)
        dst.setframerate(TARGET_SAMPLE_RATE)
        dst.writeframes(frames)


def play_on_robot(network_interface: str, wav_path: Path, stream_name: str) -> None:
    ChannelFactoryInitialize(0, network_interface)

    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()

    pcm_list, sample_rate, num_channels, is_ok = read_wav(str(wav_path))
    if not is_ok or sample_rate != TARGET_SAMPLE_RATE or num_channels != TARGET_CHANNELS:
        raise RuntimeError("Generated WAV is not compatible with the robot audio stream requirements.")

    play_pcm_stream(audio_client, pcm_list, stream_name)
    audio_client.PlayStop(stream_name)


def build_default_output_path() -> Path:
    return Path(__file__).with_name("generated_tts_16k_mono.wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an English speech WAV file and optionally play it on a Unitree G1 robot."
    )
    parser.add_argument("text", help="English text to synthesize.")
    parser.add_argument(
        "--output",
        default=str(build_default_output_path()),
        help="Path of the normalized WAV file to generate.",
    )
    parser.add_argument(
        "--voice",
        help="Optional Windows TTS voice name, for example: Microsoft Zira Desktop",
    )
    parser.add_argument(
        "--network-interface",
        help="Network interface to reach the robot. If provided, the script will stream the WAV to the robot.",
    )
    parser.add_argument(
        "--stream-name",
        default="word_to_audio",
        help="Logical stream name sent to the robot audio service.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav = Path(temp_dir) / "tts_raw.wav"
        synthesize_windows_tts(args.text, temp_wav, args.voice)
        normalize_wav(temp_wav, output_path)

    print(f"[INFO] Generated WAV: {output_path}")
    print(f"[INFO] Format: {TARGET_SAMPLE_RATE} Hz, mono, 16-bit PCM")

    if args.network_interface:
        print(f"[INFO] Streaming audio to robot via interface: {args.network_interface}")
        play_on_robot(args.network_interface, output_path, args.stream_name)
        print("[INFO] Playback request sent to robot")

    return 0


if __name__ == "__main__":
    sys.exit(main())
