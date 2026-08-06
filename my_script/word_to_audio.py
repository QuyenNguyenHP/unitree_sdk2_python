import argparse
import audioop
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2


def read_wav(filename):
    try:
        with open(filename, "rb") as f:
            def read(fmt):
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

            chunk_id, = read("<I")
            if chunk_id != 0x46464952:
                print(f"[ERROR] chunk_id != 'RIFF': {hex(chunk_id)}")
                return [], -1, -1, False

            _chunk_size, = read("<I")
            format_tag, = read("<I")
            if format_tag != 0x45564157:
                print(f"[ERROR] format != 'WAVE': {hex(format_tag)}")
                return [], -1, -1, False

            subchunk1_id, = read("<I")
            subchunk1_size, = read("<I")

            if subchunk1_id == 0x4B4E554A:
                f.seek(subchunk1_size, 1)
                subchunk1_id, = read("<I")
                subchunk1_size, = read("<I")

            if subchunk1_id != 0x20746D66:
                print(f"[ERROR] subchunk1_id != 'fmt ': {hex(subchunk1_id)}")
                return [], -1, -1, False

            if subchunk1_size not in [16, 18]:
                print(f"[ERROR] subchunk1_size != 16 or 18: {subchunk1_size}")
                return [], -1, -1, False

            audio_format, = read("<H")
            if audio_format != 1:
                print(f"[ERROR] audio_format != PCM (1): {audio_format}")
                return [], -1, -1, False

            num_channels, = read("<H")
            sample_rate, = read("<I")
            byte_rate, = read("<I")
            block_align, = read("<H")
            bits_per_sample, = read("<H")

            expected_byte_rate = sample_rate * num_channels * bits_per_sample // 8
            if byte_rate != expected_byte_rate:
                print(f"[ERROR] byte_rate mismatch: got {byte_rate}, expected {expected_byte_rate}")
                return [], -1, -1, False

            expected_align = num_channels * bits_per_sample // 8
            if block_align != expected_align:
                print(f"[ERROR] block_align mismatch: got {block_align}, expected {expected_align}")
                return [], -1, -1, False

            if bits_per_sample != 16:
                print(f"[ERROR] Only 16-bit samples supported, got {bits_per_sample}")
                return [], -1, -1, False

            if subchunk1_size == 18:
                extra_size, = read("<H")
                if extra_size != 0:
                    print(f"[ERROR] extra_size != 0: {extra_size}")
                    return [], -1, -1, False

            while True:
                subchunk2_id, subchunk2_size = read("<II")
                if subchunk2_id == 0x61746164:
                    break
                f.seek(subchunk2_size, 1)

            raw_pcm = f.read(subchunk2_size)
            if len(raw_pcm) != subchunk2_size:
                print("[ERROR] Failed to read full PCM data")
                return [], -1, -1, False

            return list(raw_pcm), sample_rate, num_channels, True

    except Exception as exc:
        print(f"[ERROR] read_wav() failed: {exc}")
        return [], -1, -1, False


def play_pcm_stream(client, pcm_list, stream_name="word_to_audio", chunk_size=96000, sleep_time=1.0):
    pcm_data = bytes(pcm_list)
    stream_id = str(int(time.time() * 1000))
    offset = 0
    chunk_index = 0

    while offset < len(pcm_data):
        chunk = pcm_data[offset:offset + chunk_size]
        ret_code, _ = client.PlayStream(stream_name, stream_id, chunk)
        if ret_code != 0:
            raise RuntimeError(f"Failed to send chunk {chunk_index}, return code: {ret_code}")

        print(f"[INFO] Chunk {chunk_index} sent successfully")
        offset += len(chunk)
        chunk_index += 1
        time.sleep(sleep_time)


def find_tts_backend(preferred_backend=None):
    candidates = []
    if preferred_backend:
        candidates.append(preferred_backend)
    candidates.extend(["espeak-ng", "espeak", "pico2wave"])

    for backend in candidates:
        if shutil.which(backend):
            return backend

    raise RuntimeError(
        "No supported TTS backend found. Install one of: espeak-ng, espeak, pico2wave."
    )


def synthesize_linux_tts(text, output_path: Path, backend: str, voice=None):
    if backend == "espeak-ng":
        command = [backend, "-w", str(output_path), text]
        if voice:
            command[1:1] = ["-v", voice]
    elif backend == "espeak":
        command = [backend, "-w", str(output_path), text]
        if voice:
            command[1:1] = ["-v", voice]
    elif backend == "pico2wave":
        command = [backend, "-w", str(output_path)]
        if voice:
            command.extend(["-l", voice])
        command.append(text)
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"TTS synthesis failed with backend {backend}.")


def normalize_wav(input_path: Path, output_path: Path):
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


def play_on_robot(network_interface, wav_path: Path, stream_name):
    ChannelFactoryInitialize(0, network_interface)

    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()

    pcm_list, sample_rate, num_channels, is_ok = read_wav(str(wav_path))
    if not is_ok or sample_rate != TARGET_SAMPLE_RATE or num_channels != TARGET_CHANNELS:
        raise RuntimeError("Generated WAV is not compatible with the robot audio stream requirements.")

    play_pcm_stream(audio_client, pcm_list, stream_name)
    audio_client.PlayStop(stream_name)


def build_default_output_path():
    return Path(__file__).with_name("generated_tts_16k_mono.wav")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an English speech WAV file on Ubuntu and optionally play it on a Unitree G1 robot."
    )
    parser.add_argument("text", help="English text to synthesize.")
    parser.add_argument(
        "--output",
        default=str(build_default_output_path()),
        help="Path of the normalized WAV file to generate.",
    )
    parser.add_argument(
        "--backend",
        choices=["espeak-ng", "espeak", "pico2wave"],
        help="Preferred Linux TTS backend. If omitted, the script auto-detects one.",
    )
    parser.add_argument(
        "--voice",
        help="Optional voice/language name supported by the chosen backend, for example: en-us",
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


def main():
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = find_tts_backend(args.backend)
    print(f"[INFO] Using TTS backend: {backend}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wav = Path(temp_dir) / "tts_raw.wav"
        synthesize_linux_tts(args.text, temp_wav, backend, args.voice)
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
