import argparse
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send text to the Unitree G1 robot and let the robot speak it with TtsMaker."
    )
    parser.add_argument("network_interface", help="Network interface used to reach the robot.")
    parser.add_argument("text", help="Text that the robot should speak.")
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Speaker ID passed to TtsMaker. Default: 0",
    )
    parser.add_argument(
        "--volume",
        type=int,
        help="Optional robot speaker volume from 0 to 100.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=5.0,
        help="Seconds to keep the script alive after sending TTS. Default: 5.0",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ChannelFactoryInitialize(0, args.network_interface)

    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()

    if args.volume is not None:
        if not 0 <= args.volume <= 100:
            raise ValueError("--volume must be between 0 and 100.")
        ret = audio_client.SetVolume(args.volume)
        if ret != 0:
            raise RuntimeError(f"SetVolume failed with code: {ret}")
        print(f"[INFO] Robot volume set to {args.volume}")

    ret = audio_client.TtsMaker(args.text, args.speaker_id)
    if ret != 0:
        raise RuntimeError(f"TtsMaker failed with code: {ret}")

    print("[INFO] TTS request sent to robot successfully")
    print(f"[INFO] Text: {args.text}")

    if args.wait > 0:
        print(f"[INFO] Waiting {args.wait:.1f} seconds before exit")
        time.sleep(args.wait)

    return 0


if __name__ == "__main__":
    sys.exit(main())
