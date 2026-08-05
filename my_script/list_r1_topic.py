#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
import sys
from typing import Iterable, List


def build_cyclonedds_uri(interface_name: str, domain_id: int) -> str:
    return f"""
<CycloneDDS>
  <Domain Id="{domain_id}">
    <General>
      <Interfaces>
        <NetworkInterface name="{interface_name}"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>
""".strip()


def extract_topics(lines: Iterable[str]) -> List[str]:
    topics = set()

    for line in lines:
        for match in re.finditer(r"\brt/[A-Za-z0-9_./-]+\b", line):
            topics.add(match.group(0))

    return sorted(topics)


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("Cách chạy:")
        print("  python3 list_r1_topic.py <ten_card_mang> [domain_id]")
        print("")
        print("Ví dụ:")
        print("  python3 list_r1_topic.py enp3s0")
        print("  python3 list_r1_topic.py enp3s0 0")
        return 1

    interface_name = sys.argv[1]
    domain_id = int(sys.argv[2]) if len(sys.argv) == 3 else 0

    cyclonedds_bin = shutil.which("cyclonedds")
    if not cyclonedds_bin:
        print("Không tìm thấy lệnh 'cyclonedds'.")
        print("Bạn cần cài Python binding của CycloneDDS để có CLI này.")
        return 2

    env = os.environ.copy()
    env["CYCLONEDDS_URI"] = build_cyclonedds_uri(interface_name, domain_id)

    print(f"Card mạng DDS : {interface_name}")
    print(f"DDS Domain ID : {domain_id}")
    print("Đang quét topic từ R1...")
    print("")

    try:
        result = subprocess.run(
            [cyclonedds_bin, "ps"],
            env=env,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("Lệnh cyclonedds chạy quá lâu và đã bị dừng.")
        return 3

    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    topics = extract_topics(output.splitlines())

    if topics:
        print("Các topic tìm thấy:")
        for topic in topics:
            print(f"  - {topic}")
        print("")
        print(f"Tổng cộng: {len(topics)} topic")
        return 0

    print("Chưa tìm thấy topic nào.")
    print("")
    print("Output gốc từ cyclonedds:")
    print(output.strip() or "(không có output)")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())