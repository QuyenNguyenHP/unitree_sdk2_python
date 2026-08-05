import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient


def main():
    # Kiểm tra tham số đầu vào.
    # Cần ít nhất tên card mạng đang kết nối với robot, ví dụ: enp2s0.
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <networkInterface> [turn]")
        print(f"Example: python3 {sys.argv[0]} enp2s0")
        print(f"Example: python3 {sys.argv[0]} enp2s0 turn")
        sys.exit(1)

    # Lấy tên network interface từ dòng lệnh.
    network_interface = sys.argv[1]
    # Nếu có thêm đối số "turn" thì robot sẽ vừa xoay vừa vẫy tay.
    turn_around = len(sys.argv) > 2 and sys.argv[2].lower() == "turn"

    # In thông tin để người vận hành kiểm tra trước khi gửi lệnh thật cho robot.
    print("WARNING: Make sure the robot has enough free space before continuing.")
    print(f"Using network interface: {network_interface}")
    print(f"Wave mode: {'turn and wave' if turn_around else 'wave only'}")
    input("Press Enter to continue...")

    # Khởi tạo hệ thống DDS communication trên card mạng đã chọn.
    # Đây là bước bắt buộc trước khi tạo client để nói chuyện với robot.
    ChannelFactoryInitialize(0, network_interface)

    # Tạo client điều khiển locomotion của R1.
    loco_client = LocoClient()
    # Đặt thời gian timeout cho RPC call.
    loco_client.SetTimeout(10.0)
    # Đăng ký API và chuẩn bị client sẵn sàng gửi lệnh.
    loco_client.Init()

    # Move the robot into locomotion mode before sending the arm task.
    print("Switching robot to start mode...")
    # Gọi trực tiếp API SetFsmId để nhận được mã trả về thực tế.
    # 811 là trạng thái Start trong LocoClient.
    start_code = loco_client.SetFsmId(811)
    print(f"Start result: {start_code}")
    # Chờ ngắn để robot kịp chuyển mode trước khi gửi lệnh tiếp theo.
    time.sleep(2)

    print("Sending wave-hand command...")
    # Gọi trực tiếp API SetTaskId để nhận được mã trả về thực tế.
    # task_id = 0: chỉ vẫy tay
    # task_id = 1: vừa xoay vừa vẫy tay
    wave_code = loco_client.SetTaskId(1 if turn_around else 0)
    print(f"WaveHand result: {wave_code}")


if __name__ == "__main__":
    main()
