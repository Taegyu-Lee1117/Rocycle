import time
import serial

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


PORT = '/dev/ttyACM0'
BAUD = 9600

FORWARD_SPEED = 100
STOP_SPEED = 0


class ConveyorNode(Node):

    def __init__(self):
        super().__init__('conveyor_node')

        self.serial_port = None

        # [수정 — 8일차, 관제PC] 시리얼 포트를 ROS 파라미터로 외부화.
        # 기본값은 기존 하드코딩 값과 동일해서 기존 실행 방식은 그대로
        # 동작한다. 아두이노를 뽑았다 꽂으면 커널이 다음 빈 번호를
        # 할당해서 `/dev/ttyACM0` -> `/dev/ttyACM1`로 옮겨가는데(실제로
        # 발생), 그러면 이 노드가 열지 못한다. `/dev/serial/by-id/...`
        # 는 장치 고유 ID 기반이라 재연결해도 안 바뀌므로 그쪽을
        # 넘기면 된다 -- 카메라가 `/dev/v4l/by-id/...`를 쓰는 것과
        # 같은 이유다(카메라는 이번에 video2 -> video3로 바뀌었는데도
        # by-id 덕분에 아무 문제 없었다).
        self.declare_parameter('serial_port', PORT)
        self.port = self.get_parameter('serial_port').value

        self.command_sub = self.create_subscription(
            String,
            '/conveyor_command',
            self.command_callback,
            10
        )

        self.connect_serial()

        self.get_logger().info('conveyor_node started')
        self.get_logger().info(
            'waiting for /conveyor_command'
        )


    def connect_serial(self):

        try:
            self.serial_port = serial.Serial(
                self.port,
                BAUD,
                timeout=1
            )

            # Arduino 연결 직후 리셋 대기
            time.sleep(2.2)

            self.get_logger().info(
                f'컨베이어 연결 성공: {self.port}'
            )

            # 노드 시작 시 안전하게 정지
            self.send_speed(STOP_SPEED)

        except Exception as e:

            self.serial_port = None

            self.get_logger().error(
                f'컨베이어 연결 실패: {e}'
            )


    def send_speed(self, speed):

        if self.serial_port is None:
            self.get_logger().error(
                '시리얼 포트가 연결되어 있지 않습니다.'
            )
            return

        try:
            command = f'{speed}\n'

            self.serial_port.write(
                command.encode('utf-8')
            )

            self.serial_port.flush()

            self.get_logger().info(
                f'컨베이어 속도 전송: {speed}'
            )

        except Exception as e:

            self.get_logger().error(
                f'컨베이어 명령 전송 실패: {e}'
            )


    def command_callback(self, msg):

        command = msg.data.strip().upper()

        self.get_logger().info(
            f'/conveyor_command 수신: {command}'
        )

        if command == 'START':

            self.send_speed(FORWARD_SPEED)

            self.get_logger().info(
                '컨베이어 시작'
            )

        elif command == 'STOP':

            self.send_speed(STOP_SPEED)

            self.get_logger().info(
                '컨베이어 정지'
            )

        else:

            self.get_logger().warn(
                f'알 수 없는 컨베이어 명령: {command}'
            )


    def destroy_node(self):

        # 노드 종료 시 반드시 컨베이어 정지
        if self.serial_port is not None:

            try:
                self.send_speed(STOP_SPEED)
                self.serial_port.close()

            except Exception:
                pass

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = ConveyorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
