import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from std_msgs.msg import String

from voice_processing.stt import STT
from voice_processing.keyword_extraction import ExtractKeyword, load_api_key
from voice_processing.wakeup_word import WakeupWord


class GetKeyword(Node):

    def __init__(self):
        super().__init__('get_keyword')

        # /get_keyword 서비스
        self.srv = self.create_service(
            Trigger,
            '/get_keyword',
            self.get_keyword_callback
        )

        # /voice_command 토픽 publisher
        self.voice_command_pub = self.create_publisher(
            String,
            '/voice_command',
            10
        )

        self.get_logger().info('get_keyword node started')
        self.get_logger().info('Publisher ready: /voice_command')


    def get_keyword_callback(self, request, response):

        stt = None
        extractor = None
        wakeup = None

        try:
            key = load_api_key()

            stt = STT(key)
            extractor = ExtractKeyword(key)
            wakeup = WakeupWord()

            self.get_logger().info(
                'Hello Rokey를 말하세요 (최대 30초 대기).'
            )

            detected = wakeup.wait(
                timeout=30.0,
                keep_running=rclpy.ok
            )

            if not detected:
                response.success = False
                response.message = '호출어 감지 실패'
                return response

            self.get_logger().info(
                '호출어 감지. 지금부터 5초 동안 명령을 말하세요.'
            )

            sentence = stt.speech2text()

            self.get_logger().info(
                f'STT 인식 결과: {sentence}'
            )

            command = extractor.extract_keyword(sentence)

            self.get_logger().info(
                f'명령 분류 결과: {command}'
            )

            if command == 'UNKNOWN':
                response.success = False
                response.message = f'명령을 인식하지 못했습니다: {sentence}'
                return response

            # --------------------------------
            # /voice_command 토픽 발행
            # --------------------------------
            msg = String()
            msg.data = command

            self.voice_command_pub.publish(msg)

            self.get_logger().info(
                f'/voice_command 발행: {command}'
            )

            response.success = True
            response.message = command

            return response

        except Exception as e:

            self.get_logger().error(
                f'{type(e).__name__}: {e}'
            )

            response.success = False
            response.message = str(e)

            return response

        finally:

            if wakeup is not None:
                try:
                    wakeup.close()
                except Exception:
                    pass

            if stt is not None:
                try:
                    stt.close()
                except Exception:
                    pass

            if extractor is not None:
                try:
                    extractor.close()
                except Exception:
                    pass


def main(args=None):

    rclpy.init(args=args)

    node = GetKeyword()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
