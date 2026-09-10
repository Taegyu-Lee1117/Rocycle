import json
import os

import cv2
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


class RecycleDetectionNode(Node):

    def __init__(self):
        super().__init__('recycle_detection_node')

        # 기본값은 기존 ROS2 package 내부 모델을 사용한다.
        # Docker에서는 model_path 파라미터로 /models/recycle_best.pt를 전달한다.
        package_path = get_package_share_directory('object_detection')
        default_model_path = os.path.join(
            package_path,
            'resource',
            'recycle_best.pt'
        )

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('device', 'cpu')

        self.model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('device').value

        self.get_logger().info(f'YOLO model: {self.model_path}')
        self.get_logger().info(f'YOLO device: {self.device}')

        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        # C270 영상
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        # YOLO 박스가 그려진 확인용 영상
        self.image_pub = self.create_publisher(
            Image,
            '/recycle_detection/image',
            10
        )

        # 이후 tracking_node 연결용 검출 데이터
        self.detection_pub = self.create_publisher(
            String,
            '/recycle_detection/detections',
            10
        )

        self.get_logger().info('Recycle detection node started.')
        self.get_logger().info('Waiting for C270 /image_raw ...')


    def image_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        results = self.model.predict(
            frame,
            conf=0.5,
            device=self.device,
            verbose=False
        )

        result = results[0]
        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                class_name = self.model.names[cls_id]

                detection = {
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'cx': cx,
                    'cy': cy,
                    'bbox': [x1, y1, x2, y2]
                }

                detections.append(detection)

        detection_msg = String()
        detection_msg.data = json.dumps(detections)
        self.detection_pub.publish(detection_msg)

        annotated = result.plot()

        image_msg = self.bridge.cv2_to_imgmsg(
            annotated,
            encoding='bgr8'
        )

        image_msg.header = msg.header
        self.image_pub.publish(image_msg)


def main(args=None):

    rclpy.init(args=args)

    node = RecycleDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

