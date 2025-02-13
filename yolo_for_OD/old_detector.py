# import torch
# import yaml
# import cv2
# import numpy as np
# import pyrealsense2 as rs
# from ultralytics import YOLO
#
# # Load configuration
# with open("configs/config_640_480_v2.yaml", "r") as file:
#     config = yaml.safe_load(file)
#
# MODEL_PATH = config["model"]["path"]
# CONFIDENCE_THRESHOLD = config["model"]["confidence_threshold"]
#
#
# class ObjectDetector:
#     def __init__(self):
#         """Initialize YOLO model and RealSense camera"""
#         self.model = YOLO(MODEL_PATH)
#         # self.model = YOLO("./runs/detect/train6/weights/best.pt")
#
#         # Initialize RealSense camera
#         self.pipeline = rs.pipeline()
#         config = rs.config()
#         config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#         config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#         self.pipeline.start(config)
#
#         # Get depth scale
#         self.align = rs.align(rs.stream.color)
#         self.depth_scale = self.get_depth_scale()
#
#     def get_depth_scale(self):
#         """Retrieve the depth scale from the RealSense sensor"""
#         profile = self.pipeline.get_active_profile()
#         depth_sensor = profile.get_device().first_depth_sensor()
#         return depth_sensor.get_depth_scale()
#
#     def get_frames(self):
#         """Retrieve synchronized RGB and depth frames"""
#         frames = self.pipeline.wait_for_frames()
#         aligned_frames = self.align.process(frames)
#
#         color_frame = aligned_frames.get_color_frame()
#         depth_frame = aligned_frames.get_depth_frame()
#
#         if not color_frame or not depth_frame:
#             return None, None
#
#         # Convert to numpy arrays
#         color_image = np.asanyarray(color_frame.get_data())
#         depth_image = np.asanyarray(depth_frame.get_data())
#
#         return color_image, depth_image
#
#     def detect_objects(self, image, depth_frame):
#         """Detect objects and return bounding boxes with real-world coordinates"""
#         results = self.model(image)[0]
#
#         detections = []
#         for box in results.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             confidence = float(box.conf[0])
#             class_id = int(box.cls[0])
#
#             if confidence > CONFIDENCE_THRESHOLD:
#                 # Get depth at center of bounding box
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 depth_value = depth_frame[center_y, center_x] * self.depth_scale  # Convert to meters
#
#                 detections.append({
#                     "bbox": [x1, y1, x2 - x1, y2 - y1],
#                     "confidence": confidence,
#                     "class_id": class_id,
#                     "depth": depth_value,
#                     "center": (center_x, center_y)
#                 })
#
#         return detections

import torch
import yaml
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load configuration
with open("configs/config_640_480_v3.yaml", "r") as file:
    config = yaml.safe_load(file)

MODEL_PATH = config["model"]["path"]
CONFIDENCE_THRESHOLD = config["model"]["confidence_threshold"]

class ObjectDetector:
    def __init__(self):
        """Initialize YOLO model and RealSense camera"""
        # self.model = YOLO(MODEL_PATH)
        self.model = YOLO("./runs/detect/train10/weights/best.pt")

        # Initialize RealSense camera
        self.pipeline = rs.pipeline()
        config_rs = rs.config()
        config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # Enable both IR streams (indices 1 and 2)
        config_rs.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        config_rs.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
        self.pipeline.start(config_rs)

        # Align depth to color image
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.get_depth_scale()

        # IR threshold for creating a confidence mask (tune this value as needed)
        self.ir_confidence_threshold = 50

    def get_depth_scale(self):
        """Retrieve the depth scale from the RealSense sensor"""
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()

    def get_frames(self):
        """
        Retrieve synchronized color, depth, and both IR frames.
        Combine the IR images into one confidence map to refine the depth image.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # Retrieve color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        # Retrieve both IR frames
        ir_frame1 = aligned_frames.get_infrared_frame(1)
        ir_frame2 = aligned_frames.get_infrared_frame(2)

        if not color_frame or not depth_frame or not ir_frame1 or not ir_frame2:
            return None, None

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image1 = np.asanyarray(ir_frame1.get_data())
        ir_image2 = np.asanyarray(ir_frame2.get_data())

        # -------------------------------------------------------
        # Combine the two IR images to form a confidence map.
        # Here we compute the average intensity per pixel.
        # You could also use a different fusion method if desired.
        # -------------------------------------------------------
        combined_ir = ((ir_image1.astype(np.float32) + ir_image2.astype(np.float32)) / 2).astype(np.uint8)

        # Create a mask where the combined IR intensity is above the threshold.
        confidence_mask = combined_ir > self.ir_confidence_threshold

        # Use the confidence mask to filter the depth image:
        # For pixels with low IR confidence, set the depth to 0 (or apply interpolation)
        enhanced_depth = np.where(confidence_mask, depth_image, 0)

        return color_image, enhanced_depth

    def detect_objects(self, image, depth_frame):
        """
        Detect objects and return bounding boxes with real-world coordinates.
        The depth is taken from the enhanced depth image.
        """
        results = self.model(image)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                # Get depth at the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # Multiply by the depth scale to convert raw units to meters
                depth_value = depth_frame[center_y, center_x] * self.depth_scale

                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "depth": depth_value,
                    "center": (center_x, center_y)
                })

        return detections
