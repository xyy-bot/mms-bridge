import torch
import yaml
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
from ultralytics import YOLO

# Load configuration
with open("configs/config_720p.yaml", "r") as file:
    config = yaml.safe_load(file)

MODEL_PATH = config["model"]["path"]
CONFIDENCE_THRESHOLD = config["model"]["confidence_threshold"]


class ObjectDetector:
    def __init__(self):
        """Initialize YOLO model and optimized RealSense camera"""
        # Load YOLOv8 model
        self.model = YOLO("./runs/detect/train4/weights/best.pt")

        # Initialize RealSense camera
        self.pipeline = rs.pipeline()
        config = rs.config()

        # 🔹 Enable high-resolution streams with higher FPS
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # Higher FPS
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        # Start pipeline
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.get_depth_scale()

        # 🔹 Apply camera settings for best quality
        self.set_camera_settings()

        # 🔹 Initialize post-processing filters (for depth quality)
        # self.decimation = rs.decimation_filter()  # Reduces depth resolution for better performance
        # self.spatial = rs.spatial_filter()  # Smooths depth values (removes noise)
        # self.temporal = rs.temporal_filter()  # Reduces flickering noise

        # 🔹 Multi-threaded frame processing to improve FPS
        self.color_frame = None
        self.depth_frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def set_camera_settings(self):
        """Adjust RealSense settings for best image and depth quality"""
        device = self.profile.get_device()
        sensors = device.query_sensors()

        for sensor in sensors:
            if sensor.is_depth_sensor():
                sensor.set_option(rs.option.visual_preset, 5)  # High-accuracy preset
                sensor.set_option(rs.option.laser_power, 100)  # Max laser power
                sensor.set_option(rs.option.emitter_enabled, 1)  # Enable depth emitter

            else:  # RGB Sensor
                None
                # sensor.set_option(rs.option.sharpness, 100)  # Increase sharpness
                # sensor.set_option(rs.option.exposure, 100)  # Adjust exposure (manual mode)
                # sensor.set_option(rs.option.gain, 16)  # Increase gain for better brightness
                # sensor.set_option(rs.option.white_balance, 4500)  # Adjust white balance
                # sensor.set_option(rs.option.enable_auto_exposure, 1)  # Enable auto exposure


        print("✅ RealSense camera settings applied.")

    def get_depth_scale(self):
        """Retrieve the depth scale from the RealSense sensor"""
        depth_sensor = self.profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()

    def update_frames(self):
        """Continuously update frames in a separate thread for real-time processing"""
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            self.color_frame = aligned_frames.get_color_frame()
            self.depth_frame = aligned_frames.get_depth_frame()

    def get_frames(self):
        """Retrieve the latest color and depth frames"""
        if self.color_frame and self.depth_frame:
            color_image = np.asanyarray(self.color_frame.get_data())
            depth_image = np.asanyarray(self.depth_frame.get_data())

            # 🔹 Apply post-processing filters to improve depth quality
            # depth_frame_filtered = self.decimation.process(self.depth_frame)
            # depth_frame_filtered = self.spatial.process(depth_frame_filtered)
            # depth_frame_filtered = self.temporal.process(depth_frame_filtered)

            # depth_filtered_image = np.asanyarray(depth_frame_filtered.get_data())

            return color_image, depth_image

        return None, None

    def detect_objects(self, image, depth_frame):
        """Detect objects and return bounding boxes with real-world coordinates"""
        results = self.model(image)[0]

        height, width = depth_frame.shape  # Get depth image size

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            if confidence > CONFIDENCE_THRESHOLD:
                # Compute center of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 🔹 Ensure center_x, center_y are within valid depth image range
                center_x = max(0, min(center_x, width - 1))
                center_y = max(0, min(center_y, height - 1))

                # Retrieve depth value safely
                depth_value = depth_frame[center_y, center_x] * self.depth_scale  # Convert to meters

                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": confidence,
                    "class_id": class_id,
                    "depth": depth_value,
                    "center": (center_x, center_y)
                })

        return detections

    def stop(self):
        """Stop RealSense and terminate threading"""
        self.running = False
        self.thread.join()
        self.pipeline.stop()
