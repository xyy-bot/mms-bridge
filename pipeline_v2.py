#!/usr/bin/env python3
"""
Main Pipeline Script for Object Detection, Tactile Reasoning,
and Grasping Force Classification.
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps

# Import Realsense and Serial libraries
import pyrealsense2 as rs
import serial

# Import object detection and tactile processing modules
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import supervision as sv

# Import robot control interfaces
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
# =============================================================================
# Setup Paths for Local Packages
# =============================================================================

ROOT_DIR = "/home/au-robotics/PycharmProjects/object_detection_xy"
sys.path.append(ROOT_DIR)

LOCAL_PACKAGES = [
    "vqvae_for_force",
    "arduino_gelsight_data_logging",
    "yolo_for_OD",
    "robotiq_driver"
]

for package in LOCAL_PACKAGES:
    package_path = os.path.join(ROOT_DIR, package)
    if package_path not in sys.path:
        sys.path.append(package_path)

# Import Robotiq gripper control modules
from robotiq_driver.robotiq_gripper_control import RobotiqGripper

# Import VQVAE-based force classification modules
from vqvae_for_force.infernce_vqvae import load_model_from_checkpoint, classify_sample_with_classifier

# Import data logging modules for Arduino and GelSight
from arduino_gelsight_data_logging.arduino_gelsight_data_logger_revised import ArduinoReader, GelSightReader

# Import coordinate transformation function
from yolo_for_OD.coordinate_transformation import transform_to_robot_frame

# =============================================================================
# Device and Robot Setup
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.bfloat16

# Robot control interfaces
rtde_c = RTDEControlInterface("192.168.1.254")
rtde_r = RTDEReceiveInterface("192.168.1.254")
gripper = RobotiqGripper(rtde_c)
gripper.activate()
gripper.set_force(0)
gripper.set_speed(0)
print("Gripper activated.")
time.sleep(2)

# Camera and Arduino configuration
gelsight_source = 6  # Adjust to your correct camera index
gelsight_sampling_interval = 0.05  # 20 Hz sampling rate
arduino_port = "/dev/ttyACM0"
arduino_baud_rate = 115200
arduino_sampling_interval = 0.001  # 1 kHz sampling rate


# =============================================================================
# Util Functions
# =============================================================================

def load_paligemma2_for_OD_VQA(pali_checkpoint_path, MODEL_ID="google/paligemma2-3b-pt-224"):
    """Load the Paligemma2 model for object detection and VQA."""
    print("Loading Paligemma2 Model...")
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base_model, pali_checkpoint_path).to(DEVICE)
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    return model, processor


def capture_realsense_image():
    """Capture an image from the RealSense camera and convert it to a PIL Image."""
    print("Capturing image from RealSense Camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    image_np = np.asanyarray(color_frame.get_data())
    # Convert from BGR to RGB using slicing (::-1)
    image = Image.fromarray(image_np[:, :, ::-1])
    # image = ImageOps.mirror(ImageOps.flip(image))
    pipeline.stop()
    return image


def detect_objects(image, model, processor):
    """
    Run object detection on an image using the Paligemma2 model.

    Returns:
        detected_objects (list): List of dicts containing bbox and label.
        annotated_image (PIL.Image): Image annotated with detections.
    """
    print("Running object detection...")
    prefix = "<image>detect banana ; cola can ; cucumber ; eggplant ; garlic ; " \
             "green paprika ; lemon ; potato ; red paprika ; tomato"
    CLASSES = prefix.replace("<image>detect ", "").split(" ; ")
    inputs = processor(text=prefix, images=image, return_tensors="pt").to(TORCH_DTYPE).to(DEVICE)
    prefix_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generation = generation[0][prefix_length:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    w, h = image.size
    detections = sv.Detections.from_lmm(
        lmm='paligemma',
        result=decoded,
        resolution_wh=(w, h),
        classes=CLASSES
    )

    # Process detections into a list of objects
    detected_objects = []
    for bbox, class_id in zip(detections.xyxy, detections.class_id):
        x_min, y_min, x_max, y_max = bbox
        width, height = x_max - x_min, y_max - y_min
        detected_objects.append({
            'bbox': (x_min, y_min, width, height),
            'label': CLASSES[class_id]
        })

    # Annotate image with detections
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator(smart_position=True).annotate(annotated_image, detections)
    return detected_objects, annotated_image


def tactile_reasoning(image, label, deformation_mode, model, processor):
    """
    Perform tactile reasoning given a tactile image, its label, and deformation mode.
    Returns the decoded answer.
    """
    print("Running tactile reasoning...")
    prefix_text = (f"Describe the tactile image for {label} and tell me if it's real or fake "
                   f"as it underwent {deformation_mode} when grasping.")
    prefix = "<image>" + prefix_text
    inputs = processor(text=prefix, images=image, return_tensors="pt").to(TORCH_DTYPE).to(DEVICE)
    prefix_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generation = generation[0][prefix_length:]
    decoded_answer = processor.decode(generation, skip_special_tokens=True)
    return decoded_answer


def get_robot_pose_matrix():
    """
    Retrieve the current robot TCP pose and convert it to a 4x4 homogeneous transformation matrix.
    """
    tcp_pose = rtde_r.getActualTCPPose()
    x, y, z, rx, ry, rz = tcp_pose
    rotation_vector = np.array([rx, ry, rz], dtype=float)
    R, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = [x, y, z]
    return pose_mat


def get_coord_of_object(bbox, base2ee_matrix, config_path="./yolo_for_OD/configs/config_640_480_v4.yaml"):
    """
    Calculate the robot-frame coordinates of an object given its bounding box.
    """
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    object_coords = transform_to_robot_frame((x_center, y_center), 0.38256, base2ee_matrix, config_path)
    return object_coords


def clip_force_data_list(force_list, force_threshold=2.0, pre_offset=2, clip_length=1200, scale_factor=100.0):
    """
    Clips a segment of force data from a list of (time, force) tuples.
    Force values are divided by 'scale_factor' (default 100).

    Returns:
        sample_tensor (torch.Tensor): Tensor of shape (seq_len, 2) ready for VQVAE.
    """
    index_found = None
    for idx, (t, f) in enumerate(force_list):
        if f > force_threshold:
            index_found = idx
            break
    if index_found is None:
        print("Force never exceeded the threshold.")
        return []
    start_idx = max(0, index_found - pre_offset)
    end_idx = start_idx + clip_length
    clipped_segment = force_list[start_idx:end_idx]

    # Assume constant sampling interval
    if len(force_list) > 1:
        sampling_interval = force_list[1][0] - force_list[0][0]
    else:
        sampling_interval = 0.001

    # Reset time and scale force values
    clipped_list = [(i * sampling_interval, f / scale_factor) for i, (_, f) in enumerate(clipped_segment)]
    data_array = np.array(clipped_list, dtype=np.float32)
    sample_tensor = torch.tensor(data_array, dtype=torch.float32)
    return sample_tensor


# =============================================================================
# Main Pipeline Execution
# =============================================================================

if __name__ == "__main__":

    # Create necessary directories
    for folder in ['test_result/force_data', 'test_result/tactile_data', 'test_result/rgb_data']:
        os.makedirs(folder, exist_ok=True)

    # Load Models
    pali_checkpoint_path = "paligemma2/paligemma_for_OD_VQA/check_point/paligemma2_od_vqa_finetune_v1/checkpoint-3420"
    pali_ODVQA_model, pali_ODVQA_processor = load_paligemma2_for_OD_VQA(pali_checkpoint_path)
    print("Paligemma2 loaded.")

    vqvae_checkpoint_path = "vqvae_for_force/checkpoints/fourclass_full_train_further/best_checkpoint.pth"
    vqvae_model = load_model_from_checkpoint(vqvae_checkpoint_path, device=DEVICE)
    print("VQVAE loaded.")



    # Mapping from VQVAE predicted labels to deformation modes
    deformation_mode_mapping = {
        0: "irregular deformation",
        1: "minimal deformation",
        2: "slightly deformation",
        3: "uniform deformation"
    }

    # Move robot to initial position
    initial_pos = [1.53400719165802, -1.4003892701915284, 0.9106853644000452,
                   -1.0817401868155976, -1.5660360495196741, 1.5324363708496094]
    rtde_c.moveJ(initial_pos, 0.05, 0.05, False)
    print("Robot moved to initial position.")
    initial_TCP_pose = rtde_r.getActualTCPPose()
    print("Wait for robot stabilized")
    time.sleep(2)

    # Activate gripper and set parameters


    experiment_name = "pipeline_test8"
    force_log_dir = os.path.join("test_result", "force_data", experiment_name)
    tactile_log_dir = os.path.join("test_result", "tactile_data", experiment_name)

    # Capture an RGB image
    rgb_image = capture_realsense_image()
    print("RGB image captured.")

    # Detect objects in the image
    detections, img_with_bbox = detect_objects(rgb_image, pali_ODVQA_model, pali_ODVQA_processor)
    rgb_filename = os.path.join("test_result/rgb_data", f"{experiment_name}_rgb.jpg")
    img_with_bbox_filename = os.path.join("test_result/rgb_data", f"{experiment_name}_img_bbox.jpg")
    rgb_image.save(rgb_filename)
    img_with_bbox.save(img_with_bbox_filename)
    print("Detected Objects:", detections)
    baseTee_matrix = get_robot_pose_matrix()

    # Process each detected object
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        label = detection['label']
        object_coords = get_coord_of_object(bbox,baseTee_matrix)
        logging_name = f"{label}_{i}"

        # Move robot to object location
        target_pose = initial_TCP_pose.copy()
        target_pose[0:2] = np.array(object_coords[0:2]).tolist()
        rtde_c.moveL(target_pose, 0.05, 0.05, False)
        print("Robot moved above the", label)
        time.sleep(1)

        # Move robot to grasping pose
        grasping_pose = target_pose.copy()
        grasping_pose[2] = 0.230000
        rtde_c.moveL(grasping_pose, 0.05, 0.05, False)

        # Start force and tactile data logging
        experiment_start_time = time.time()
        arduino_reader = ArduinoReader(
            serial_port=arduino_port,
            baud_rate=arduino_baud_rate,
            log_dir=force_log_dir,
            experiment_name=logging_name,
            sampling_interval=arduino_sampling_interval
        )
        arduino_reader.experiment_start_time = experiment_start_time

        gelsight_reader = GelSightReader(
            video_source=gelsight_source,
            log_dir=tactile_log_dir,
            experiment_name=logging_name,
            sampling_interval=gelsight_sampling_interval
        )
        gelsight_reader.experiment_start_time = experiment_start_time

        # Start data collection
        arduino_reader.start()
        gelsight_reader.start()
        time.sleep(2)

        # Perform grasping actions
        gripper.close()
        time.sleep(5)
        arduino_reader.stop()
        gelsight_reader.stop()
        gripper.open()

        print("Tactile information obtained")

        # Process collected force data and tactile image
        grasping_force = arduino_reader.data
        tactile_image_rgb = cv2.cvtColor(gelsight_reader.last_frame, cv2.COLOR_BGR2RGB)
        tactile_image = Image.fromarray(tactile_image_rgb)

        grasping_force_processed = clip_force_data_list(grasping_force)
        deformation_mode_id, class_probs = classify_sample_with_classifier(
            vqvae_model, grasping_force_processed, DEVICE
        )
        deformation_mode = deformation_mode_mapping.get(deformation_mode_id, "Unknown")
        print(f"Object '{label}': deformation mode - {deformation_mode}")

        answer_for_tactile = tactile_reasoning(
            tactile_image, label, deformation_mode, pali_ODVQA_model, pali_ODVQA_processor
        )
        print(f"Tactile reasoning answer: {answer_for_tactile}")

        # Move robot back to safe position
        return_pos = grasping_pose.copy()
        return_pos[2] = initial_TCP_pose[2]
        rtde_c.moveL(return_pos, 0.05, 0.05, False)

    rtde_c.moveL(initial_TCP_pose, 0.05, 0.05, False)
    print("\nPipeline Completed.")

    rtde_c.stopScript()
    rtde_c.disconnect()
    rtde_r.disconnect()

