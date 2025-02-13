import sys
import os

from torch.library import define

# Get the absolute path of the root directory
# ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = "/home/au-robotics/PycharmProjects/object_detection_xy"
sys.path.append(ROOT_DIR)  # Ensure Python recognizes the root directory

# Define local package directories
LOCAL_PACKAGES = [
    "vqvae_for_force",
    "arduino_gelsight_data_logging",
    "yolo_for_OD",
    "robotiq_driver"]

# Add each package directory to sys.path if it's not already included
for package in LOCAL_PACKAGES:
    package_path = os.path.join(ROOT_DIR, package)
    if package_path not in sys.path:
        sys.path.append(package_path)

import time
import torch
import numpy as np
from PIL import Image

# import for camera connection and arduino connection
import pyrealsense2 as rs
import cv2
import serial

# import for robot control
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq_driver.robotiq_gripper_control import RobotiqGripper

# import for object detection and tactile image processing
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import supervision as sv

# import for grasping force classification
from vqvae_for_force.utils.transform import NormalizeTransform
from vqvae_for_force.infernce_vqvae import load_model_from_checkpoint, classify_sample_with_classifier

# import arduino reader and gelsight reader
from arduino_gelsight_data_logging.arduino_gelsight_data_logger_revised import ArduinoReader, GelSightReader

# Import other function modules
from yolo_for_OD.coordinate_transformation import transform_to_robot_frame

# Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.bfloat16
rtde_c = RTDEControlInterface("192.168.1.254")
rtde_r = RTDEReceiveInterface("192.168.1.254")
gripper = RobotiqGripper(rtde_c)
gelsight_source = 0  # change to your correct camera index
gelsight_sampling_interval = 0.05  # intended 20 Hz
arduino_port = "/dev/ttyACM0"
arduino_baud_rate = 115200
arduino_sampling_interval = 0.001  # intended 1 kHz
###############################################################################################

# Load Paligemma2 for object detection and visual question answering
def load_paligemma2_for_OD_VQA(pali_checkpoint_path, MODEL_ID = "google/paligemma2-3b-pt-224"):
    print("Loading Paligemma2 Model...")
    base_model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-pt-224")
    model = PeftModel.from_pretrained(base_model, pali_checkpoint_path).to(DEVICE)
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    return model, processor

# Capture Image from RealSense Camera
def capture_realsense_image():
    print("Capturing Image from RealSense Camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    image = np.asanyarray(color_frame.get_data())
    image = Image.fromarray(image[:, :, ::-1])
    pipeline.stop()
    return image

# Perform Object Detection
def detect_objects(image, model, processor):
    print("Running Object Detection...")
    prefix = "<image>detect banana ; cola can ; cucumber ; eggplant ; garlic ; green paprika ; lemon ; potato ; red paprika ; tomato"
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
        classes=CLASSES)
    # Extract bbox and class labels
    detected_objects = []
    for bbox, class_id in zip(detections.xyxy, detections.class_id):
        x_min, y_min, x_max, y_max = bbox  # Extract bbox coordinates
        width, height = x_max - x_min, y_max - y_min  # Compute width and height
        detected_objects.append({
            'bbox': (x_min, y_min, width, height),
            'label': CLASSES[class_id]  # Convert class ID to label
        })
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator(smart_position=True).annotate(annotated_image, detections)
    return detected_objects, annotated_image

def tactile_reasoning(image, label, deformation_mode, model, processor):
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
    tcp_pose = rtde_r.getActualTCPPose()
    x, y, z, rx, ry, rz = tcp_pose
    # Convert axis-angle (rotation vector) to rotation matrix
    rotation_vector = np.array([rx, ry, rz], dtype=float)
    R, _ = cv2.Rodrigues(rotation_vector)  # Convert to 3x3 rotation matrix
    # Build the 4x4 homogeneous transformation matrix
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = [x, y, z]
    return pose_mat


def get_coord_of_object(bbox, config_path="./yolo_for_OD/configs/config_640_480_v3.yaml"):
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    baseTee_matrix = get_robot_pose_matrix()
    object_coords = transform_to_robot_frame((x_center, y_center), 0.1, baseTee_matrix, config_path)
    return object_coords


def clip_force_data_list(force_list, force_threshold=2.0, pre_offset=2, clip_length=1200, scale_factor = 100.0):
    index_found = None
    for idx, (t, f) in enumerate(force_list):
        if f > force_threshold:
            index_found = idx
            break
    if index_found is None:
        print("Force never exceeded the threshold.")
        return []
    # Calculate the start index, ensuring it doesn't go negative.
    start_idx = max(0, index_found - pre_offset)
    end_idx = start_idx + clip_length
    # Extract the segment.
    clipped_segment = force_list[start_idx:end_idx]
    # Here we assume a constant sampling interval.
    if len(force_list) > 1:
        sampling_interval = force_list[1][0] - force_list[0][0]
    else:
        sampling_interval = 0.001  # default value if there's only one sample
    clipped_list = [(i * sampling_interval, f/scale_factor) for i, (_, f) in enumerate(clipped_segment)]
    data_array = np.array(clipped_list)  # Expected shape: (seq_len, 2)
    # Ensure the data is in float32.
    data_array = data_array.astype(np.float32)
    # Convert the NumPy array into a torch.Tensor.
    sample_tensor = torch.tensor(data_array, dtype=torch.float32)
    return sample_tensor

# Main Pipeline Execution
##################################################################
experiment_name = "pipeline_test1"

deformation_mode_mapping = {
    0: "irregular deformation",
    1: "minimal deformation",
    2: "slightly deformation",
    3: "uniform deformation"
}

initial_pos = [1.53400719165802, -1.4003892701915284, 0.9106853644000452, -1.0817401868155976, -1.5660360495196741, 1.5324363708496094]
rtde_c.moveJ(initial_pos, 0.05, 0.05, False)

print("Robot moved to inital pos")

# Create directories if they do not exist
try:
    os.makedirs('test_result/force_data', exist_ok=True)
    os.makedirs('test_result/tactile_data', exist_ok=True)
    os.makedirs('test_result/rgb_data', exist_ok=True)
except Exception as e:
    print(f"Error creating directories: {e}")
    sys.exit(1)

# Load Models
pali_checkpoint_path = "paligemma2/paligemma_for_OD_VQA/check_point/paligemma2_od_vqa_finetune_v1/checkpoint-3420"
pali_ODVQA_model, pali_ODVQA_processor = load_paligemma2_for_OD_VQA(pali_checkpoint_path)
print("Paligemma2 loaded")

# Load VQVAE for Force Classification
vqvae_checkpoint_path = "vqvae_for_force/checkpoints/fourclass_full_train_further/best_checkpoint.pth"
vqvae_model = load_model_from_checkpoint(vqvae_checkpoint_path, device=DEVICE)
print("VQVAE loaded")



# Capture Image from Camera
rgb_image = capture_realsense_image()
# rgb_image.show()
print("RGB image captured")

# Detect Objects
detections, img_with_bbox = detect_objects(rgb_image, pali_ODVQA_model, pali_ODVQA_processor)
filename = os.path.join("test_result/rgb_data", f"{experiment_name}_rgb.jpg")
rgb_image.save(filename)
print("Detected Objects:", detections)

for detection in detections:
    bbox = detection['bbox']
    label = detection['label']
    object_coords = get_coord_of_object(bbox)

    initial_TCP_pose = rtde_r.getActualTCPPose()
    target_pose = initial_TCP_pose.copy()
    target_pose[0:2] = object_coords[0:2].tolist()
    rtde_c.moveL(target_pose, 0.05, 0.05, False)
    print("Robot moved to the top of object")
    time.sleep(1)

    gripper.activate()  # returns to previous position after activation
    gripper.set_force(0)  # from 0 to 100 %
    gripper.set_speed(0)  # from 0 to 100 %
    print("Gripper activated")
    time.sleep(2)

    grasping_pose = target_pose.copy()
    grasping_pose[2] = 0.230000
    rtde_c.moveL(grasping_pose, 0.05, 0.05, False)

    experiment_start_time = time.time()
    arduino_reader = ArduinoReader(
        serial_port=arduino_port,
        baud_rate=arduino_baud_rate,
        log_dir='test_result/force_data',
        experiment_name=experiment_name,
        sampling_interval=arduino_sampling_interval
    )
    arduino_reader.experiment_start_time = experiment_start_time

    gelsight_reader = GelSightReader(
        video_source=gelsight_source,
        log_dir='test_result/tactile_data',
        experiment_name=experiment_name,
        sampling_interval=gelsight_sampling_interval
    )
    gelsight_reader.experiment_start_time = experiment_start_time

    arduino_reader.start()
    gelsight_reader.start()
    time.sleep(2)
    gripper.close()
    time.sleep(5)
    arduino_reader.stop()
    gelsight_reader.stop()
    gripper.open()

    grasping_force = arduino_reader.data
    tactile_image_rgb = cv2.cvtColor(gelsight_reader.last_frame, cv2.COLOR_BGR2RGB)
    tactile_image = Image.fromarray(tactile_image_rgb)

    grasping_force_processed = clip_force_data_list(grasping_force)
    deformation_mode_id, class_probs = classify_sample_with_classifier(vqvae_model,grasping_force_processed,DEVICE)
    deformation_mode = deformation_mode_mapping.get(deformation_mode_id, "Unknown")
    answer_for_tactile = tactile_reasoning(tactile_image, label, deformation_mode, pali_ODVQA_model, pali_ODVQA_processor)

    return_pos = grasping_pose.copy()
    return_pos[2] =  initial_TCP_pose[2]
    rtde_c.moveL(return_pos, 0.05, 0.05, False)

print("\nPipeline Completed.")
rtde_c.stopScript()
rtde_c.disconnect()
rtde_r.disconnect()




