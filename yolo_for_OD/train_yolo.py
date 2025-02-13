from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # You can also use yolov8s.pt, yolov8m.pt, etc.

# Train the model
model.train(
    data="./dataset/dataset.yaml",  # Path to dataset.yaml
    epochs=500,  # Number of training epochs
    batch=64,  # Adjust based on your GPU memory
    imgsz=640,  # Image size
    device="cuda"  # Use "cuda" for GPU, or "cpu" if no GPU available
)
