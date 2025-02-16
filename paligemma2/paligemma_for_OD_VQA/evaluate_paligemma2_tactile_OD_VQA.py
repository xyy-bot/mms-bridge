import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel, PeftConfig
from supervision.metrics import MeanAveragePrecision, MetricTarget
import supervision as sv
import numpy as np

##########################################
# JSONLDataset for VQA Inference
##########################################
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        """
        Args:
            jsonl_file_path: Path to the JSONL file with VQA annotations.
            image_directory_path: Directory where the images are stored.
                                  (e.g. if an entry has "real_green_bell_pepper_1/frame_0038.jpg",
                                  and images are in "./dataset/dynamic_frames", then set image_directory_path accordingly)
        """
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self):
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path).convert("RGB")
        return image, entry


##########################################
# Setup: Instantiate Validation Dataset
##########################################
# Create train and validation datasets.
# Create train and validation datasets.

valid_vqa_v1_dataset = JSONLDataset(
    jsonl_file_path="./dataset/valid_VQA_v1_annotation.jsonl",
    image_directory_path="./dataset/vqa/validation",
)

valid_vqa_v2_dataset = JSONLDataset(
    jsonl_file_path="./dataset/valid_VQA_v2_annotation.jsonl",
    image_directory_path="./dataset/vqa_v2/validation",
)

valid_od_dataset = JSONLDataset(
    jsonl_file_path="./dataset/valid_OD_annotation.jsonl",
    image_directory_path="./dataset/detection",
)

valid_vqa_dataset = torch.utils.data.ConcatDataset([valid_vqa_v1_dataset,valid_vqa_v2_dataset])



##########################################
# Setup Model, Processor, and Device.
##########################################
MODEL_ID = "google/paligemma2-3b-pt-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.bfloat16

# Update this checkpoint path to your fine-tuned VQA checkpoint.
checkpoint_path = "check_point/paligemma2_od_vqa_augmented_v3/checkpoint-9255"

# Load the PEFT config and the fine-tuned model.
config = PeftConfig.from_pretrained(checkpoint_path)
base_model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, checkpoint_path).to(DEVICE)
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

##########################################
# Loop over the Validation Set for Inference
##########################################
results = []  # Optional: to store all the outputs
for idx in tqdm(range(len(valid_vqa_dataset)), desc="Evaluating validation samples"):
    # Retrieve a sample.
    image, label = valid_vqa_dataset[idx]

    # Build the prompt using the prefix (question). Prepend with "<image>".
    prompt = "<image>" + label["prefix"]

    # Process the inputs.
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(TORCH_DTYPE).to(DEVICE)

    # Get the length of the prompt tokens.
    prefix_length = inputs["input_ids"].shape[-1]

    # Generate the answer.
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # Only decode tokens that were generated beyond the prompt.
    generation = generation[0][prefix_length:]
    decoded_answer = processor.decode(generation, skip_special_tokens=True)

    # Print the results for this sample.
    print(f"Sample {idx}:")
    print("Question (prefix):", label["prefix"])
    print("Ground Truth (suffix):", label["suffix"])
    print("Generated Answer:", decoded_answer)
    print("-" * 50)

    # (Optional) Save the results for later evaluation.
    results.append({
        "question": label["prefix"],
        "ground_truth": label["suffix"],
        "generated": decoded_answer
    })

# Optionally, you can write the results to a JSON file.
with open("inference_results_15epoch_manually_split.json", "w") as f:
    json.dump(results, f, indent=2)

print("Inference on the vqa validation set completed.")

## OD evaluation

images = []
targets = []
predictions = []
CLASSES = valid_od_dataset[0][1]['prefix'].replace("detect ", "").split(" ; ")

with torch.inference_mode():
    for i in tqdm(range(len(valid_od_dataset))):
        image, label = valid_od_dataset[i]
        prefix = "<image>" + label["prefix"]
        suffix = label["suffix"]

        inputs = processor(
            text=prefix,
            images=image,
            return_tensors="pt"
        ).to(TORCH_DTYPE).to(DEVICE)

        prefix_length = inputs["input_ids"].shape[-1]

        generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generation = generation[0][prefix_length:]
        generated_text = processor.decode(generation, skip_special_tokens=True)

        w, h = image.size
        prediction = sv.Detections.from_lmm(
            lmm='paligemma',
            result=generated_text,
            resolution_wh=(w, h),
            classes=CLASSES)

        prediction.class_id = np.array([CLASSES.index(class_name) for class_name in prediction['class_name']])
        prediction.confidence = np.ones(len(prediction))

        target = sv.Detections.from_lmm(
            lmm='paligemma',
            result=suffix,
            resolution_wh=(w, h),
            classes=CLASSES)

        target.class_id = np.array([CLASSES.index(class_name) for class_name in target['class_name']])

        images.append(image)
        targets.append(target)
        predictions.append(prediction)



map_metric = MeanAveragePrecision(metric_target=MetricTarget.BOXES)
map_result = map_metric.update(predictions, targets).compute()

print(map_result)
