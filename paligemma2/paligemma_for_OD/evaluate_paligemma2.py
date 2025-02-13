import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import supervision as sv
import torch
import numpy as np
from tqdm import tqdm
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from supervision.metrics import MeanAveragePrecision, MetricTarget
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, PaliGemmaProcessor, AutoModelForPreTraining

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
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
        image = Image.open(image_path)
        return image, entry

train_dataset = JSONLDataset(
    jsonl_file_path="./dataset/_annotations.train.jsonl",
    image_directory_path="./dataset",
)
valid_dataset = JSONLDataset(
    jsonl_file_path="./dataset/_annotations.valid.jsonl",
    image_directory_path="./dataset",
)

CLASSES = train_dataset[0][1]['prefix'].replace("detect ", "").split(" ; ")

MODEL_ID ="google/paligemma2-3b-pt-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TORCH_DTYPE = torch.bfloat16

checkpoint_path = "check_point/paligemma2_object_detection_v4_frozen/checkpoint-12160"

config = PeftConfig.from_pretrained(checkpoint_path)
base_model = AutoModelForPreTraining.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, checkpoint_path).to(DEVICE)
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

images = []
targets = []
predictions = []

with torch.inference_mode():
    for i in tqdm(range(len(valid_dataset))):
        image, label = valid_dataset[i]
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
map_result.plot()