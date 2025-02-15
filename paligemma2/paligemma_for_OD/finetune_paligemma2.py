import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import supervision as sv
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

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
    jsonl_file_path="./dataset_augmented/_annotations.train.jsonl",
    image_directory_path="./dataset_augmented",
)
valid_dataset = JSONLDataset(
    jsonl_file_path="./dataset_augmented/_annotations.valid.jsonl",
    image_directory_path="./dataset_augmented",
)


CLASSES = train_dataset[0][1]['prefix'].replace("detect ", "").split(" ; ")

images = []
for i in range(36):
    image, label = train_dataset[i]
    detections = sv.Detections.from_lmm(
        lmm='paligemma',
        result=label["suffix"],
        resolution_wh=(image.width, image.height),
        classes=CLASSES)

    image = sv.BoxAnnotator(thickness=4).annotate(image, detections)
    image = sv.LabelAnnotator(text_scale=2, text_thickness=4).annotate(image, detections)
    images.append(image)

# sv.plot_images_grid(images, (6, 6))


MODEL_ID = "google/paligemma2-3b-pt-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BitsAndBytes configuration for 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA configuration targeting transformer modules (text side)
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# Load the model
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")

# Freeze vision encoder and multimodal projector BEFORE wrapping with LoRA
if hasattr(model, "vision_tower"):
    for param in model.vision_tower.parameters():
        param.requires_grad = False
else:
    print("Warning: 'vision_tower' not found in model.")

if hasattr(model, "multi_modal_projector"):
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False
else:
    print("Warning: 'multi_modal_projector' not found in model.")

# Apply LoRA to the remaining (non-frozen) parts of the model
model = get_peft_model(model, lora_config)

# Optionally, print trainable parameters to verify freezing worked as expected
model.print_trainable_parameters()

TORCH_DTYPE = model.dtype

# Load the processor
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)


def augment_suffix(suffix):
    parts = suffix.split(' ; ')
    random.shuffle(parts)
    return ' ; '.join(parts)


def collate_fn(batch):
    images, labels = zip(*batch)

    paths = [label["image"] for label in labels]
    prefixes = ["<image>" + label["prefix"] for label in labels]
    suffixes = [augment_suffix(label["suffix"]) for label in labels]

    inputs = processor(
        text=prefixes,
        images=images,
        return_tensors="pt",
        suffix=suffixes,
        padding="longest"
    ).to(TORCH_DTYPE).to(DEVICE)

    return inputs

args = TrainingArguments(
    num_train_epochs=40,
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=50,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    output_dir="./check_point/paligemma2_OD_augmented_dataset_40epoch",
    bf16=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    args=args
)

trainer.train()