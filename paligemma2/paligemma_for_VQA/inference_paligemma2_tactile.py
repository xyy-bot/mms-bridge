import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel, PeftConfig

##########################################
# JSONL Dataset for VQA Inference
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
        # Construct the full path to the image using the relative path stored in JSONL.
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path).convert("RGB")
        return image, entry

##########################################
# Instantiate the dataset for inference.
##########################################
# Here we assume that your validation (or inference) JSONL is in "./dataset"
# and that the images are located in "./dataset/dynamic_frames".
valid_dataset = JSONLDataset(
    jsonl_file_path="./dataset/_annotations.valid.jsonl",
    image_directory_path="./dataset/dynamic_frames",
)

##########################################
# Setup Model, Processor, and Device.
##########################################
MODEL_ID = "google/paligemma2-3b-pt-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.bfloat16

# Update the checkpoint path to point to your VQA fine-tuned checkpoint.
checkpoint_path = "check_point/paligemma2_vqa_finetune_manually_v1/checkpoint-1770"  # <-- Replace XXXX with your actual checkpoint number

# Load the PEFT configuration and model.
config = PeftConfig.from_pretrained(checkpoint_path)
base_model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, checkpoint_path).to(DEVICE)
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

##########################################
# Inference: Generate an answer for one sample.
##########################################
# Select a sample from the validation dataset.
# (You can loop over the dataset for batch inference.)
image, label = valid_dataset[32]  # change the index as desired

# Build the prompt. For VQA, the prefix (i.e. the question) is provided.
# prompt = "<image>" + label["prefix"]
# Note: We do not pass the "suffix" (i.e. the answer) as input; the model should generate it.
# image, lael = valid_dataset[53]
# prompt = "<image>Describe the tactile image for lemon and tell me if it's real or fake as it underwent dramatic deformation when grasping."
# Process the image and text prompt.

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to(TORCH_DTYPE).to(DEVICE)

# Compute the length of the prompt tokens.
prefix_length = inputs["input_ids"].shape[-1]

# Generate the answer (with no sampling).
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=256, do_sample=True)
    # Only decode tokens generated after the prompt.
    generation = generation[0][prefix_length:]
    decoded_answer = processor.decode(generation, skip_special_tokens=True)
    print("Generated Answer:", decoded_answer)

##########################################
# Optionally, overlay the answer on the image and display it.
##########################################
# draw = ImageDraw.Draw(image)
# # Try to load a TrueType font; fallback to default if unavailable.
# try:
#     font = ImageFont.truetype("arial.ttf", size=20)
# except Exception:
#     font = ImageFont.load_default()
#
# # Prepare the text to overlay.
# text_to_draw = f"Answer: {decoded_answer}"
# # Draw the text at position (10, 10) with red color.
# draw.text((10, 10), text_to_draw, fill="red", font=font)
# image.show()
