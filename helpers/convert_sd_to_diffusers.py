import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as torch_transforms
from diffusers import StableDiffusionPipeline
from torchvision.transforms.functional import InterpolationMode

# change these settings before using
merged_ckpt = '3.ckpt'  # set to the .ckpt file
model_id_local = '/data/models/SD-1-5'  # set to the downloaded Stable Diffusion v1.5 folder

# ---------- #

output_dir = merged_ckpt.replace('.ckpt', '')
os.makedirs(output_dir, exist_ok=True)

# load models
# SD core
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id_local,
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None
)
pipeline = StableDiffusionPipeline.from_single_file(
    pretrained_model_link_or_path=merged_ckpt,
    config=model_id_local,
    original_config_file=os.path.join(model_id_local, 'v1-inference.yaml'),
    tokenizer=pipeline.tokenizer,
    text_encoder=pipeline.text_encoder,
    torch_dtype=torch.float16,
    feature_extractor=None,
    safety_checker=None,
)

pipeline.save_pretrained(output_dir, safe_serialization=True)
