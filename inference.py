# code for running inference

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

model_id = None # todo: upload trained model to huggingface
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_auth_token=True
).to("cuda")

image_path = None
image = load_image(image_path)

image = pipeline("derain the image", image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7.5).images[0]
image.save("image.png")