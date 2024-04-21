import torch, os, glob
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

model_id = 'maagic6/weather-restoration'
input_folder = './input_snow'
image_filepaths = glob.glob(os.path.join(input_folder, "*.jpg"))
output_folder = './'

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None
).to("cuda")
pipeline.enable_attention_slicing(slice_size="max")
pipeline.enable_model_cpu_offload()

for image_filepath in image_filepaths:

    output_filepath = os.path.join(output_folder, os.path.basename(image_filepath))
    if os.path.exists(output_filepath):
        continue

    image = Image.open(image_filepath)
    image = image.convert("RGB")
    print(image)
    width, height = image.size
    min_dim = min(width, height)
    size = width*height

    if size < 589824:
        scale_factor = 768 / min_dim
    else:
        scale_factor = 1
    
    image = image.resize((int(width * scale_factor), int(height * scale_factor)), resample=Image.BILINEAR)
    print(scale_factor)
    print(image)
    output = pipeline("remove snow from the image", num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, image=image).images[0]
    output = output.resize((width, height), resample=Image.BILINEAR)
    output.save(os.path.basename(image_filepath))