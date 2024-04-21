import gradio as gr
import torch
import clip
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionInstructPix2PixPipeline

class_model = "trained_model_epoch_1.pth" #path to clip checkpoint
checkpoint = torch.load(class_model, map_location="cuda")
model_id = "maagic6/weather-restoration"


clip_model, preprocess_clip = clip.load("ViT-B/16", device="cuda")
clip_model.load_state_dict(checkpoint)
clip_model.eval()
clip_model.to("cuda")

def preprocess_image(image_path):
    # Define the transform to preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
    ])
    image = Image.open(image_path).convert("RGB")
    tensor_image = transform(image)
    print("Shape of tensor is:")
    print(tensor_image.shape)
    return tensor_image

def classify_and_recover(filepath, option):
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to("cuda")
    if option == "Auto":
        print(filepath)
        image = Image.open(filepath)
        image = image.convert("RGB")
        image3 = preprocess_clip(Image.open(filepath)).unsqueeze(0).to("cuda")
        height, width = image.size
        image = image.resize((height*2, width*2), resample=Image.BILINEAR)

        instructions = ['Snow', 'Raindrop', 'Haze', 'Rain']
        with torch.no_grad():
            text = clip.tokenize(instructions).to('cuda')
            #image_features = clip_model.encode_image(image3)
            #text_features = clip_model.encode_text(text)
            logits_per_image, logits_per_text = clip_model(image3, text)
            probs = logits_per_image.softmax(dim=1).cpu().numpy()
            predicted_label = torch.argmax(logits_per_image)
            predicted_label = instructions[predicted_label]
            print(probs)
            print(predicted_label)

        torch.cuda.empty_cache()

        if predicted_label == "Rain":
            pipeline.enable_attention_slicing(slice_size="max")
            pipeline.enable_model_cpu_offload()
            image = pipeline("remove rain from the image-", num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, image=image).images[0]

        if predicted_label == "Raindrop":
            pipeline.enable_attention_slicing(slice_size="max")
            pipeline.enable_model_cpu_offload()
            image = pipeline("remove raindrops from the image", num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, image=image).images[0]

        if predicted_label == "Snow":
            pipeline.enable_attention_slicing(slice_size="max")
            pipeline.enable_model_cpu_offload()
            image = pipeline("remove snow from the image", num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, image=image).images[0]

        if predicted_label == "Haze":
            pipeline.enable_attention_slicing(slice_size="max")
            pipeline.enable_model_cpu_offload()
            image = pipeline("remove haze from the image", num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, image=image).images[0]

        return predicted_label, image

demo = gr.Interface(
    fn=classify_and_recover,
    inputs=[gr.Image(type="filepath"), gr.Radio(["Auto", "Haze", "Snow", "Rain", "Raindrop"])],
    outputs=[gr.Label(), gr.Image(type="pil")]
    #flagging_options=["blurry", "incorrect", "other"],
)

if __name__ == "__main__":
    demo.launch()
