import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from PIL import Image
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL
from moozo_ai.utils import get_face_embedding

base_model_path = "/Models"
ip_ckpt = "/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin"
device = "cuda"

def get_ip_model():
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )


    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)
    return ip_model

def inference_ip_model(ip_model, prompt, image_url, 
                       negative_prompt= "monochrome, lowres, bad anatomy, worst quality, low quality, blurry", num_samples=1):
    faceid_embeds = get_face_embedding(image_url)
    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=num_samples,
        width=1024, height=1024,
        num_inference_steps=30, guidance_scale=7.5, seed=2023
    )
    print(images)
    return images

