import os
from moozo_ai.utils import upload_to_cloud, save_pil_images
import runpod
from moozo_ai.core import inference_ip_model, get_ip_model

ip_model = get_ip_model()

async def generate_images(job):
    job_input = job['input']
    job_prompt = job_input.get("prompt")
    job_image_url = job_input.get("image_url")
    job_negative_prompt = job_input.get("negative_prompt", "monochrome, lowres, bad anatomy, worst quality, low quality, blurry")
    job_num_samples = job_input.get("num_samples", 1)
    os.makedirs('saved', exist_ok=True)
    try:
        images_pil = inference_ip_model(ip_model, job_prompt, job_image_url, job_negative_prompt, job_num_samples)
        links = await upload_to_cloud(images_pil)
        return {"links": links}
    except Exception as e:
        return {"error": str(e)}
    
runpod.serverless.start({"handler": generate_images})
