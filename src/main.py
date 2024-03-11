import os
from moozo_ai.utils import upload_to_cloud, save_pil_images
import runpod
from moozo_ai.core import inference_ip_model, get_ip_model

ip_model = get_ip_model()

async def generate_images(job):
    job_prompt = job["prompt"]
    job_image_url = job["image_url"]
    job_negative_prompt = job["negative_prompt"]
    job_num_samples = job["num_samples"]
    os.makedirs('saved', exist_ok=True)
    if job_negative_prompt == "":
        job_negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    try:
        images_pil = inference_ip_model(ip_model, job_prompt, job_image_url, job_negative_prompt, job_num_samples)
        save_pil_images(images_pil)
        links = upload_to_cloud()
        return {"links": links}
    except Exception as e:
        return {"error": str(e)}
    
runpod.serverless.start({"handler": generate_images})
