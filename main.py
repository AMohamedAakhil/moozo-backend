import os
from moozo_ai.utils import download_models, upload_to_cloud, save_pil_images
import subprocess
from moozo_ai.core import inference_ip_model, get_ip_model
from typing import Optional
from fastapi import FastAPI


app = FastAPI()

if not os.path.exists('IP-Adapter-FaceID'):
    download_models()
    print("downloaded models")


subprocess.run("mkdir", "-p", "saved")

ip_model = get_ip_model()

@app.post("/generate_images/")
async def generate_images(prompt: str, image_url: str, 
                          negative_prompt: Optional[str] = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
                          num_samples: Optional[int] = 2):
    try:
        images_pil = inference_ip_model(ip_model, prompt, image_url, negative_prompt, num_samples)
        save_pil_images(images_pil)
        links = upload_to_cloud()
        return {"links": links}
    except Exception as e:
        return {"error": str(e)}
    

'''
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''