import os
import argparse
from moozo_ai.utils import upload_to_cloud, save_pil_images
# import runpod
from moozo_ai.core import inference_ip_model, get_ip_model

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Generate images using moozo_ai")

# Define the arguments that the script accepts
parser.add_argument('--prompt', help='The job prompt', required=True)
parser.add_argument('--image_url', help='The job image URL', required=True)
parser.add_argument('--negative_prompt', help='The job negative prompt', required=True)
parser.add_argument('--num_samples', type=int, help='The number of samples to generate', required=True)

# Parse the arguments
args = parser.parse_args()

ip_model = get_ip_model()

async def generate_images():
    job_prompt = args.prompt
    job_image_url = args.image_url
    job_negative_prompt = args.negative_prompt
    job_num_samples = args.num_samples
    os.makedirs('saved', exist_ok=True)
    try:
        images_pil = inference_ip_model(ip_model, job_prompt, job_image_url, job_negative_prompt, job_num_samples)
        save_pil_images(images_pil)
        links = upload_to_cloud()
        return {"links": links}
    except Exception as e:
        return {"error": str(e)}

# Add an entry point to run the generate_images function
if __name__ == "__main__":
    result = (generate_images())
    print(result)
