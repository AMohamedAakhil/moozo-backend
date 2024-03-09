
from insightface.app import FaceAnalysis
import torch
from diffusers.utils import load_image
import numpy as np
from huggingface_hub import hf_hub_download
from imgur_python import Imgur
import os

from dotenv import load_dotenv
load_dotenv()

import cloudinary
import cloudinary.uploader
import cloudinary.api


def download_models():
    hf_hub_download(
        repo_id='h94/IP-Adapter-FaceID',
        filename='ip-adapter-faceid_sdxl.bin',
        local_dir='IP-Adapter-FaceID')
    hf_hub_download(
        repo_id='h94/IP-Adapter',
        filename='sdxl_models/image_encoder/config.json',
        local_dir='IP-Adapter')
    hf_hub_download(
        repo_id='h94/IP-Adapter',
        filename='sdxl_models/image_encoder/pytorch_model.bin',
        local_dir='IP-Adapter')


def get_image_from_url(url):
    image = load_image(url)
    image = np.array(image.convert('RGB'))  
    return image


def get_face_embedding(url):
    image = get_image_from_url(url)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(image)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    return faceid_embeds


def save_pil_images(images):
    for i, image in enumerate(images):
        image.save(f"saved/image_{i}.png")

def upload_to_cloud():
    config = cloudinary.config(secure=True)
    urls = []
    for image in os.listdir('saved'):
        img_path = os.path.join('saved', image)
        cloudinary.uploader.upload(img_path, public_id="quickstart_butterfly", unique_filename = False, overwrite=True)
        srcURL = cloudinary.CloudinaryImage("quickstart_butterfly").build_url()
        urls.append(srcURL)
    return urls
