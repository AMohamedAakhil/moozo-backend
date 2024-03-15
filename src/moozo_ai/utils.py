
from insightface.app import FaceAnalysis
import torch
from diffusers.utils import load_image
import numpy as np
import os

import cloudinary
import cloudinary.uploader
import cloudinary.api

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
    cloudinary.config( 
     cloud_name = "ddospzdve", 
     api_key = "496917724689965", 
     api_secret = "nkyggyOuBRdEwCaWEpjYIIwhf8U" 
     )
    urls = []
    for image in os.listdir('saved'):
        img_path = os.path.join('saved', image)
        res =cloudinary.uploader.upload(img_path, use_filename=False,  unique_filename = True, overwrite=True, folder='moozo_ai')
        srcURL = res.get("url")
        print("Uploaded URL: ", srcURL)
        urls.append(srcURL)
    return urls

