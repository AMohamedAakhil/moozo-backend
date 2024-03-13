
from insightface.app import FaceAnalysis
import torch
from diffusers.utils import load_image
import numpy as np
import io
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

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
    config = cloudinary.config(secure=True)
    urls = []
    for image in os.listdir('saved'):
        img_path = os.path.join('saved', image)
        cloudinary.uploader.upload(img_path, public_id="quickstart_butterfly", unique_filename = False, overwrite=True)
        srcURL = cloudinary.CloudinaryImage("quickstart_butterfly").build_url()
        urls.append(srcURL)
    return urls
