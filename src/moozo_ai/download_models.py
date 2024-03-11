import os
from huggingface_hub import hf_hub_download, snapshot_download

def download_models():
    # Check if models already exist before downloading
    if not os.path.exists('/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin'):
        hf_hub_download(
            repo_id='h94/IP-Adapter-FaceID',
            filename='ip-adapter-faceid_sdxl.bin',
            local_dir='/IP-Adapter-FaceID')
    
    if not os.path.exists('/Models/RealVisXL_V3.0'):
        snapshot_download(
            repo_id='SG161222/RealVisXL_V3.0',
            local_dir='/Models')

# Execute the download_models function
download_models()


"""
    hf_hub_download(
        repo_id='h94/IP-Adapter',
        filename='sdxl_models/image_encoder/config.json',
        local_dir='/IP-Adapter-FaceID')
    hf_hub_download(
        repo_id='h94/IP-Adapter',
        filename='sdxl_models/image_encoder/pytorch_model.bin',
        local_dir='/IP-Adapter-FaceID')
"""