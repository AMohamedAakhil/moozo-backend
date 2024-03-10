from huggingface_hub import hf_hub_download

def download_models():
    hf_hub_download(
        repo_id='h94/IP-Adapter-FaceID',
        filename='ip-adapter-faceid_sdxl.bin',
        local_dir='IP-Adapter-FaceID')
    hf_hub_download(
        repo_id='SG161222/RealVisXL_V3.0',
        local_dir='Models')
    hf_hub_download(
        repo_id='h94/IP-Adapter',
        filename='sdxl_models/image_encoder/config.json',
        local_dir='IP-Adapter')
    hf_hub_download(
        repo_id='h94/IP-Adapter',
        filename='sdxl_models/image_encoder/pytorch_model.bin',
        local_dir='IP-Adapter')
    
download_models()