# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time

    repo_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(
        repo_id, 
        subfolder="scheduler", 
        prediction_type="v_prediction"
    )
    model = StableDiffusionPipeline.from_pretrained(
        repo_id, 
        torch_dtype=torch.float16, 
        revision="fp16", 
        scheduler=scheduler
    )

if __name__ == "__main__":
    download_model()
