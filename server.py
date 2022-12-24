# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

import uvicorn
import subprocess
import app as user_src
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
user_src.init()

# Create the http server app
app = FastAPI()

# Define your model parameters here:
class Item(BaseModel):
    prompt: str
    negative: str
    height: int 
    width: int 
    num_inference_steps: int 
    guidance_scale: int 
    input_seed: int

# Healthchecks verify that the environment is correct on Banana Serverless
@app.get('/healthcheck')
async def healthcheck():
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True

    return {"state": "healthy", "gpu": gpu}


# Inference POST handler at '/' is called for every http call from Banana
@app.post('/') 
async def inference(model_inputs: Item):
    output = user_src.inference(vars(model_inputs))
    return output


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)