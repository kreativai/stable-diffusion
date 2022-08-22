import torch
import os
from diffusers import StableDiffusionPipeline
from huggingface_hub.hf_api import HfApi
# make sure you're logged in with `huggingface-cli login`
token_value = str(open('./token.txt').read()).strip()
from huggingface_hub.commands.user import _login; 
_login(HfApi(), token=token_value)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)  
pipe = pipe.to("cuda")
from torch import autocast

prompt = "a photograph of an astronaut riding a horse"
with autocast("cuda"):
  image = pipe(prompt)["sample"][0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

# Now to display an image you can do either save it such as:
os.makedirs('./outputs', exist_ok=True)
image.save("./outputs/astronaut_rides_horse.png")
