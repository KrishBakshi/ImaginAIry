from diffusers import DiffusionPipeline
import gradio as gr
import numpy as np
import os
import access_token

from huggingface_hub import login
import os

# Retrieve the token from an environment variable
# access_token = access_token.access_token  # Replace with the correct variable name

# if access_token is None:
#     raise ValueError("Token is not set in the environment variable.")

# # Log in using the token
# login(token=access_token)

# Define a function that takes a text input and returns an image.
def text_to_image(text : str):
    pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
    pipe.load_lora_weights("gokaygokay/Flux-Game-Assets-LoRA-v2")
    prompt = text
    image = pipe(prompt).images[0]
    return image

# Create a Gradio interface that takes a textbox input, runs it through the text_to_image function, and returns output to an image.
demo = gr.Interface(fn=text_to_image, inputs="textbox", outputs="image")

# Launch the interface.
if __name__ == "__main__":
    demo.launch(show_error=True)
    

