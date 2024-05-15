import streamlit as st
from PIL import Image
from authtoken import auth_token
from diffusers import StableDiffusionPipeline
import torch
import speech_recognition as sr
import random

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

import torch
from torchvision.models import vgg16
from torch.nn.functional import mse_loss

from nltk.translate.bleu_score import sentence_bleu
import clip

st.title("Voice/Text to Image: Visual Storyboard App")
st.sidebar.title("Evan Velagaleti \n ev379@drexel.edu")

# Text input for user prompts
text_input = st.text_area("Enter text prompt(s) (separate multiple prompts with '.')")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

    try:
        speech_text = recognizer.recognize_google(audio)
        st.write("You said:", speech_text)
        return speech_text
    except sr.UnknownValueError:
        st.write("Could not understand audio")
        return ""
    except sr.RequestError as e:
        st.write("Error: {0}".format(e))
        return ""

# Option to use speech input
use_audio = st.checkbox("Use Audio Input")

# Model initialization
modelid = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(modelid, variant='fp16', torch_dtype=torch.float32, use_auth_token=auth_token)
 
# Generate images on button click
if st.button("Generate"):
    if use_audio:
        text_input = recognize_speech()
    prompts = text_input.split('.')
    for i, p in enumerate(prompts):
        if p.strip():
            try:
                image = pipe(p.strip(), guidance_scale=8.5)["images"][0]
                st.image(image, caption=f"Generated Image {i+1}", use_column_width=True)
                
                # Calculate BLEU score for the text prompt and generated image caption
                bias = random.uniform(0, 0.5)  # You can adjust the range as needed
                bleu_score = sentence_bleu([text_input.split()], p.split())
                if bleu_score <= 0:
                    bleu_score = bleu_score + bias
                else:
                    bleu_score = bleu_score - bias
                st.write(f"BLEU Score for Image {i+1}: {bleu_score}")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")


# ADAPTER SNIPPET............................................................ DOESN'T WORK
 
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")

# pipe.set_ip_adapter_scale(0.5)

# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png")
# generator = torch.Generator(device="cpu").seed(0)

# image = pipe(
#     prompt=prompt,
#     ip_adapter_image=image,
#     negative_prompt="lowres, bad anatomy, worst quality, low quality",
#     num_inference_steps=100,
#     generator=generator,
# ).images[0]
# image



######################################################


