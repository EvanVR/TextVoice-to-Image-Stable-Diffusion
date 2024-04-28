import tkinter as tk
from PIL import ImageTk
from authtoken import auth_token
import speech_recognition as sr
from diffusers import StableDiffusionPipeline
import customtkinter as ctk
import torch

app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float32, use_auth_token=auth_token)

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        speech_text = recognizer.recognize_google(audio)
        print("You said:", speech_text)
        prompt.delete(0, tk.END)  
        prompt.insert(0, speech_text)  
        generate() 
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error: {0}".format(e))


def generate():
    text = prompt.get()
    image = pipe(text, guidance_scale=8.5)["images"][0]
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=recognize_speech)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
