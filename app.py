import tkinter as tk
from PIL import ImageTk
from authtoken import auth_token
import speech_recognition as sr
from diffusers import StableDiffusionPipeline
import customtkinter as ctk
import torch
from tkinter import messagebox

app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(modelid, variant='fp16', torch_dtype=torch.float32, use_auth_token=auth_token)

# def recognize_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening...")
#         audio = recognizer.listen(source)

#     try:
#         speech_text = recognizer.recognize_google(audio)
#         print("You said:", speech_text)
#         prompt.delete(0, tk.END)  
#         prompt.insert(0, speech_text)  
#         generate() 
#     except sr.UnknownValueError:
#         print("Could not understand audio")
#     except sr.RequestError as e:
#         print("Error: {0}".format(e))


def generate():
    text = prompt.get()
    prompts = text.split('.')  
    row = 0
    for i, p in enumerate(prompts):
        if p.strip(): 
            try:
                image = pipe(p.strip(), guidance_scale=8.5)["images"][0]
                image.save(f'generatedimage_{i}.png')
                img = ImageTk.PhotoImage(image)
                label = ctk.CTkLabel(app, image=img)
                label.image = img  # Keep a reference to the image to prevent it from being garbage collected
                label.grid(row=row, column=0)
                row += 1
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()







# import tkinter as tk
# from PIL import ImageTk, Image
# from authtoken import auth_token
# from diffusers import StableDiffusionPipeline
# import customtkinter as ctk
# import torch
# from tkinter import messagebox
# from concurrent.futures import ThreadPoolExecutor

# app = tk.Tk()
# app.geometry("532x632")
# app.title("Stable Bud")
# ctk.set_appearance_mode("dark")

# prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
# prompt.place(x=10, y=10)

# lmain = ctk.CTkLabel(app, height=512, width=512)
# lmain.place(x=10, y=110)

# modelid = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(modelid, variant='fp16', torch_dtype=torch.float32, use_auth_token=auth_token)

# executor = ThreadPoolExecutor(max_workers=5)  # Adjust max_workers based on your needs

# def generate_images(prompts):
#     images = []
#     for prompt in prompts:
#         try:
#             image = pipe(prompt.strip(), guidance_scale=8.5)["images"][0]
#             images.append(image)
#         except Exception as e:
#             messagebox.showerror("Error", f"An error occurred: {e}")
#     return images

# def generate():
#     text = prompt.get()
#     prompts = text.split('.')
#     prompts = [p.strip() for p in prompts if p.strip()]
    
#     future = executor.submit(generate_images, prompts)
#     future.add_done_callback(display_images)

# def display_images(future):
#     images = future.result()
#     if images:
#         row, col = 0, 0
#         for i, image in enumerate(images):
#             image.save(f'generatedimage_{i}.png')
#             img = ImageTk.PhotoImage(image)
#             label = ctk.CTkLabel(app, image=img)
#             label.image = img  # Keep a reference to the image to prevent it from being garbage collected
#             label.grid(row=row, column=col)
#             col += 1
#             if col == 3:  # Change the number of columns as needed
#                 col = 0
#                 row += 1
#     else:
#         messagebox.showinfo("Info", "No images generated.")

# trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
# trigger.configure(text="Generate")
# trigger.place(x=206, y=60)

# app.mainloop()
