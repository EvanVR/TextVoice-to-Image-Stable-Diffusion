import speech_recognition as sr
import tkinter as tk

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        text_display.config(state=tk.NORMAL)
        text_display.delete('1.0', tk.END)
        text_display.insert(tk.END, text)
        text_display.config(state=tk.DISABLED)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error: {0}".format(e))

# Create the GUI
root = tk.Tk()
root.title("Audio to Text")
root.geometry("400x200")

# Create a text widget to display the recognized text
text_display = tk.Text(root, height=5, width=50)
text_display.pack(pady=20)
text_display.config(state=tk.DISABLED)

# Create a button to start the speech recognition
button = tk.Button(root, text="Start", command=recognize_speech)
button.pack(pady=10)

root.mainloop()
