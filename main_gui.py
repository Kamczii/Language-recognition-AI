import pickle
import tkinter as tk

import numpy as np
from PIL import ImageTk, Image
from keras import Sequential
from keras.saving.save import load_model

model_type = "trigrams"
model: Sequential = load_model(model_type + "_recognition")
with open(model_type + '_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open(model_type + '_labels.pkl', 'rb') as f:
    labels = pickle.load(f)


def predict(sentence: str):
    X = vectorizer.fit_transform([sentence])
    predictions = model.predict(X)
    return labels[np.argmax(predictions[0])]

# def predict(string):
#     return string


def language_recognizer():
    eng_flag = ImageTk.PhotoImage(Image.open("./static/eng.png"))
    deu_flag = ImageTk.PhotoImage(Image.open("./static/deu.png"))
    esp_flag = ImageTk.PhotoImage(Image.open("./static/esp.png"))
    por_flag = ImageTk.PhotoImage(Image.open("./static/por.png"))
    default = ImageTk.PhotoImage(Image.open("./static/default.png"))
    image_list = [eng_flag, deu_flag, esp_flag, por_flag, default]

    prediction = predict(text_input.get())

    if prediction == "eng":
        lang_lab = tk.Label(root, image=image_list[0], width=300, height=200)
        lang_lab.photo = image_list[0]
    elif prediction == "deu":
        lang_lab = tk.Label(root, image=image_list[1], width=300, height=200)
        lang_lab.photo = image_list[1]
    elif prediction == "esp":
        lang_lab = tk.Label(root, image=image_list[2], width=300, height=200)
        lang_lab.photo = image_list[2]
    elif prediction == "por":
        lang_lab = tk.Label(root, image=image_list[3], width=300, height=200)
        lang_lab.photo = image_list[3]
    else:
        lang_lab = tk.Label(root, image=default, width=300, height=200)
        lang_lab.photo = default

    # lang_label = tk.Label(root, text="Wykryty język to: " + text_input.get(), padx=10, pady=10)
    # lang_label.grid(row=3, column=0, columnspan=3)

    lang_lab.grid(row=4, column=0, columnspan=3)


root = tk.Tk()
root.title("Projekt SI")
root.iconbitmap("./static/icon.ico")

title = tk.Label(root, text="Wykrywanie języka")
title.grid(row=0, column=0, columnspan=4, padx=20, pady=20)

info = tk.Label(root, text="Wpisz zdanie w dowolnym języku") # TODO: Info o wspieranych językach tzn. tych co są w tablicy "labels"
info.grid(row=1)

text_input = tk.Entry(root, width=60, borderwidth=5, relief=tk.FLAT)
text_input.grid(row=2)

b = tk.Button(root, text='SUBMIT', command=language_recognizer, width=50, padx=5, pady=15)
b.grid(row=3, column=0, columnspan=3)

root.mainloop()
