import pickle

from keras import Sequential
from tensorflow.keras.models import load_model
import numpy as np

model: Sequential = load_model("trigrams_recognition")
with open('trigrams_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('trigrams_labels.pkl', 'rb') as f:
    labels = pickle.load(f)
sentences = ["Whats up dude? Here we go again so listen to me. I want to be your friend.",
             "Haben Sie Lust mit zu mir zu kommen und alles das zu tun, was ich allen anderen morgen sowieso erzählen werde?.",
             "Vale, puedes quedarte a mi lado, siempre que no hables sobre la tempertura.",
             "Me lo strizzeresti, per favore?",
             "E aí, cara? Aqui vamos nós de novo, então me escute. Eu quero ser seu amigo.",
             "Aqui vamos nós de novo, então me escute"]

X = vectorizer.fit_transform(sentences)
predictions = model.predict(X)

print([labels[np.argmax(label)] for label in predictions])