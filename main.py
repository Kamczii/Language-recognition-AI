import pickle
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("trigrams_recognition")
with open('count_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# X = vectorizer.fit_transform(["Whats up dude? Here we go again so listen to me. I want to be your friend."])
X = vectorizer.fit_transform(["Haben Sie Lust mit zu mir zu kommen und alles das zu tun, was ich allen anderen morgen sowieso erzählen werde?."])
# X = vectorizer.fit_transform(["Vale, puedes quedarte a mi lado, siempre que no hables sobre la tempertura."])
# X = vectorizer.fit_transform(["Whats up dude? Here we go again so listen to me. I want to be your friend."])
predictions = model.predict(X)

print(encoder.classes_[np.argmax(predictions[0])])