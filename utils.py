from collections import Counter
from dataclasses import dataclass
from typing import List

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.utils import np_utils, Sequence

import pandas as pd

# CONSTANTS
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

languages = ["eng", "deu", "spa", "ita", "por", "fra"]  # wspierane języki
sentences_by_lang = 200000  # ile zdań bierzemy pod uwagę w poszczególnych językach
trigrams_by_lang = 200  # ile trigramów z danego języka bierzemy pod uwagę


# HELPER FUNCTIONS
def get_trigrams(sentence: str):
    """
    Wydobywa trigramy ze zdania.
    :param sentence: Zdanie w postaci łańcucha znaków.
    :return: Wszystkie trigramy znajdujące się w zdaniu.
    """
    trigrams = []
    for i, c in enumerate(sentence):
        trigram = sentence[i:(i + 3)]
        if len(trigram) == 3:
            trigrams.append(trigram)
    return trigrams


def get_trigrams_sets(df: pd.DataFrame) -> tuple[set, dict]:
    """
    Funkcja dla każdego zdania w tabeli zlicza ilość wystąpień trigramów,
    następnie dla każdego języka zwraca  TRIGRAMS_BY_LANG najpopularniejszych trigramów.
    Na końcu tworzy zbiór składający się z najpopularniejszych trigramów każdego języka.
    :param df: DataFrame zawierająca kolumny "sentence" oraz "lang"
    :return: biór najpopularniejszych trigramów ogólnie, słownik [lang] -> [trigramy w danym języku...]
    """
    all_trigrams: set[str] = set()
    lang_trigrams: dict[str, list[tuple[str, int]]] = dict()
    for lang in languages:
        trigrams = Counter()
        series = df[df["lang"] == lang]["sentence"]
        for sentence in series:
            tri = get_trigrams(sentence)
            trigrams.update(tri)
        trigrams += Counter()  # usuwa elemnty z count=0
        mc = trigrams.most_common(trigrams_by_lang)
        all_trigrams.update([v[0] for v in mc])
        lang_trigrams[lang] = mc
    return all_trigrams, lang_trigrams


def encode(y, encoder):
    """
    OneHotEncoding języków
    y: lista języków
    return: lista list postaci [[1.,0.,0.,0.], [0.,1.,0.,0.].... ]]
    """

    y_encoded = encoder.transform(y)
    y_dummy = np_utils.to_categorical(y_encoded)

    return y_dummy


def get_words_set(df: pd.DataFrame):
    all_words: set[str] = set()
    for lang in ["eng", "deu", "spa", "ita"]:
        words = Counter()
        series = df[df["lang"] == lang]["sentence"]
        for sentence in series:
            words.update([w for w in sentence.split(" ") if len(w) > 1])

        mc = words.most_common(trigrams_by_lang)
        all_words.update([v[0] for v in mc])
    return all_words


def prepare_dataframe() -> pd.DataFrame:
    """
    Filtruje potrzebne dane do dalszych obliczeń.
    :return: Dataframe postaci
        | lang  | sentence
    0   | "eng" | "What is it?"
    1   | "deu" | "Ich bin da!"
    """

    # Wczytujemy plik csv tworząc DF z dwoma kolumanmi "lang" oraz "sentence"
    csv_file = pd.read_csv('sentences.csv', on_bad_lines='skip', sep='\t', index_col=0, names=["lang", "sentence"])

    # Filtrujemy tabelę, zostawiamy tylko wspierane języki. Dla każdego języka zostawiamy SENTENCES_BY_LANG zdań.
    dataset = csv_file[csv_file['lang'].isin(languages)]
    results = pd.DataFrame(columns=["lang", "sentence"])
    for l in languages:
        ds = dataset[dataset["lang"] == l].sample(sentences_by_lang)
        results = pd.concat([results, ds])
    results["sentence"] = results[
        "sentence"].str.lower()  # pomijamy wielkość liter, aby nie traktować osobno np. "He" i "he"
    return results


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skalowanie min max
    :param df: DataFrame zawierający featury
    :return: Każda kolumna ma wartości w zakresie 0.0-1.0
    """
    train_min = df.min()  # najmniejsza wartość z każdej kolumny
    train_max = df.max()  # największa wartość z każdej kolumny

    df = (df - train_min) / (train_max - train_min)

    assert not df.isnull().values.any(), "NaN w wynikowym DataFrame"

    return df


@dataclass
class FFN_Hyperparams:
    # stale (zalezne od problemu)
    num_inputs: int
    num_outputs: int

    # do znalezienia optymalnych (np. metodą Random Search) [w komentarzu zakresy wartości, w których można szukać]
    hidden_dims: List[int]  # [10] -- [100, 100, 100]
    activation_fcn: str  # wybor z listy, np: ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'softplus']


def build_model(hp: FFN_Hyperparams):
    """
    Kompiluje model na podstawie podanych parametrów
    :param hp: parametry modelu
    :return: skompilowany model gotowy do treningu
    """
    model = Sequential()

    for i, dim in enumerate(hp.hidden_dims):
        if i == 0:
            model.add(Dense(dim, activation=hp.activation_fcn, input_shape=[hp.num_inputs]))
        else:
            model.add(Dense(dim, activation=hp.activation_fcn))

    model.add(Dense(hp.num_outputs, name='wyjsciowa',
                    activation='softmax'))  # model regresyjny, activation=None w warstwie wyjściowej

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def create_feature_dictionary(elements: set) -> dict:
    """
    Tworzy słownik featurów, każdemu featerowi przypisuje index
    :param elements: zbiór featerów np. trigramy albo słowa
    :return: słownik postaci np. "word":0, "bird":1, "achtung": 2......
    """
    dic = dict()
    for i, t in enumerate(elements):
        dic[t] = i
    return dic


def create_encoder() -> LabelEncoder:
    """
    Tworzy koder, który można wykorzystać do kodowania języków
    :return:
    """
    encoder = LabelEncoder()
    encoder.fit(languages)
    return encoder


def test_model(model: Sequential, encoder: LabelEncoder, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
    predictions = model.predict(X_test)
    labels = [encoder.classes_[np.argmax(prediction)] for prediction in predictions]
    correct = [encoder.classes_[np.argmax(y)] for y in y_test]
    return accuracy_score(correct, labels)

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y