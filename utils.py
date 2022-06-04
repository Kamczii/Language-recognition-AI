from collections import Counter

import pandas as pd

# CONSTANTS
from keras.utils import np_utils

languages = ["eng", "deu", "spa", "ita"]  # wspierane języki
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
