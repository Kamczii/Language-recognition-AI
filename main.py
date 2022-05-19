from collections import Counter

import matplotlib.pyplot as pt
import numpy as np
import pandas as pd

supported_langs = ["eng", "deu"]  # , "fra", "rus", "spa", "pol"]


def get_trigrams(sentence: str):
    sentence = sentence.replace(" ", "_")
    trigrams = []
    for i, c in enumerate(sentence):
        trigrams.append(sentence[i:(i + 3)])
    return trigrams


def create_matrix(dataset):
    most_common_trigrams = set()
    most_common_trigrams_by_lang = {}
    for lang in supported_langs:
        trigrams = Counter()
        series = dataset[dataset['lang'] == lang]["sentence"].head(20)
        for sentence in series:
            trigrams.update(get_trigrams(sentence))
        mc = trigrams.most_common(20)
        most_common_trigrams.update(mc)
        most_common_trigrams_by_lang[lang] = mc


dataset = pd.read_csv('sentences.csv', on_bad_lines='skip', sep='\t', index_col=0, names=["index", "lang", "sentence"])

supported_languages_mask = dataset['lang'].isin(supported_langs)
sentences = dataset[supported_languages_mask]

create_matrix(sentences)


## trigrams
# eng
# deu
# fra
# rus
