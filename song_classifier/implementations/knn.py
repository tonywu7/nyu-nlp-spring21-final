# Copyright 2021 Nour Abdelmoneim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain
from typing import Dict, List, Set, Tuple

import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier

from ..scoring import export, print_score, score, stats
from ..settings import CATEGORIES, KEYWORDS, TESTING_RATIO
from ..training import convert_songs, samples
from .prep import init_nltk
from .tfidf import master_tf_idf

TFIDFVectors = Dict[str, Dict[str, float]]


def sparsematrix(vectors: TFIDFVectors, features: Set[str]) -> Tuple[List[str], np.ndarray]:
    features = {k: i for i, k in enumerate(features)}
    matrix = np.empty((len(vectors), len(features)))
    samples = [None for i in range(len(vectors))]
    for idx, (key, tfidf) in enumerate(vectors.items()):
        samples[idx] = key
        for term, val in tfidf.items():
            matrix[idx, features[term]] = val
    return samples, matrix


def kmeans(tf_idf_vectors: TFIDFVectors, features: Set[str], categories: List[str]):
    model = sklearn.cluster.KMeans(n_clusters=len(categories))
    samples, matrix = sparsematrix(tf_idf_vectors, features)
    labels = model.fit_predict(matrix)
    return labels


def knn_train(tf_idf_vectors: TFIDFVectors, features: Set[str], targets: List[Tuple[str, str]], categories: List[str]):
    model = KNeighborsClassifier(n_neighbors=7)
    samples, matrix = sparsematrix(tf_idf_vectors, features)
    labels = np.array(targets)
    model.fit(matrix, labels[:, 1])
    return model


def knn_classify(model: KNeighborsClassifier, features: Set[str], test_vectors: TFIDFVectors):
    samples, matrix = sparsematrix(test_vectors, features)
    labels = model.predict(matrix)
    return dict(zip(samples, labels))


def run():
    init_nltk()

    print('Loading songs')
    training, testing, train_labels, test_truths = samples(TESTING_RATIO, CATEGORIES, KEYWORDS)
    train_lyrics, train_wordbag, train_titles = convert_songs([*chain(*training.values())])
    test_lyrics, test_wordbag, test_titles = convert_songs([*chain(*testing.values())])
    features = train_wordbag | test_wordbag

    print('Calculating tf-idf')
    train_tfidf = master_tf_idf(train_lyrics, train_titles, features)
    test_tfidf = master_tf_idf(test_lyrics, test_titles, features)

    print('Fitting')
    model = knn_train(train_tfidf, features, [*train_labels.items()], CATEGORIES)
    print('Classifying')
    predictions = dict(knn_classify(model, features, test_tfidf))

    stats(predictions, test_truths, CATEGORIES)
    scores = score(predictions, test_truths, CATEGORIES)
    print_score(scores)
    export(dict(predictions), test_truths)