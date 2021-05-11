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
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier

from ..collector import samples
from ..scoring import print_score, score, stats
from ..training import convert_songs, sort_songs
from .tfidf import master_tf_idf
from .tfidf2 import init_nltk

TFIDFVectors = Dict[str, Dict[str, float]]


def export_matrix(matrix: np.ndarray, filename: str):
    df = pd.DataFrame(matrix)
    df.to_csv(filename, index=False, header=False)


def sparsematrix(vectors: TFIDFVectors, features: Set[str], name: str = None) -> Tuple[List[str], np.ndarray]:
    features = {k: i for i, k in enumerate(features)}
    matrix = np.empty((len(vectors), len(features)))
    samples = [None for i in range(len(vectors))]
    for idx, (key, tfidf) in enumerate(vectors.items()):
        samples[idx] = key
        for term, val in tfidf.items():
            matrix[idx, features[term]] = val
    if name and False:
        export_matrix(matrix, name)
    return samples, matrix


def kmeans(tf_idf_vectors: TFIDFVectors, features: Set[str], categories: List[str]):
    model = sklearn.cluster.KMeans(n_clusters=len(categories))
    samples, matrix = sparsematrix(tf_idf_vectors, features)
    labels = model.fit_predict(matrix)
    return labels


def knn_train(tf_idf_vectors: TFIDFVectors, features: Set[str], targets: List[Tuple[str, str]], neighbors):
    model = KNeighborsClassifier(n_neighbors=neighbors)
    # samples, matrix = sparsematrix(tf_idf_vectors, features, 'matrix-training.csv')
    samples, matrix = sparsematrix(tf_idf_vectors, features)
    labels = np.array(targets)
    model.fit(matrix, labels[:, 1])
    return samples, model


def knn_classify(model: KNeighborsClassifier, features: Set[str], test_vectors: TFIDFVectors):
    # samples, matrix = sparsematrix(test_vectors, features, 'matrix-testing.csv')
    samples, matrix = sparsematrix(test_vectors, features)
    labels = model.predict(matrix)
    return dict(zip(samples, labels))


def run(ratio, categories, keywords, postprocessors, min_weight, knn_n_neighbors, *args, **kwargs):
    init_nltk()

    print('Loading songs', flush=True)

    training, testing = samples(ratio, categories, keywords, min_weight)
    train_labels, test_truths = sort_songs(training, testing)

    train_lyrics, train_wordbag, train_titles = convert_songs([*chain(*training.values())], postprocessors)
    test_lyrics, test_wordbag, test_titles = convert_songs([*chain(*testing.values())], postprocessors)
    features = train_wordbag | test_wordbag

    print('Calculating tf-idf', flush=True)
    train_tfidf = master_tf_idf(train_lyrics, train_titles, features)
    test_tfidf = master_tf_idf(test_lyrics, test_titles, features)

    print('Fitting', flush=True)
    titlemap, model = knn_train(train_tfidf, features, [*train_labels.items()], knn_n_neighbors)
    # (pd.DataFrame([[k, train_labels[k], ' '.join(train_lyrics[i])] for i, k in enumerate(titlemap)])
    #  .to_csv('labels-training.csv', index=False, header=False))

    print('Classifying', flush=True)
    predictions = dict(knn_classify(model, features, test_tfidf))
    # (pd.DataFrame([[k, test_truths[k], ' '.join(test_lyrics[i])] for i, k in enumerate(predictions)])
    #  .to_csv('labels-testing.csv', index=False, header=False))

    stats(predictions, test_truths, categories)
    scores = score(predictions, test_truths, categories)
    print_score(*scores)
    # export(dict(predictions), test_truths)
