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

import logging
from itertools import chain
from typing import Dict, Hashable, List, Set, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier

from ..app import N_NEIGHBORS, TEXT_PROCESSORS, get_settings
from ..database import Song
from .tfidf2 import Document, DocumentCollection, process_cluster, tf_idf

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
    samples, matrix = sparsematrix(tf_idf_vectors, features, 'matrix-training.csv')
    labels = np.array(targets)
    model.fit(matrix, labels[:, 1])
    return samples, model


def knn_classify(model: KNeighborsClassifier, features: Set[str], test_vectors: TFIDFVectors):
    samples, matrix = sparsematrix(test_vectors, features, 'matrix-testing.csv')
    labels = model.predict(matrix)
    return dict(zip(samples, labels))


def run(training: Dict[str, List[Song]], testing: Dict[str, List[Song]]) -> Tuple[Dict[Hashable, str], Dict[Hashable, str]]:
    log = logging.getLogger('cosine')

    metadocs = {k: [Document(s.lyrics, title=s.title, label=k) for s in ss] for k, ss in training.items()}
    testdocs = {k: [Document(s.lyrics, title=s.title, label=k) for s in ss] for k, ss in testing.items()}

    log.info('Preprocessing text')
    processors = get_settings()[TEXT_PROCESSORS]

    for doc in chain(*metadocs.values(), *testdocs.values()):
        doc.postprocess_tokens(*processors)
        doc.calc_freq()

    vectors = DocumentCollection()
    for doc in chain(*metadocs.values()):
        vectors.add(doc)

    queries = DocumentCollection()
    for doc in chain(*testdocs.values()):
        queries.add(doc)

    vectors.calc_idf()
    vectors.calc_mag(vectors.idf)
    process_cluster(queries, vectors)
    queries.calc_mag(vectors.idf)
    queries.calc_idf()

    log.info('Calculating tf-idf')

    global_word_set = vectors.word_set & queries.word_set

    train_tfidf = {d.id: tf_idf(d, vectors.idf, global_word_set) for d in vectors}
    test_tfidf = {d.id: tf_idf(d, queries.idf, global_word_set) for d in queries}

    truths = {d.id: d.label for d in queries}

    log.info('Fitting')
    knn_n_neighbors = get_settings()[N_NEIGHBORS]
    titlemap, model = knn_train(train_tfidf, global_word_set, [(d.id, d.label) for d in vectors], knn_n_neighbors)

    log.info('Classifying')
    predictions = dict(knn_classify(model, global_word_set, test_tfidf))

    return predictions, truths
