# Copyright 2021 Tony Wu +https://github.com/tonywu7/
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
from typing import Dict, Hashable, List, Tuple

from ..app import TEXT_PROCESSORS, get_settings
from ..database import Song
from .tfidf2 import (Document, DocumentCollection, pairwise_vector_sim,
                     process_cluster, similarity_tf_idf)


def evaluate(queries: DocumentCollection, documents: DocumentCollection):
    """Calculate similarity scores for all queries and documents."""
    for q in queries:
        scores: Dict[str, float] = {}
        for d in documents:
            s = similarity_tf_idf(q, d, documents.idf)
            scores[d.label] = s
        for d, s in sorted(scores.items(), key=lambda t: t[1], reverse=True):
            yield q, d, s
            break


def run(training: Dict[str, List[Song]], testing: Dict[str, List[Song]]) -> Tuple[Dict[Hashable, str], Dict[Hashable, str]]:
    """Run the cosine similarity algorithm.

    Parameters
    ----------
    training : Dict[str, List[Song]]
        Training dataset
    testing : Dict[str, List[Song]]
        Testing dataset

    Returns
    -------
    Tuple[Dict[Hashable, str], Dict[Hashable, str]]
        Predictions and truths as dictionaries
    """
    log = logging.getLogger('cosine')

    metadocs = {k: Document('\n '.join([s.lyrics for s in ss]), label=k) for k, ss in training.items()}
    testdocs = {k: [Document(s.lyrics, title=s.title, label=k) for s in ss] for k, ss in testing.items()}

    log.info('Preprocessing text')
    processors = get_settings()[TEXT_PROCESSORS]

    for doc in chain(metadocs.values(), *testdocs.values()):
        doc.postprocess_tokens(*processors)
        doc.calc_freq()

    vectors = DocumentCollection()
    for doc in metadocs.values():
        vectors.add(doc)

    queries = DocumentCollection()
    for doc in chain(*testdocs.values()):
        queries.add(doc)

    log.info('Calculating tf-idf')

    vectors.calc_idf()
    vectors.calc_mag(vectors.idf)
    process_cluster(queries, vectors)
    queries.calc_mag(vectors.idf)

    log.info('Category pairwise similarities\n' + str(pairwise_vector_sim(vectors)))

    log.info('Testing')

    predictions = {}
    ground_truths = {}

    for song, pred, conf in evaluate(queries, vectors):
        predictions[song.id] = pred
        ground_truths[song.id] = song.label

    return predictions, ground_truths
