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

import math
import re
import string
import uuid
from collections import defaultdict
from itertools import chain
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

from ..collector import samples
from ..scoring import print_score, score, stats


def init_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


def term_frequencies(tokens: List[str], test: Callable[[str], bool] = lambda t: True,
                     transform: Callable[[str], str] = lambda t: t) -> Dict[str, int]:
    freq = defaultdict(int)
    for t in tokens:
        t = transform(t)
        if test(t):
            freq[t] += 1
    return freq


def overlay(*stats: Dict[str, int]) -> Dict[str, int]:
    """Accumulate multiple term frequency table into one."""
    combined = defaultdict(int)
    for d in stats:
        for k, v in d.items():
            combined[k] += v
    return combined


STOP_WORDS = {
    'a', 'the', 'an', 'and', 'or', 'but', 'about', 'above', 'after', 'along', 'amid', 'among',
    'as', 'at', 'by', 'for', 'from', 'in', 'into', 'like', 'minus', 'near', 'of', 'off', 'on',
    'onto', 'out', 'over', 'past', 'per', 'plus', 'since', 'till', 'to', 'under', 'until', 'up',
    'via', 'vs', 'with', 'that', 'can', 'cannot', 'could', 'may', 'might', 'must',
    'need', 'ought', 'shall', 'should', 'will', 'would', 'have', 'had', 'has', 'having', 'be',
    'is', 'am', 'are', 'was', 'were', 'being', 'been', 'get', 'gets', 'got', 'gotten',
    'getting', 'seem', 'seeming', 'seems', 'seemed',
    'enough', 'both', 'all', 'your' 'those', 'this', 'these',
    'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',
    'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',
    'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',
    'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',
    'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',
    'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',
    'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',
    'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace',
    'anything', 'anytime' 'anywhere', 'everybody', 'everyday',
    'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',
    'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',
    'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their', 'theirs',
    'you', 'your', 'yours', 'me', 'my', 'mine', 'I', 'we', 'us', 'much', 'and/or',
    *string.punctuation,
    'verse', 'chorus', 'intro', 'repeat', 'instrumental', 'vocal', 'pre',
}

ACCEPT_TAGS = {
    # 'FW',
    'JJ', 'JJR', 'JJS',
    'NN', 'NNP', 'NNPS', 'NNS',
    'RB', 'RBR', 'RBS',
    # 'UH',
    # 'VB', 'VBD', 'VBZ', 'VBG', 'VBP', 'VBN',
}

ALPHABETS = re.compile(r'[A-Za-z]')
SPLITTING = re.compile(r'\W+')


class Document:
    lemmatize = WordNetLemmatizer().lemmatize

    def __init__(self, text: str, title=None, label=None):
        self.text: List[str] = nltk.tokenize.word_tokenize(text)
        self.title = title
        self.label = label
        self.id = uuid.uuid4()

    def postprocess_tokens(self, *functions):
        tokens = self.text
        for f in functions:
            tokens = f(tokens)
        self.text = tokens

    @classmethod
    def filter_stop_words(cls, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in STOP_WORDS]

    @classmethod
    def to_lower(cls, tokens: List[str]) -> List[str]:
        return [t.lower() for t in tokens]

    @classmethod
    def filter_by_pos(cls, tokens: List[str]) -> List[str]:
        pos = nltk.pos_tag(tokens)
        tokens = [t for t, p in pos if p in ACCEPT_TAGS]
        return tokens

    @classmethod
    def strip_punctuation(cls, tokens: List[str]) -> List[str]:
        return [t.strip(string.punctuation) for t in tokens]

    @classmethod
    def remove_non_alphabetic(cls, tokens: List[str]) -> List[str]:
        return [t for t in tokens if ALPHABETS.search(t)]

    @classmethod
    def keep_min_length(cls, tokens: List[str], length=3) -> List[str]:
        return [t for t in tokens if len(t) >= length]

    @classmethod
    def lemmatized(cls, tokens: List[str]) -> List[str]:
        lemmatized = []
        for word in tokens:
            lemma = cls.lemmatize(word)
            if lemma != word:
                lemmatized.append(lemma)
        return [*tokens, *lemmatized]

    @classmethod
    def split_punctuation(cls, tokens: List[str]) -> List[str]:
        occurrences = []
        for word in tokens:
            parts = SPLITTING.split(word)
            if len(parts) > 1:
                occurrences.extend(parts)
        return [*tokens, *occurrences]

    @property
    def word_set(self) -> Set[str]:
        return self.term_freq.keys()

    def calc_freq(self):
        """Calculate term frequency for this document."""
        self.term_freq = term_frequencies(self.text, lambda t: t not in STOP_WORDS)

    def calc_mag(self, idf: Dict[str, int]) -> float:
        """Calculate the length of the document's vector (part of the denominator of the tf-idf cosine)."""
        self.magnitude = math.sqrt(sum((self.term_freq[word] * idf[word]) ** 2 for word in self.word_set))


class Clusters:
    """Co-occurrence finding using pointwise mutual information."""

    lemmatize = WordNetLemmatizer().lemmatize

    NEIGHBORHOOD_SIZE = 10
    IDF_TARGET = math.log(1400 / 3)
    PMI_TARGET = 10

    def __init__(self, idf: Dict[str, int]):
        self.ind_occur: Dict[str, int] = defaultdict(int)
        self.co_occur: Dict[Tuple[str, str], int] = defaultdict(int)
        self.total: int = 0
        self.targets: Set[str] = {w for w, v in idf.items() if v >= self.IDF_TARGET}
        self.pmi: Dict[Tuple[str, str], float] = {}
        self.map: Dict[str, Set[str]] = defaultdict(set)

    def add(self, doc: Document):
        self.ind_occur = overlay(self.ind_occur, doc.term_freq)
        self.total += len(doc.text)
        size = self.NEIGHBORHOOD_SIZE
        cooccurrences: Set[Tuple[int, int]] = set()
        for i, word in enumerate(doc.text):
            word = self.lemmatize(word)
            if word in self.targets:
                neighborhood = doc.text[i - size:i + size]
                k = i - size
                if k < 0:
                    k = 0
                for j, neighbor in enumerate(neighborhood, k):
                    neighbor = self.lemmatize(neighbor)
                    if neighbor in self.targets and word != neighbor:
                        cooccurrences.add(tuple(sorted([i, j])))
        for i, j in cooccurrences:
            self.co_occur[tuple(sorted([doc.text[i], doc.text[j]]))] += 1

    def calc_pmi(self):
        N = self.total
        for words, f_joint in self.co_occur.items():
            w1, w2 = words
            f_1 = self.ind_occur[w1]
            f_2 = self.ind_occur[w2]
            if not f_1 or not f_2:
                continue
            self.pmi[w1, w2] = math.log(f_joint / N / (f_1 / N) / (f_2 / N))

    def map_clusters(self):
        for pair, pmi in sorted(self.pmi.items(), key=lambda t: t[1], reverse=True):
            if pmi < self.PMI_TARGET:
                break
            w1, w2 = pair
            self.map[w1].add(w2)
            self.map[w2].add(w1)

    def modify_query(self, doc: Document):
        to_add = set()
        for word in doc.text:
            cluster = self.map[word]
            to_add |= cluster
        doc.text.extend(to_add)


class DocumentCollection:
    def __init__(self):
        self.docs: Dict[int, Document] = {}
        self.terms: Dict[str, int]
        self.idf: Dict[str, int]
        self.filter: Set[int] = self.docs.keys()

    def __getitem__(self, idx: int):
        return self.docs[idx]

    def add(self, doc: Document):
        self.docs[doc.id] = doc

    def calc_idf(self):
        self.terms = defaultdict(int)
        for doc in self.docs.values():
            for word in doc.word_set:
                self.terms[word] += 1
        N = len(self.docs)
        self.idf = defaultdict(int, {k: math.log(N / v) for k, v in self.terms.items() if v})

    def calc_mag(self, idf: Dict[str, int], **kwargs):
        for d in self.docs.values():
            d.calc_mag(idf, **kwargs)

    @property
    def word_set(self) -> Set[str]:
        words = set()
        for doc in self.docs.values():
            words |= doc.word_set
        return words

    def __iter__(self) -> Iterator[Document]:
        for idx, doc in self.docs.items():
            if idx in self.filter:
                yield doc


def process_cluster(queries: DocumentCollection, documents: DocumentCollection):
    clusters = Clusters(documents.idf)
    for doc in documents:
        clusters.add(doc)
    clusters.calc_pmi()
    clusters.map_clusters()
    for doc in queries:
        clusters.modify_query(doc)
        doc.calc_freq()


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


def tf_idf(q: Document, idf: Dict[str, int], word_set: Optional[Set[str]] = None) -> Dict[str, int]:
    word_set = word_set or q.word_set
    return {word: q.term_freq[word] * idf[word] for word in word_set}


def tf_idf_sqr(q: Document, d: Document, idf: Dict[str, int], word_set: Optional[Set[str]] = None) -> Dict[str, int]:
    word_set = word_set or q.word_set & d.word_set
    return {word: q.term_freq[word] * d.term_freq[word] * idf[word] ** 2
            for word in word_set}


def similarity_tf_idf(q: Document, d: Document, idf: Dict[str, int]) -> float:
    """Calculate tf-idf weighted cosine similarity score.

    Expects the length of the document/query vertex to be precalculated
    and stored in the `Document.magnitude` attribute.
    """
    s1 = tf_idf_sqr(q, d, idf)
    try:
        return sum(s1.values()) / (q.magnitude * d.magnitude)
    except ZeroDivisionError:
        return 0.0


def vector_distances(docs: DocumentCollection) -> pd.DataFrame:
    keys = [d.label for d in docs]
    dist = pd.DataFrame(index=keys, columns=keys)
    for i in docs:
        for j in docs:
            sim = similarity_tf_idf(i, j, docs.idf)
            dist.loc[i.label, j.label] = sim
    return dist


def run(ratio, categories, keywords, postprocessors, min_weight, *args, **kwargs):
    init_nltk()

    print('Loading songs', flush=True)

    training, testing = samples(ratio, categories, keywords, min_weight)

    print('Dataset stats:', flush=True)
    for k in training.keys():
        print(f'{k}: training={len(training[k])} testing={len(testing[k])}', flush=True)

    metadocs = {k: Document('\n '.join([s.lyrics for s in ss]), label=k) for k, ss in training.items()}
    testdocs = {k: [Document(s.lyrics, title=s.title, label=k) for s in ss] for k, ss in testing.items()}

    print('Training', flush=True)

    for doc in chain(metadocs.values(), *testdocs.values()):
        doc.postprocess_tokens(*postprocessors)
        doc.calc_freq()

    vectors = DocumentCollection()
    for doc in metadocs.values():
        vectors.add(doc)

    queries = DocumentCollection()
    for doc in chain(*testdocs.values()):
        queries.add(doc)

    vectors.calc_idf()
    vectors.calc_mag(vectors.idf)
    process_cluster(queries, vectors)
    queries.calc_mag(vectors.idf)

    print(vector_distances(vectors), flush=True)

    print('Testing', flush=True)

    predictions = {}
    ground_truths = {}

    for song, pred, conf in evaluate(queries, vectors):
        predictions[song.id] = pred
        ground_truths[song.id] = song.label

    stats(predictions, ground_truths, categories)
    scores = score(predictions, ground_truths, categories)
    print_score(*scores)

    return vectors
