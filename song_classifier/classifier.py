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

from __future__ import annotations

import math
import re
import string
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain
from typing import (Callable, Dict, Generator, Iterable, Iterator, List, Set,
                    Tuple)

import nltk
from nltk.stem.snowball import EnglishStemmer

from .database import get_db

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
}

STEMMER: Callable[[str], str] = None
ALPHABETS = re.compile(r'[A-Za-z]')
SPLITTING = re.compile(r'\W+')


def init_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    return EnglishStemmer(True).stem


def term_frequencies(tokens: List[str], test: Callable[[str], bool] = lambda t: True,
                     transform: Callable[[str], str] = lambda t: t) -> Dict[str, int]:
    freq = defaultdict(int)
    for t in tokens:
        t = transform(t)
        if test(t):
            freq[t] += 1
    return freq


def similarity_tf_idf(q: Document, d: Document, idf: Dict[str, int]) -> float:
    """Calculate tf-idf weighted cosine similarity score.

    Expects the length of the document/query vertex to be precalculated
    and stored in the `Document.magnitude` attribute.
    """
    s1 = [q.term_freq[word] * d.term_freq[word] * idf[word] ** 2
          for word in q.word_set & d.word_set]
    try:
        return sum(s1) / (q.magnitude * d.magnitude)
    except ZeroDivisionError:
        return 0.0


def similarity_vectors(q: Document, d: Document) -> float:
    """Calculate a cosine using only term frequencies."""
    s1 = [q.term_freq[word] * d.term_freq[word] for word in q.word_set]
    s2 = [q.term_freq[word] ** 2 for word in q.word_set]
    s3 = [d.term_freq[word] ** 2 for word in q.word_set]
    try:
        return sum(s1) / (math.sqrt(sum(s2)) * math.sqrt(sum(s3)))
    except ZeroDivisionError:
        return 0.0


def overlay(*stats: Dict[str, int]) -> Dict[str, int]:
    """Accumulate multiple term frequency table into one."""
    combined = defaultdict(int)
    for d in stats:
        for k, v in d.items():
            combined[k] += v
    return combined


class Document:
    def __init__(self):
        self.id: int
        self.author: str = ''
        self.ref: str = ''

        self.text: List[str]
        self.term_freq: Dict[str, int]
        self.magnitude: float

    def parse(self):
        buf = []
        text = []
        while True:
            try:
                line = yield
            except GeneratorExit:
                text.extend(buf)
                self.text = nltk.word_tokenize(' '.join(text))
                break
            line = line.strip()
            if line[:2] == '.I':
                self.id = int(line[3:])
            elif line[:2] == '.T':
                continue
            elif line[:2] == '.A':
                text.extend(buf)
                buf.clear()
            elif line[:2] == '.B':
                self.author = ' '.join(buf)
                buf.clear()
            elif line[:2] == '.W':
                self.ref = ' '.join(buf)
                buf.clear()
            else:
                buf.append(line)

    def postprocess_tokens(self):
        tokens = self.text
        tokens = self.split_punctuation(tokens)
        tokens = self.strip_punctuation(tokens)
        tokens = self.remove_non_alphabetic(tokens)
        tokens = self.stemmed(tokens)
        tokens = self.keep_min_length(tokens, 3)
        self.text = tokens

    def strip_punctuation(self, tokens: List[str]) -> List[str]:
        return [t.strip(string.punctuation) for t in tokens]

    def remove_non_alphabetic(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if ALPHABETS.search(t)]

    def keep_min_length(self, tokens: List[str], length) -> List[str]:
        return [t for t in tokens if len(t) >= length]

    def stemmed(self, tokens: List[str]) -> List[str]:
        stemmed = []
        for word in tokens:
            stem = STEMMER(word)
            if stem != word:
                stemmed.append(stem)
        return [*tokens, *stemmed]

    def split_punctuation(self, tokens: List[str]) -> List[str]:
        occurrences = []
        for word in self.text:
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

    def __iter__(self) -> Iterator[Document]:
        for idx, doc in self.docs.items():
            if idx in self.filter:
                yield doc

    @contextmanager
    def selection(self, *ids: int):
        """
        Show only selected documents. Within this context, using this object
        as an iterator will only yield selected documents.
        """
        self.filter = set(ids)
        try:
            yield
        finally:
            self.filter = self.docs.keys()


def parse_documents(source: Iterable[str]) -> DocumentCollection:
    docs = DocumentCollection()
    doc: Document = None
    parser: Generator = None
    for line in source:
        if line[:2] == '.I':
            if parser:
                parser.close()
            if doc:
                docs.add(doc)
            doc = Document()
            parser = doc.parse()
            parser.send(None)
        parser.send(line)
    parser.close()
    docs.add(doc)
    return docs


def evaluate(queries: DocumentCollection, documents: DocumentCollection, limit=1):
    """Calculate similarity scores for all queries and documents."""
    for idx, q in enumerate(queries, start=1):
        scores = []
        for d in documents:
            s = similarity_tf_idf(q, d, documents.idf)
            scores.append((d.id, s))
        for d, s in [*sorted(scores, key=lambda t: t[1], reverse=True)][:limit]:
            yield idx, d, s


class Clusters:
    """Co-occurrence finding using pointwise mutual information."""

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
            word = STEMMER(word)
            if word in self.targets:
                neighborhood = doc.text[i - size:i + size]
                k = i - size
                if k < 0:
                    k = 0
                for j, neighbor in enumerate(neighborhood, k):
                    neighbor = STEMMER(neighbor)
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


def process_cluster(queries: DocumentCollection, documents: DocumentCollection):
    clusters = Clusters(documents.idf)
    for doc in documents:
        clusters.add(doc)
    clusters.calc_pmi()
    clusters.map_clusters()
    for doc in queries:
        clusters.modify_query(doc)
        doc.calc_freq()


def main():
    global STEMMER
    STEMMER = init_nltk()

    db = get_db()
    CATEGORIES = ('happy', 'sad', 'relaxed', 'angry')
    playlists = {k: db.playlist_title_search(k) for k in CATEGORIES}
    songs = {k: [*chain.from_iterable([p.songs for p in ps])] for k, ps in playlists.items()}
    training = {k: v[:math.floor(len(v) * 0.8)] for k, v in songs.items()}
    development = {k: v[math.floor(len(v) * 0.8):] for k, v in songs.items()}

    print(f'Training size {sum(len(v) for v in training.values())}')
    print(f'Testing size {sum(len(v) for v in development.values())}')
    print(f'# songs { {k: len(v) for k, v in songs.items()} }')

    sources = {k: Document() for k in CATEGORIES}
    targets = {k: DocumentCollection() for k, v in development.items()}

    i = 0
    for k, d in sources.items():
        d.id = i
        d.text = []
        for s in training[k]:
            d.text.extend(nltk.tokenize.word_tokenize(s.lyrics))
        i += 1

    corpus = DocumentCollection()
    for d in sources.values():
        corpus.add(d)

    for k, ss in development.items():
        for s in ss:
            d = Document()
            d.id = s.id
            d.text = nltk.tokenize.word_tokenize(s.lyrics)
            targets[k].add(d)

    for doc in chain(corpus, *targets.values()):
        doc.postprocess_tokens()
        doc.calc_freq()
    corpus.calc_idf()
    corpus.calc_mag(corpus.idf)
    for queries in targets.values():
        process_cluster(queries, corpus)
        queries.calc_mag(corpus.idf)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for truth, queries in targets.items():
        for idx, doc_id, score in evaluate(queries, corpus):
            prediction = CATEGORIES[doc_id]
            if truth == prediction:
                tp[truth] += 1
            else:
                fp[prediction] += 1
                fn[truth] += 1

    precisions = defaultdict(int)
    recall = defaultdict(int)
    fscore = defaultdict(int)

    print('\nPRECISION')
    for c in CATEGORIES:
        try:
            p = precisions[c] = tp[c] / (tp[c] + fp[c])
        except ZeroDivisionError:
            print(f'  {c} -1')
        else:
            print(f'  {c} {p:.3f}')
    print(f'global {sum(tp.values()) / (sum(tp.values()) + sum(fp.values())):.3f}')

    print('\nRECALL')
    for c in CATEGORIES:
        try:
            r = recall[c] = tp[c] / (tp[c] + fn[c])
        except ZeroDivisionError:
            print(f'  {c} -1')
        else:
            print(f'  {c} {r:.3f}')
    print(f'global {sum(tp.values()) / (sum(tp.values()) + sum(fn.values())):.3f}')

    print('\nF-score')
    for c in CATEGORIES:
        try:
            f = fscore[c] = tp[c] / (tp[c] + .5 * (fp[c] + fn[c]))
        except ZeroDivisionError:
            print(f'  {c} -1')
        else:
            print(f'  {c} {f:.3f}')
    print(f'global {sum(tp.values()) / (sum(tp.values()) + .5 * (sum(fp.values()) + sum(fn.values()))):.3f}')
