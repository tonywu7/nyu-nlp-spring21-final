# MIT License
#
# Copyright (c) 2021 Tony Wu +https://github.com/tonywu7/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
CSCI-480.057 HW4 Ad-hoc IR.

Author: Tony Wu

Syntax
---
python3 main.py [documents file] [query file] [output file]
"""

from __future__ import annotations

import math
import re
import string
import sys
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain
from typing import (Callable, Dict, Generator, Iterable, Iterator, List, Set,
                    Tuple)

import nltk
from nltk.stem.snowball import EnglishStemmer

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


def evaluate(queries: DocumentCollection, documents: DocumentCollection):
    """Calculate similarity scores for all queries and documents."""
    for idx, q in enumerate(queries, start=1):
        scores = []
        for d in documents:
            s = similarity_tf_idf(q, d, documents.idf)
            scores.append((d.id, s))
        for d, s in sorted(scores, key=lambda t: t[1], reverse=True):
            yield idx, d, s


def main():
    with open(sys.argv[1]) as fc, open(sys.argv[2]) as fq:
        documents = parse_documents(fc)
        queries = parse_documents(fq)

    for doc in chain(documents, queries):
        doc.postprocess_tokens()
        doc.calc_freq()
    documents.calc_idf()
    documents.calc_mag(documents.idf)
    process_cluster(queries, documents)
    queries.calc_mag(documents.idf)

    if len(sys.argv[3:]) > 1:
        query_ids, doc_ids = sys.argv[3], sys.argv[4]
        if query_ids != '-':
            query_ids = [int(idx) for idx in query_ids.split(',')]
        else:
            query_ids = queries.docs.keys()
        if doc_ids != '-':
            doc_ids = [int(idx) for idx in doc_ids.split(',')]
        else:
            doc_ids = documents.docs.keys()
        with queries.selection(*query_ids), documents.selection(*doc_ids):
            for idx, doc_id, score in evaluate(queries, documents):
                print(idx, doc_id, score)
        exit()

    with open(sys.argv[3], 'w+') as output:
        for idx, doc_id, score in evaluate(queries, documents):
            if score > 0:
                output.write(f'{idx} {doc_id} {score:.17f}\n')


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


if __name__ == '__main__':
    STEMMER = init_nltk()
    main()
