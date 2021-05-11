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

import re
import string
from typing import List

import nltk
from nltk.stem import WordNetLemmatizer


def init_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)


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

ALPHABETS = re.compile(r'[A-Za-z]')
SPLITTING = re.compile(r'\W+')


class Document:
    lemmatize = WordNetLemmatizer().lemmatize

    def __init__(self, text: str):
        self.text: List[str] = nltk.tokenize.word_tokenize(text)
        self.postprocess_tokens()

    def postprocess_tokens(self):
        tokens = self.text
        tokens = [t.lower() for t in tokens]
        tokens = [t for t in tokens if t not in STOP_WORDS]
        tokens = self.split_punctuation(tokens)
        tokens = self.strip_punctuation(tokens)
        tokens = self.remove_non_alphabetic(tokens)
        tokens = self.lemmatized(tokens)
        tokens = self.keep_min_length(tokens, 3)
        self.text = tokens

    def strip_punctuation(self, tokens: List[str]) -> List[str]:
        return [t.strip(string.punctuation) for t in tokens]

    def remove_non_alphabetic(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if ALPHABETS.search(t)]

    def keep_min_length(self, tokens: List[str], length) -> List[str]:
        return [t for t in tokens if len(t) >= length]

    def lemmatized(self, tokens: List[str]) -> List[str]:
        lemmatized = []
        for word in tokens:
            lemma = self.lemmatize(word)
            if lemma != word:
                lemmatized.append(lemma)
        return [*tokens, *lemmatized]

    def split_punctuation(self, tokens: List[str]) -> List[str]:
        occurrences = []
        for word in self.text:
            parts = SPLITTING.split(word)
            if len(parts) > 1:
                occurrences.extend(parts)
        return [*tokens, *occurrences]
