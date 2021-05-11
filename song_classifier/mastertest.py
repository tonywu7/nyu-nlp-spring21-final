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

import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from .implementations.knn2 import run as run_knn
from .implementations.tfidf2 import Document
from .implementations.tfidf2 import run as run_cosine

CATEGORIES_4 = ('happy', 'sad', 'relaxed', 'angry')
KEYWORDS_4 = {
    'happy': ['happy', 'joy', 'awesome', 'party', 'city', 'love', 'sex', 'summer', 'spring', 'pop', 'yay', 'fun', 'club', 'nightlife', 'dance', 'romance', 'motivational', 'electro', 'beach', 'radio', 'beautiful', 'pretty', 'christmas', 'disco', 'birthday', 'edm', 'energetic', 'festival', 'inspirational', 'jog', 'uplifting', 'training', 'happiness'],
    'sad': ['sad', 'blues', 'breakup', 'ache', 'wish', 'die', 'alone', 'drowning', 'reminisce', 'funeral', 'dead', 'dark', 'broken', 'remember', 'forget', 'forgot', 'break', 'hope', 'lone', 'depressed', 'depression'],
    'relaxed': ['relax', 'chill', 'home', 'study', 'night', 'evening', 'high', 'weed', 'reggae', 'jazz', 'piano', 'winter', 'star', 'meditatcalm', 'soft', 'dream', 'work', 'classical', 'rap', 'hiphop', 'hip-hop', 'hip' 'hop', 'late', 'fall', 'autumn', 'sleep', 'asmr', 'country', 'indie', 'tranquil'],
    'angry': ['fuck', 'bitch', 'angry', 'mad', 'pissed', 'shit', 'rock', 'metal', 'death', 'gym', 'workout', 'hell', 'demon', 'punk', 'devil'],
}

CATEGORIES_2 = ('positive', 'negative')
KEYWORDS_2 = {
    'positive': ['happy', 'joy', 'awesome', 'party', 'city', 'love', 'sex', 'summer', 'spring', 'pop', 'yay', 'fun', 'club', 'nightlife', 'dance', 'romance', 'motivational', 'electro', 'beach', 'radio', 'beautiful', 'pretty', 'christmas', 'disco', 'birthday', 'edm', 'energetic', 'festival', 'inspirational', 'jog', 'uplifting', 'training', 'happiness', 'relax', 'chill', 'home', 'study', 'night', 'evening', 'high', 'weed', 'reggae', 'jazz', 'piano', 'winter', 'star', 'meditatcalm', 'soft', 'dream', 'work', 'classical', 'rap', 'hiphop', 'hip-hop', 'hip' 'hop', 'late', 'fall', 'autumn', 'sleep', 'asmr', 'country', 'indie', 'tranquil'],
    'negative': ['sad', 'blues', 'breakup', 'ache', 'wish', 'die', 'alone', 'drowning', 'reminisce', 'funeral', 'dead', 'dark', 'broken', 'remember', 'forget', 'forgot', 'break', 'hope', 'lone', 'depressed', 'depression', 'fuck', 'bitch', 'angry', 'mad', 'pissed', 'shit', 'rock', 'metal', 'death', 'gym', 'workout', 'hell', 'demon', 'punk', 'devil'],
}


def run_test(name, function, parameters, seed=None):
    import sys
    out = open(f'{name}.txt', 'w+')
    sys.stdout = out
    random.seed(seed)
    from .app import Application
    Application('project')
    function(*parameters)


def params_1():
    algorithms = {'cosine': run_cosine}
    categories = {'cat4': CATEGORIES_4}
    keywords = {'playlists4': KEYWORDS_4}
    postprocessors = {
        'lexical,syntactic': [
            Document.remove_non_alphabetic,
            Document.split_punctuation,
            Document.strip_punctuation,
            Document.to_lower,
            Document.filter_stop_words,
            Document.lemmatized,
            Document.filter_by_pos,
        ],
        'lexical': [
            Document.split_punctuation,
            Document.strip_punctuation,
            Document.remove_non_alphabetic,
            Document.to_lower,
            Document.filter_stop_words,
            Document.lemmatized,
        ],
        'characters': [
            Document.split_punctuation,
            Document.strip_punctuation,
            Document.remove_non_alphabetic,
            Document.to_lower,
        ],
    }
    min_weight = {str(i): i for i in (3,)}
    knn_n_neighbors = {str(i): i for i in (0,)}
    return algorithms, categories, keywords, postprocessors, min_weight, knn_n_neighbors


def params_2():
    algorithms = {'knn': run_knn}
    categories = {'cat4': CATEGORIES_4}
    keywords = {'playlists4': KEYWORDS_4}
    postprocessors = {
        'lexical,syntactic': [
            Document.remove_non_alphabetic,
            Document.split_punctuation,
            Document.strip_punctuation,
            Document.to_lower,
            Document.filter_stop_words,
            Document.lemmatized,
            Document.filter_by_pos,
        ],
    }
    min_weight = {str(i): i for i in (2,)}
    knn_n_neighbors = {str(i): i for i in (5,)}
    return algorithms, categories, keywords, postprocessors, min_weight, knn_n_neighbors


def test(seed: int):
    algorithms, categories, keywords, postprocessors, min_weight, knn_n_neighbors = params_1()

    jobs = []

    for aname, a in algorithms.items():
        for (cname, c), (kname, k) in zip(categories.items(), keywords.items()):
            for pname, p in postprocessors.items():
                for mname, m in min_weight.items():
                    if aname == 'cosine':
                        name = f'algo={aname};cat={cname};post={pname};threshold={mname}'
                        function = a
                        parameters = (.8, c, k, p, m)
                        jobs.append((run_test, name, function, parameters, seed))
                    else:
                        for nname, n in knn_n_neighbors.items():
                            name = f'algo={aname};cat={cname};post={pname};threshold={mname};neighbor={nname}'
                            function = a
                            parameters = (.8, c, k, p, m, n)
                            jobs.append((run_test, name, function, parameters, seed))

    with ProcessPoolExecutor(8) as executor:
        futures = {}
        for f, name, *args in jobs:
            futures[executor.submit(f, name, *args)] = name

        for fut in as_completed(futures):
            fut.result()
            print(futures[fut])
