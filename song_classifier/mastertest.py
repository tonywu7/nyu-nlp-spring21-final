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

from .implementations.knn import run as run_knn
from .implementations.prep import Document
from .implementations.tfidf import run as run_cosine

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


def run_test(seed, name, function, parameters):
    import sys
    out = open(f'{name}.txt', 'w+')
    sys.stdout = out
    random.seed(seed)
    from .app import Application
    Application('project')
    function(*parameters)


def test(seed: int):
    algorithms = {'knn': run_knn, 'cosine': run_cosine}
    categories = {'cat4': CATEGORIES_4, 'cat2': CATEGORIES_2}
    keywords = {'playlists4': KEYWORDS_4, 'playlist2': KEYWORDS_2}
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
        'characters': [
            Document.split_punctuation,
            Document.strip_punctuation,
            Document.remove_non_alphabetic,
            Document.to_lower,
        ],
        'lexical': [
            Document.split_punctuation,
            Document.strip_punctuation,
            Document.remove_non_alphabetic,
            Document.to_lower,
            Document.filter_stop_words,
            Document.lemmatized,
        ],
    }
    min_weight = {str(i): i for i in (2, 1)}
    knn_n_neighbors = {str(i): i for i in (10, 7)}

    jobs = []

    for aname, a in algorithms.items():
        for (cname, c), (kname, k) in zip(categories.items(), keywords.items()):
            for pname, p in postprocessors.items():
                for mname, m in min_weight.items():
                    if aname == 'cosine':
                        name = f'algo={aname};cat={cname};post={pname};threshold={mname}'
                        function = a
                        parameters = (.8, c, k, p, m)
                        jobs.append((run_test, seed, name, function, parameters))
                    else:
                        for nname, n in knn_n_neighbors.items():
                            name = f'algo={aname};cat={cname};post={pname};threshold={mname};neighbor={nname}'
                            function = a
                            parameters = (.8, c, k, p, m, n)
                            jobs.append((run_test, seed, name, function, parameters))

    with ProcessPoolExecutor(8) as executor:
        futures = {}
        for f, seed, name, *args in jobs:
            futures[executor.submit(f, seed, name, *args)] = name

        for fut in as_completed(futures):
            fut.result()
            print(futures[fut])
