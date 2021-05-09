# Copyright 2021 Nour Abdelmoneim, Thomas Kim, Lucas Ortiz, Tony Wu
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
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Set

import nltk

from .database import Playlist, Song, get_db
from .implementations.tfidf import (get_all_tf, get_idf, get_similarity,
                                    get_tf_idf, get_tf_idf_category)
from .implementations.tfidf2 import Document, init_nltk
from .scoring import export, score


def run():
    # Desired size of development set / total samples
    RATIO = .6

    init_nltk()

    # Load songs

    print('Loading songs')

    db = get_db()
    CATEGORIES = ('relaxed', 'sad', 'angry', 'happy')
    KEYWORDS = {
        'happy': ['happy', 'joy', 'awesome', 'party', 'city', 'love', 'sex', 'summer', 'spring', 'pop', 'yay', 'fun', 'club', 'nightlife', 'dance', 'romance', 'motivational', 'electro', 'beach', 'radio', 'beautiful', 'pretty', 'christmas', 'disco', 'birthday', 'edm', 'energetic', 'festival', 'inspirational', 'jog', 'uplifting', 'training', 'happiness'],
        'sad': ['sad', 'blues', 'breakup', 'ache', 'wish', 'die', 'alone', 'drowning', 'reminisce', 'funeral', 'dead', 'dark', 'broken', 'remember', 'forget', 'forgot', 'break', 'hope', 'lone', 'depressed', 'depression'],
        'relaxed': ['relax', 'chill', 'home', 'study', 'night', 'evening', 'high', 'weed', 'reggae', 'jazz', 'piano', 'winter', 'star', 'meditatcalm', 'soft', 'dream', 'work', 'classical', 'rap', 'hiphop', 'hip-hop', 'hip' 'hop', 'late', 'fall', 'autumn', 'sleep', 'asmr', 'country', 'indie', 'tranquil'],
        'angry': ['fuck', 'bitch', 'angry', 'mad', 'pissed', 'shit', 'rock', 'metal', 'death', 'gym', 'workout', 'hell', 'demon', 'punk', 'devil'],
    }
    playlists = {k: [*chain.from_iterable(db.playlist_title_search(t) for t in v)] for k, v in KEYWORDS.items()}
    songs_reversed: Dict[Song, Dict[str, List[Playlist]]] = defaultdict(lambda: defaultdict(list))
    for k, v in playlists.items():
        for p in v:
            for s in p.songs:
                songs_reversed[s][k].append(p)

    dev_set: Dict[str, List[Song]] = {k: [] for k in CATEGORIES}
    test_set: Dict[str, List[Song]] = {k: [] for k in CATEGORIES}

    for s, c in songs_reversed.items():
        if len(c) > 1:
            continue
        for k, ps in c.items():
            if len(ps) >= 1:
                # A song have an 80% chance to go into the development set
                # To reproduce the same sets, seed RNG
                if random.random() < RATIO:
                    dev_set[k].append(s)
                else:
                    test_set[k].append(s)

    ground_truths: Dict[str, str] = {}
    for cat, songs in test_set.items():
        for song in songs:
            ground_truths[song.title] = cat

    def dissect_data(songs: Dict[str, List[Song]]):
        lyrics: Dict[str, List[List[str]]] = {k: [] for k in CATEGORIES}
        wordbags: Dict[str, Set[str]] = {k: set() for k in CATEGORIES}
        titles: Dict[str, List[str]] = {k: [] for k in CATEGORIES}
        for cat, songs_in_cat in songs.items():
            wordbag = wordbags[cat]
            for song in songs_in_cat:
                doc = Document()
                doc.text = nltk.tokenize.word_tokenize(song.lyrics)
                doc.postprocess_tokens()
                lyrics[cat].append(doc.text)
                titles[cat].append(song.title)
                wordbag.update(doc.text)
        return lyrics, wordbags, titles

    print('Training')

    # Training
    dev_lyrics, dev_wordbags, dev_titles = dissect_data(dev_set)

    training_tf = get_all_tf(
        [*chain(*dev_lyrics.values())],
        [*chain(*dev_titles.values())],
    )
    category_idfs: Dict[str, Dict[str, float]] = {}
    for cat in CATEGORIES:
        category_idfs[cat] = get_idf(dev_wordbags[cat], dev_lyrics[cat])

    category_tf_idfs = {}
    category_vectors = {}
    for cat in CATEGORIES:
        category_tf_idfs[cat] = cat_tf_idf = get_tf_idf(
            category_idfs[cat], training_tf,
            dev_lyrics[cat], dev_titles[cat],
        )
        category_vectors[cat] = get_tf_idf_category(
            cat_tf_idf, dev_wordbags[cat],
        )

    print('Testing')

    # Testing

    test_lyrics, test_wordbags, test_titles = dissect_data(test_set)
    test_wordbags = [*chain(*test_wordbags.values())]
    test_lyrics = [*chain(*test_lyrics.values())]
    test_titles = [*chain(*test_titles.values())]

    testing_tf = get_all_tf(test_lyrics, test_titles)
    testing_idf = get_idf(test_wordbags, test_lyrics)

    testing_tf_idf = get_tf_idf(testing_idf, testing_tf, test_lyrics, test_titles)

    song_similarities = {k: None for k in test_titles}
    predictions = {k: None for k in test_titles}

    for title, vec in testing_tf_idf.items():
        song_similarities[title] = get_similarity(vec, category_vectors, CATEGORIES)
        predictions[title] = song_similarities[title][0][0]

    stats = score(predictions, ground_truths, CATEGORIES)

    print('Dataset stats:')
    for k in CATEGORIES:
        print(f'{k}: development={len(dev_set[k])} testing={len(test_set[k])}')

    print('Scores:')
    for k, (p, r, f) in stats.items():
        print(f'{k}: precision={p:.3f} recall={r:.3f} f-score={f:.3f}')

    export(predictions, ground_truths)
