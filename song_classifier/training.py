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
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Set, Tuple

from .database import Playlist, Song, get_db
from .implementations.prep import Document

Songs = List[Song]
Lyrics = List[List[str]]
Titles = List[str]
Wordbag = Set[str]

LabeledSongs = Dict[str, Songs]
GroundTruths = Dict[str, str]


def convert_songs(songs: List[Song]) -> Tuple[Lyrics, Wordbag, Titles]:
    lyrics = []
    titles = []
    wordbag = set()
    for song in songs:
        doc = Document(song.lyrics)
        lyrics.append(doc.text)
        titles.append(song.title)
        wordbag.update(doc.text)
    return lyrics, wordbag, titles


def label_songs(songs: Dict[str, Songs]) -> Tuple[Dict[str, Lyrics], Dict[str, Wordbag], Dict[str, Titles]]:
    lyrics = {}
    wordbags = {}
    titles = {}
    for cat, songs_in_cat in songs.items():
        lyrics[cat], wordbags[cat], titles[cat] = convert_songs(songs_in_cat)
    return lyrics, wordbags, titles


def samples(sample_ratio: float, categories: List[str], keywords: Dict[str, List[str]], min_weight=2) -> Tuple[LabeledSongs, LabeledSongs, GroundTruths]:
    db = get_db()
    playlists = {k: [*chain.from_iterable(db.playlist_title_search(t) for t in v)] for k, v in keywords.items()}
    songs_reversed: Dict[Song, Dict[str, List[Playlist]]] = defaultdict(lambda: defaultdict(list))
    for k, v in playlists.items():
        for p in v:
            for s in p.songs:
                songs_reversed[s][k].append(p)

    training: Dict[str, List[Song]] = {k: [] for k in categories}
    testing: Dict[str, List[Song]] = {k: [] for k in categories}

    for s, c in songs_reversed.items():
        if len(c) > 1:
            continue
        for k, ps in c.items():
            if len(ps) >= min_weight:
                # A song have a certain chance to go into the development set
                # To reproduce the same sets, seed RNG
                if random.random() < sample_ratio:
                    training[k].append(s)
                else:
                    testing[k].append(s)

    train_labels: Dict[str, str] = {}
    for cat, songs in training.items():
        for song in songs:
            train_labels[song.title] = cat

    test_truths: Dict[str, str] = {}
    for cat, songs in testing.items():
        for song in songs:
            test_truths[song.title] = cat

    print('Dataset stats:')
    for k in categories:
        print(f'{k}: training={len(training[k])} testing={len(testing[k])}')

    return training, testing, train_labels, test_truths
