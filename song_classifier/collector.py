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
from typing import Dict, List, Tuple

from .database import Playlist, Song, get_db

Songs = List[Song]
LabeledSongs = Dict[str, Songs]
GroundTruths = Dict[str, str]


def samples(RNG: random.Random, sample_ratio: float, categories: List[str],
            keywords: Dict[str, List[str]], min_weight: int) -> Tuple[LabeledSongs, LabeledSongs]:
    """Gather samples from database.

    Parameters
    ----------
    sample_ratio : float
        The ratio of samples to go into the training set vs. test set.
    categories : List[str]
        Category labels
    keywords : Dict[str, List[str]]
        Keywords for each categories to be used to retrieve playlists
    min_weight : int
        Minimum number of playlists a song to appear in for it to enter the samples

    Returns
    -------
    Tuple[LabeledSongs, LabeledSongs]
        Training/testing set, dictionaries of lists of songs mapped to labels
    """
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
                if RNG.random() < sample_ratio:
                    training[k].append(s)
                else:
                    testing[k].append(s)

    return training, testing
