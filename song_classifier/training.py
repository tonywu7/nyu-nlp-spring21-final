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

from typing import Dict, List, Set, Tuple

from .implementations.prep import Document

Songs = List
Lyrics = List[List[str]]
Titles = List[str]
Wordbag = Set[str]


def convert_songs(songs: List, postprocessors) -> Tuple[Lyrics, Wordbag, Titles]:
    lyrics = []
    titles = []
    wordbag = set()
    for song in songs:
        doc = Document(song.lyrics)
        doc.postprocess_tokens(*postprocessors)
        text = doc.text
        lyrics.append(text)
        titles.append(song.title)
        wordbag.update(text)
    return lyrics, wordbag, titles


def label_songs(songs: Dict[str, Songs], postprocessors) -> Tuple[Dict[str, Lyrics], Dict[str, Wordbag], Dict[str, Titles]]:
    lyrics = {}
    wordbags = {}
    titles = {}
    for cat, songs_in_cat in songs.items():
        lyrics[cat], wordbags[cat], titles[cat] = convert_songs(songs_in_cat, postprocessors)
    return lyrics, wordbags, titles


def sort_songs(training: Dict[str, List], testing: Dict[str, List]):
    train_labels: Dict[str, str] = {}
    for cat, songs in training.items():
        for song in songs:
            train_labels[song.title] = cat

    test_truths: Dict[str, str] = {}
    for cat, songs in testing.items():
        for song in songs:
            test_truths[song.title] = cat

    print('Dataset stats:')
    for k in training.keys():
        print(f'{k}: training={len(training[k])} testing={len(testing[k])}')

    return train_labels, test_truths
