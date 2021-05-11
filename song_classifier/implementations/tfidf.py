# Copyright 2021 Nour Abdelmoneim, Tony Wu
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
from itertools import chain
from typing import Dict, List, Tuple

from ..collector import samples
from ..scoring import print_score, score, stats
from ..settings import CATEGORIES, KEYWORDS, TESTING_RATIO
from ..training import Lyrics, Titles, Wordbag, label_songs, sort_songs
from .prep import init_nltk

Vector = Dict[str, float]


def get_tf(song_lyrics: List[str]) -> Vector:
    """Get TF value for one song.

    Parameters
    ----------
    song_lyrics : List[str]
        A list of words in a song's lyrics

    Returns
    -------
    List[float]
        Term frequencies for each term
    """
    # creating a dict where the keys are each of the words in the song
    tf_song = {w: 0 for w in song_lyrics}

    # counting the frequency of each word in the song
    for word in song_lyrics:
        tf_song[word] += 1

    song_length = len(song_lyrics)

    # dividing the frequency of each word by the length of the song
    return {word: term_freq / song_length for word, term_freq in tf_song.items()}


def get_all_tf(all_song_lyrics: Lyrics, all_song_titles: Titles) -> Dict[str, Vector]:
    """Get term frequencies for all songs.

    Parameters
    ----------
    all_song_lyrics : List[List[str]]
        A list of len(songs) containing the lyrics of each song
    all_song_titles : List[str]
        A list of song titles

    Returns
    -------
    Dict[str, List[float]]
        A dictionary of term frequency vectors mapped to song titles
    """
    tf_songs = {}
    for i, song_lyrics in enumerate(all_song_lyrics):
        song_title = all_song_titles[i]

        tf_songs[song_title] = get_tf(song_lyrics)

    return tf_songs


def get_idf(word_list: Wordbag, all_song_lyrics: Lyrics) -> Vector:
    """Get idf for a song.

    Parameters
    ----------
    word_list : List[str]
        A list that contains all the words in all the lyrics
    all_song_lyrics : List[List[str]]
        A list of len(songs) containing the lyrics of each song

    Returns
    -------
    Dict[str, float]
        idf table
    """
    idf_songs = {w: 0 for w in word_list}

    # or you will get horrifying amortized O(n) complexity.
    all_song_lyrics_sets = [set(v) for v in all_song_lyrics]

    for word in word_list:
        for song in all_song_lyrics_sets:
            if word in song:
                idf_songs[word] += 1

        idf_songs[word] = idf_songs[word] and math.log(len(all_song_lyrics) / idf_songs[word])

    return idf_songs


def get_tf_idf(idf_songs: Vector, tf_songs: Dict[str, Vector],
               all_song_lyrics: Lyrics, all_song_titles: Titles) -> Dict[str, Vector]:
    tf_idf_songs: Dict[str, Vector] = {}

    for i, song in enumerate(all_song_lyrics):
        title = all_song_titles[i]
        tf_idf_songs[title] = {word: 0 for word in song}
        for word in song:
            tf_idf_songs[title][word] = idf_songs[word] * tf_songs[title].get(word, 0)

    return tf_idf_songs


def get_tf_idf_category(tf_idf_category: Dict[str, Vector], word_list: List[str]) -> Vector:
    category_vector = {w: 0 for w in word_list}
    for vec in tf_idf_category.values():
        for word in vec:
            category_vector[word] += 1

    length = len(tf_idf_category)

    for word in category_vector:
        category_vector[word] = category_vector[word] / length

    return category_vector


def get_similarity(test_song_vec: Vector, category_vectors: Dict[str, Vector]) -> List[Tuple[str, float]]:
    song_similarity = {k: None for k in category_vectors}

    for category_name, vec in category_vectors.items():
        cat_vec = []
        for word in test_song_vec:
            if word in vec:
                cat_vec.append(vec[word])
            else:
                cat_vec.append(0.0)

        numerator = 0
        cat_denominator = 0
        song_denominator = 0

        for i, word in enumerate(test_song_vec):
            numerator += (cat_vec[i] * test_song_vec[word])
            cat_denominator += (cat_vec[i] ** 2)
            song_denominator += (test_song_vec[word] ** 2)

        denominator = math.sqrt(cat_denominator * song_denominator)

        if denominator != 0:
            song_similarity[category_name] = numerator / denominator
        else:
            song_similarity[category_name] = 0

    song_similarity = sorted(song_similarity.items(), key=lambda t: t[1], reverse=True)

    return song_similarity


def master_tf_idf(lyrics: Lyrics, titles: Titles, wordbag: Wordbag):
    tf = get_all_tf(lyrics, titles)
    idf = get_idf(wordbag, lyrics)
    tf_idf = get_tf_idf(idf, tf, lyrics, titles)
    return tf_idf


def run():
    init_nltk()

    # Load songs
    print('Loading songs')

    training, testing = samples(TESTING_RATIO, CATEGORIES, KEYWORDS)
    categories = CATEGORIES
    _, ground_truths = sort_songs(training, testing)

    train_lyrics, train_wordbags, train_titles = label_songs(training)
    test_lyrics, test_wordbags, test_titles = label_songs(testing)

    print('Training')

    training_tf = get_all_tf(
        [*chain(*train_lyrics.values())],
        [*chain(*train_titles.values())],
    )
    category_idfs: Dict[str, Dict[str, float]] = {}
    for cat in categories:
        category_idfs[cat] = get_idf(train_wordbags[cat], train_lyrics[cat])

    category_tf_idfs = {}
    category_vectors = {}
    for cat in categories:
        category_tf_idfs[cat] = cat_tf_idf = get_tf_idf(
            category_idfs[cat], training_tf,
            train_lyrics[cat], train_titles[cat],
        )
        category_vectors[cat] = get_tf_idf_category(
            cat_tf_idf, train_wordbags[cat],
        )

    from ..reflection import vector_distances
    print(vector_distances(category_vectors))

    print('Testing')

    test_wordbag_all = {*chain(*test_wordbags.values())}
    test_lyrics = [*chain(*test_lyrics.values())]
    test_titles = [*chain(*test_titles.values())]

    testing_tf_idf = master_tf_idf(test_lyrics, test_titles, test_wordbag_all)

    song_similarities = {k: None for k in test_titles}
    predictions = {k: None for k in test_titles}

    for title, vec in testing_tf_idf.items():
        song_similarities[title] = get_similarity(vec, category_vectors)
        pred, sim = song_similarities[title][0]
        predictions[title] = pred
        # print(title, pred, sim)

    stats(predictions, ground_truths, categories)
    scores = score(predictions, ground_truths, categories)
    print_score(*scores)
    # export(predictions, ground_truths)
