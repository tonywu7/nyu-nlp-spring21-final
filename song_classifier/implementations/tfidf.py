# Copyright 2021 Nour Abdelmoneim
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
from typing import Dict, List

import pandas as pd


def get_tf(song_lyrics: List[str]) -> Dict[str, float]:
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


def get_all_tf(all_song_lyrics: List[List[str]], all_song_titles: List[str]) -> Dict[str, Dict[str, float]]:
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


def get_idf(word_list: List[str], all_song_lyrics: List[List[str]]) -> Dict[str, float]:
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

        idf_songs[word] = math.log(len(all_song_lyrics) / idf_songs[word])

    return idf_songs


def get_tf_idf(idf_songs: Dict[str, float], tf_songs: Dict[str, Dict[str, float]],
               all_song_lyrics: List[List[str]], all_song_titles: List[str]):
    tf_idf_songs: Dict[str, Dict[str, float]] = {}

    for i, song in enumerate(all_song_lyrics):
        title = all_song_titles[i]
        tf_idf_songs[title] = {word: 0 for word in song}
        for word in song:
            tf_idf_songs[title][word] = idf_songs[word] * tf_songs[title].get(word, 0)

    return tf_idf_songs


def get_tf_idf_category(tf_idf_category: Dict[str, Dict[str, float]], word_list: List[str]) -> Dict[str, float]:
    category_vector = {w: 0 for w in word_list}
    for vec in tf_idf_category.values():
        for word in vec:
            category_vector[word] += 1

    length = len(tf_idf_category)

    for word in category_vector:
        category_vector[word] = category_vector[word] / length

    return category_vector


def get_similarity(test_song_vec: Dict[str, float], category_vectors: Dict[str, Dict[str, float]], categories: List[str]):
    song_similarity = dict.fromkeys(categories)

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


def save_vectors(tfidf_vectors, labels, category):
    csv_columns = ['Title', 'Vector', 'Label']
    df = pd.DataFrame(columns=csv_columns)
    df['Title'] = tfidf_vectors.keys()
    df['Vector'] = tfidf_vectors

    # labels are categories in numerical values, so happy = 1, relaxed = 2...
    df['Label'] = labels

    filePath = f'tfidfVectors_{category}.csv'
    df.to_csv(filePath, index=False, header=True)
