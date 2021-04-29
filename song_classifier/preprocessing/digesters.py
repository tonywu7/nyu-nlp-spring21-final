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

import csv
import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Generator

import langdetect
import simplejson as json

from ..app import get_settings
from ..util.logger import colored as _

log = logging.getLogger('digester')
multiprocessing.set_start_method('forkserver', True)


def csv_dir_reader(in_dir: Path):
    in_dir = Path(in_dir)
    for fp in in_dir.iterdir():
        if fp.suffix != '.csv':
            continue
        with open(fp, 'r') as f:
            log.info(f'Opening {fp}')
            reader = csv.DictReader(f, restkey='extra')
            for row in reader:
                yield row


def digest1(in_dir: str):
    in_dir = Path(in_dir)
    for row in csv_dir_reader(in_dir):
        song = {
            'title': row['SONG_NAME'],
            'artist': row['ARTIST_NAME'],
            'lyrics': ' '.join([row['LYRICS'] or '', *row.get('extra', ())]),
        }
        yield song


def digest2(in_dir: str):
    in_dir = Path(in_dir)
    for row in csv_dir_reader(in_dir):
        song = {
            'title': row['track_name'],
            'artist': row['track_artist'],
            'album': row['track_album_name'],
            'playlist': row['playlist_name'],
            'lyrics': row['lyrics'],
            'playlist_id': row['playlist_id'],
            'stats': {k: row.get(k) for k in (
                'danceability', 'energy', 'key', 'loudness',
                'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence',
                'tempo', 'duration_ms',
            )},
        }
        yield song


def digest3(in_dir: str):
    in_dir = Path(in_dir)
    for row in csv_dir_reader(in_dir):
        playlist_name = ' '.join([row['playlistname'] or '', *row.get('extra', ())])
        song = {
            'title': row['trackname'],
            'artist': row['artistname'],
            'playlist': playlist_name,
            'playlist_id': f'{row["user_id"]}/{playlist_name}',
        }
        yield song


digesters = dict({str(idx): f for idx, f in enumerate([digest1, digest2, digest3], start=1)})


def _lang_detect(text: str):
    try:
        return langdetect.detect_langs(text)
    except Exception:
        return None


def driver(f: Callable[[str, str], Generator[Dict, None, None]], *args, output, **kwargs):
    lang_threshold: Dict[str, float] = get_settings()['lang_threshold']
    output = Path(output)
    selected = []
    discarded = []
    failure = []
    with ProcessPoolExecutor(os.cpu_count() - 1) as executor:
        futures = {}
        for song in f(*args, **kwargs):
            if 'lyrics' in song:
                futures[executor.submit(_lang_detect, song['lyrics'])] = song
            else:
                selected.append(song)
        for future in as_completed(futures):
            song = futures[future]
            langs = future.result()
            if not langs:
                log.error(f'{song["title"]} - {song["artist"]}: Failed to detect language')
                failure.append(song)
            elif any((ll.prob >= lang_threshold.get(ll.lang, math.inf) for ll in langs)):
                color = 'green'
                selected.append(song)
            else:
                color = 'red'
                discarded.append(song)
            log.info(_(f'lang={langs} {song["title"]} - {song["artist"]}', color=color))
    log.info(f'{len(selected)} selected, {len(discarded)} discarded, {len(failure)} errors')
    with open(output, 'w+') as f:
        json.dump(selected, f)
    with open(output.with_name(f'{output.name}.failure.json'), 'w+') as f:
        json.dump(failure, f)
    with open(output.with_name(f'{output.name}.discarded.json'), 'w+') as f:
        json.dump(discarded, f)
