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

import logging
import uuid
from collections import defaultdict
from typing import DefaultDict, Dict, Set, Tuple

import nltk
import simplejson as json
from more_itertools import first

from ..database import Database, Identity, Playlist, Relationship, Song

Signature = Tuple[str, ...]


class RawDataConsumer:
    def __init__(self):
        self.log = logging.getLogger('consumer')
        self.songs: DefaultDict[Signature, Set[str]] = defaultdict(set)
        self.playlists: DefaultDict[Signature, Set[str]] = defaultdict(set)

        self.song_playlists: DefaultDict[Signature, Set[Signature]] = defaultdict(set)
        self.playlist_songs: DefaultDict[Signature, Set[Signature]] = defaultdict(set)

        self.entities: Set[Signature] = set()
        self._buffer: Dict[str, Signature] = {}

    @staticmethod
    def signature_nocache(*names: str) -> Signature:
        keywords = []
        for name in names:
            tokens = [t.lower() for t in nltk.tokenize.word_tokenize(name) if t.isalnum()]
            keywords.extend(tokens)
        return tuple(keywords)

    def signature(self, *names: str) -> Signature:
        keywords = []
        for name in names:
            try:
                keywords.extend(self._buffer[name])
            except KeyError:
                tokens = [t.lower() for t in nltk.tokenize.word_tokenize(name) if t.isalnum()]
                keywords.extend(tokens)
                self._buffer[name] = tokens
        return tuple(keywords)

    def append(self, data: Dict[str, str]):
        song_id = self.signature(data['title'], data['artist'])

        if 'lyrics' in data:
            self.songs[song_id].add(json.dumps(data, sort_keys=True))

        playlist = data.get('playlist')
        if playlist:
            playlist_id = self.signature(playlist, data['playlist_id'])
            self.playlists[playlist_id].add(playlist)
            self.song_playlists[song_id].add(playlist_id)
            self.playlist_songs[playlist_id].add(song_id)

    def screen(self):
        for k, v in self.playlists.items():
            if len(v) == 1:
                self.accept(k)

        lyrics: DefaultDict[Signature, Set[Signature]] = defaultdict(set)

        for k, v in self.songs.items():
            candidates = [json.loads(s) for s in v]
            for c in candidates:
                lyrics[self.signature(c['lyrics'])].add(k)

        for v in lyrics.values():
            for i in v:
                for j in v:
                    self.song_playlists[i] |= self.song_playlists[j]
            for k in self.song_playlists[i]:
                self.playlist_songs[k] |= v
            self.accept(first(v))

    def accept(self, sig: Signature):
        self.entities.add(sig)

    def export(self, db: Database):
        idx = 1
        entities = {}
        identities = []
        songs = []
        playlists = []
        pairings = []
        for sig in self.entities:
            if sig in self.playlists:
                identities.append({
                    'id': idx,
                    'uuid4': uuid.uuid4(),
                    'model': 'Playlist',
                })
                playlists.append({
                    'id': idx,
                    'title': first(self.playlists[sig]),
                })
                entities[sig] = idx
                idx += 1
            elif sig in self.songs:
                candidate = first([json.loads(s) for s in self.songs[sig]])
                tmpl = {
                    'title': None,
                    'artist': None,
                    'album': None,
                    'lyrics': None,
                    'stats': None,
                    'id_musicbrainz': None,
                    'id_genius': None,
                }
                identities.append({
                    'id': idx,
                    'uuid4': uuid.uuid4(),
                    'model': 'Song',
                })
                songs.append({'id': idx, **tmpl, **candidate})
                entities[sig] = idx
                idx += 1
        for k, v in self.song_playlists.items():
            for p in v:
                ida = entities.get(k)
                idb = entities.get(p)
                if ida and idb:
                    pairings.append({'src': idb, 'dst': ida})
        db.execute(Identity.__table__.insert(), identities)
        db.execute(Song.__table__.insert(), songs)
        db.execute(Playlist.__table__.insert(), playlists)
        db.execute(Relationship.cached['_g_playlists_songs'].insert(), pairings)


def append(*data_paths: str, consumer=None):
    consumer = consumer or RawDataConsumer()
    for path in data_paths:
        consumer.log.info(f'Opening {path}')
        with open(path) as f:
            for data in json.load(f):
                consumer.append(data)
    return consumer
