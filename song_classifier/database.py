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

from __future__ import annotations

import uuid
from itertools import chain
from typing import Dict, List

from sqlalchemy import Column, event, types

from .util.database import (FTS5, BundleABC, Identity, Relationship, UUIDType,
                            get_session, metadata)

VERSION = '0.0.1'


class Database(BundleABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global db
        db = self
        self.fts: FTS5

    @property
    def version(self):
        return VERSION

    def _init_events(self):
        super()._init_events()
        event.listen(metadata, 'after_create', self._init_fts)

    def _init_fts(self, *args, **kwargs):
        self.fts = FTS5()
        self.fts.init(get_session)

    def delete_orphans(self):
        songs = (
            self.query(Song).join(Song.playlists, isouter=True)
            .filter(Playlist.id.is_(None)).all()
        )
        playlists = (
            self.query(Playlist).join(Playlist.songs, isouter=True)
            .filter(Song.id.is_(None)).all()
        )
        for item in chain(songs, playlists):
            db.session.delete(item)

    def get_song_by_title(self, title: str) -> Song:
        return self.query(Song).filter(Song.title == title).first()

    def get_songs_by_title(self, title: str) -> List[Song]:
        return self.query(Song).filter(Song.title == title).all()

    def get_playlist_by_title(self, title: str) -> Playlist:
        return self.query(Playlist).filter(Playlist.title == title).first()

    def get_playlists_by_title(self, title: str) -> List[Playlist]:
        return self.query(Playlist).filter(Playlist.title == title).all()

    def get_all_songs(self) -> List[Song]:
        return self.query(Song).all()

    def get_all_playlists(self) -> List[Playlist]:
        return self.query(Playlist).all()

    def song_title_search(self, query: str) -> List[Song]:
        return self.fts.query(f'identity_model:Song AND song_title:{self.fts.tokenized(query)}').all()

    def playlist_title_search(self, query: str) -> List[Song]:
        return self.fts.query(f'identity_model:Playlist AND playlist_title:{self.fts.tokenized(query)}').all()

    def song_lyrics_search(self, query: str) -> List[Song]:
        return self.fts.query(f'identity_model:Song AND song_lyrics:{self.fts.tokenized(query)}').all()


def get_db() -> Database:
    return db


class Song(Identity):
    title: str = Column(types.String(), nullable=False)
    artist: str = Column(types.String())
    album: str = Column(types.String())

    lyrics: str = Column(types.String())
    stats: Dict = Column(types.JSON())

    playlists: List[Playlist] = Relationship.two_way(dict(playlists='Playlist', songs='Song'))['playlists']

    id_musicbrainz: uuid.UUID = Column(UUIDType())
    id_genius: int = Column(types.Integer())


class Playlist(Identity):
    title: str = Column(types.String(), nullable=False)

    songs = Relationship.two_way(dict(playlists='Playlist', songs='Song'))['songs']


class AcousticBrainzFeatures:
    pass
