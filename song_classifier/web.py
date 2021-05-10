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

import asyncio
import logging
import os
import random
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiofiles
import aiohttp
import simplejson as json
from bs4 import BeautifulSoup
from decouple import config
from more_itertools import chunked

from .util.logger import colored


def normalize_url(url: str) -> str:
    u = urlsplit(url)
    q = urlencode(sorted(parse_qsl(u.query)))
    return urlunsplit((*u[:3], q, u.fragment))


def cacheable(f: Callable[[], aiohttp.ClientResponse]):
    @wraps(f)
    async def wrapped(*args, url: str, **kwargs) -> aiohttp.ClientResponse:
        url = normalize_url(url)
        return await f(*args, url=url, **kwargs)
    return wrapped


class APIClient:
    USER_AGENT: str
    API_ORIGIN: Tuple[str, str]
    API_ROOT: str

    def __init__(self, sem: int, headers: Dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_http = logging.getLogger('api.http')
        self._semaphore = asyncio.BoundedSemaphore(sem)
        self._ratelimit = asyncio.Event()
        self._ratelimit.set()
        self._client: aiohttp.ClientSession
        self._extra_headers = headers or {}

    @property
    def _session(self) -> aiohttp.ClientSession:
        try:
            return self._client
        except AttributeError:
            self._client = aiohttp.ClientSession(
                headers={'User-Agent': self.USER_AGENT, **self._extra_headers},
            )
            return self._client

    def _build_url(self, endpoint, query: Dict):
        return urlunsplit((*self.API_ORIGIN, f'{self.API_ROOT}{endpoint}', urlencode(query), ''))

    async def close(self):
        await self._session.close()

    async def _wait(self, sec: float):
        if not self._ratelimit.is_set():
            return
        if not sec:
            return
        self._log_http.warning(f'Rate limit hit, pausing for {sec} seconds')
        self._ratelimit.clear()
        await asyncio.sleep(sec)
        self._ratelimit.set()

    async def _throttle(self, res: aiohttp.ClientResponse):
        if res.status == 429:
            await self._wait(20)
            return False
        retry_after = res.headers.get('retry-after')
        if retry_after:
            await self._wait(int(retry_after) + 1)
            return False
        rate_remaining = res.headers.get('x-ratelimit-remaining')
        rate_reset = res.headers.get('x-ratelimit-reset')
        if not rate_remaining or not rate_reset:
            return True
        rate_remaining = int(rate_remaining)
        rate_reset = int(rate_reset)
        rate_reset_in = rate_reset - time.time()
        if rate_remaining > 5 or rate_reset_in <= 0:
            return True
        await self._wait(rate_reset_in + 1)
        return True

    async def _request(self, *, url: str) -> aiohttp.ClientResponse:
        while True:
            await asyncio.sleep(.25 + 5 * random.random())
            async with self._semaphore:
                await self._ratelimit.wait()
                self._log_http.info(f'GET {url}')
                try:
                    res = await self._session.get(url)
                except Exception:
                    self._log_http.warning(f'Retrying {url}')
                    continue
                if await self._throttle(res):
                    return res
                self._log_http.warning(f'Retrying {url}')


class JSONExporter:
    def __init__(self, export: str, *args, **kwargs):
        self._maxfiles = asyncio.BoundedSemaphore(48)
        self._output = Path(export).resolve()
        self._written: Set[Tuple[str, str]] = set()
        self._init_dirs()

    def _init_dirs(self):
        raise NotImplementedError

    def _saved(self, prefix, identity):
        p: Path = self._output / prefix / f'{identity}.json'
        return p.exists()

    async def _export(self, prefix: str, identity: str, data: Dict):
        if self._saved(prefix, identity):
            return False
        async with self._maxfiles, aiofiles.open(self._output / prefix / f'{identity}.json', 'w+') as f:
            await f.write(json.dumps(data))
            self._written.add((prefix, identity))
            return True


class MBZ(APIClient, JSONExporter):
    API_ORIGIN = ('https', 'musicbrainz.org')
    API_ROOT = '/ws/2/'
    USER_AGENT = (
        'NYUNLPSpring21MusicSentiment/0.0.1 '
        '( https://github.com/tonywu7/nyu-nlp-spring21-final )'
    )

    def __init__(self, sem: int, export: str):
        super().__init__(sem=sem, export=export)
        self._log = logging.getLogger('mbz')

        self._artists: Dict[str, Dict] = {}
        self._tracks: Dict[str, Dict] = {}
        self._authorship: Dict[str, str] = {}

    def _init_dirs(self):
        os.makedirs(self._output / 'artist', exist_ok=True)
        os.makedirs(self._output / 'recording', exist_ok=True)

    def _url_search(self, entity_t: str, query: str, **kwargs) -> str:
        return self._build_url(entity_t, {'query': query, 'fmt': 'json', **kwargs})

    def _url_browse(self, entity_id: str, src_t: str, dst_t: str, **kwargs) -> str:
        return self._build_url(dst_t, {src_t: entity_id, 'fmt': 'json', **kwargs})

    async def search_artist(self, name: str) -> str:
        endpoint = self._url_search('artist', name, limit=1)
        res = await self._request(url=endpoint)
        data = await res.json()
        results = data.get('artists')
        if not results:
            self._log.info(f'no result for {name}')
            return
        artist = results[0]
        uuid = artist['id']
        self._log.info(f'artist {uuid} {name}')
        self._artists[uuid] = artist
        await self._export('artist', uuid, artist)
        return uuid

    async def get_recordings_by_artist(self, artist_id: str, status='official'):
        offset = 0
        limit = 100
        recordings = set()
        while True:
            endpoint = self._url_browse(
                artist_id, 'artist', 'release', status=status, inc='recordings',
                offset=offset, limit=limit,
            )
            res = await self._request(url=endpoint)
            data = await res.json()
            results = data.get('releases')
            if not results:
                break
            for r in results:
                for m in r['media']:
                    for t in m['tracks']:
                        rec = t.get('recording')
                        if not rec:
                            continue
                        if rec['id'] in recordings:
                            continue
                        recordings.add(rec['id'])
                        rec_id: str = rec['id']
                        self._authorship[artist_id] = rec_id
                        exported = await self._export('recording', rec_id, rec)
                        if not exported:
                            continue
                        title: str = rec['title']
                        yield rec_id, title
            offset += limit


class ABZ(APIClient, JSONExporter):
    API_ORIGIN = ('https', 'acousticbrainz.org')
    API_ROOT = '/api/v1/'
    USER_AGENT = (
        'NYUNLPSpring21MusicSentiment/0.0.1 '
        '( https://github.com/tonywu7/nyu-nlp-spring21-final )'
    )

    API_MAX_REQUEST = 25

    def __init__(self, sem: int, export: str):
        super().__init__(sem=sem, export=export)
        self._features: Dict[str, Dict] = {}
        self._log = logging.getLogger('acousticbrainz')

    def _init_dirs(self):
        os.makedirs(self._output / 'feature', exist_ok=True)

    async def get_features_by_ids(self, *rec_ids: str):
        assert len(rec_ids) <= self.API_MAX_REQUEST
        endpoint = self._build_url('high-level', {'recording_ids': ';'.join(rec_ids)})
        res = await self._request(url=endpoint)
        data = await res.json()
        has_feature = []
        for k, v in data.items():
            try:
                uuid.UUID(k)
            except Exception:
                continue
            features = v['0']
            self._features[k] = features
            self._log.info(f'feature {k}')
            await self._export('feature', k, features)
            has_feature.append(k)
        return has_feature


class Genius(APIClient, JSONExporter):
    API_ORIGIN = ('https', 'api.genius.com')
    API_ROOT = '/'
    USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:88.0) Gecko/20100101 Firefox/88.0'

    def __init__(self, sem: int, export: str, auth: str):
        super().__init__(sem=sem, export=export, headers={'Authorization': f'Bearer {auth}'})
        self._log = logging.getLogger('genius')

    def _init_dirs(self):
        os.makedirs(self._output / 'lyrics', exist_ok=True)

    async def _extract(self, url: str) -> str:
        res = await self._request(url=url)
        soup = BeautifulSoup((await res.read()).decode('utf8'), 'lxml')
        target = soup.find('div', {'id': 'lyrics'})
        if not target:
            return
        container = target.parent
        lines = container.find_all('span')
        lyrics = []
        for line in lines:
            for s in line.stripped_strings:
                lyrics.append(s)
        return lyrics

    async def get_lyrics(self, mbz_id: str, title: str, artist: str):
        endpoint = self._build_url('search', {'q': f'{title} {artist}'})
        res = await self._request(url=endpoint)
        data = await res.json()
        hits = data['response']['hits']
        if not hits:
            return
        hit = hits[0]
        res_type = hit['type']
        if res_type != 'song':
            return
        result = hit['result']
        page_url = result['url']
        lyrics = await self._extract(page_url)
        if not lyrics:
            return
        data = {
            'id': mbz_id,
            'title': title,
            'artist': artist,
            'lyrics': lyrics,
            'url': page_url,
        }
        self._log.info(f'lyrics {mbz_id}')
        await self._export('lyrics', mbz_id, data)


async def collect_artists(artists: List[str]):
    mbz = MBZ(5, 'instance/scrape')
    processed = 0
    for chunk in chunked(artists, 20):
        futures = [mbz.search_artist(a) for a in chunk]
        await asyncio.gather(*futures)
        processed += len(chunk)
        mbz._log.info(colored(f'Processed {processed}/{len(artists)}', attrs=['bold']))
    await mbz.close()


async def iter_artist(mbz: MBZ, abz: ABZ, gen: Genius, artist_id: str, artist_name: str):
    processed = set()
    queue_1: Dict[str, str] = {}
    queue_2: Dict[str, str] = {}
    async for rec_id, rec_title in mbz.get_recordings_by_artist(artist_id):
        if rec_id in processed:
            continue
        processed.add(rec_id)
        queue_1[rec_id] = rec_title
        if len(queue_1) < 25:
            continue
        has_feature = await abz.get_features_by_ids(*queue_1.keys())
        queue_2.update({k: queue_1[k] for k in has_feature})
        queue_1.clear()
        if len(queue_2) < 25:
            continue
        lyrics = [gen.get_lyrics(id_, title, artist_name) for id_, title in queue_2.items()]
        await asyncio.gather(*lyrics)
        queue_2.clear()


async def collect_songs(artists: List[Tuple[str, str]]):
    mbz = MBZ(5, 'instance/scrape')
    abz = ABZ(8, 'instance/scrape')
    gen = Genius(10, 'instance/scrape', config('GENIUS_OAUTH'))
    processed = 0
    for chunk in chunked(artists, 20):
        futures = [iter_artist(mbz, abz, gen, artist_id, artist_name) for artist_id, artist_name in chunk]
        await asyncio.gather(*futures)
        processed += len(chunk)
        mbz._log.info(colored(f'Processed {processed}/{len(artists)}', attrs=['bold']))
    await mbz.close()
    await abz.close()
    await gen.close()
