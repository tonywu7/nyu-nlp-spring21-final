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

import logging
from pathlib import Path

from .util.settings import Settings
from .util.types import PathLike

app: Application = None


class Application:
    def __init__(self, instance: PathLike, profiles=(), **kwargs):
        global app
        app = self

        self.root = Path(instance)
        self.log = logging.getLogger('soundfinder')
        self.settings: Settings

        self._load_settings(profiles, **kwargs)
        self._init_hierarchy()
        self._init_components()

    def _init_hierarchy(self):
        self.root.mkdir(exist_ok=True)

    def _load_settings(self, profiles, **kwargs):
        settings = Settings()
        settings.from_pyfile(Path(__file__).with_name('config.py'))
        settings.from_pyfile(self.root / 'config.py')
        for p in profiles:
            settings.from_pyfile(Path(p))
        settings['instance_path'] = self.root
        settings.update(kwargs)
        self.settings = settings

    def _init_components(self):
        from .database import Database
        Database(self.root / 'index.db', **self.settings['db':])


def get_app() -> Application:
    return app


def get_settings() -> Settings:
    return app.settings
