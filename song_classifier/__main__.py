# Copyright 2021 Nour Abdelmoneim, Thomas Kim, Lucas Ortiz, Tony Wu
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
import random
from importlib import import_module
from pathlib import Path

import click
import simplejson as json

from .app import Application
from .util.importutil import iter_module_tree
from .util.logger import config_logging
from .util.settings import Settings


@click.group()
@click.option('-a', '--instance', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('-v', '--verbose', is_flag=True, default=False)
@click.option('-s', '--seed', type=click.INT, required=False)
@click.pass_context
def main(ctx, instance, verbose, seed=None):
    if seed:
        random.seed(seed)
    ctx.ensure_object(dict)
    config_logging(level=10 if verbose else 20)
    settings = Settings()
    settings['DB_ECHO'] = verbose
    ctx.obj['SEED'] = seed
    Application(instance)


def find_commands():
    for path in iter_module_tree(str(Path(__file__).parent), depth=2):
        try:
            ctl = import_module(f'.{".".join(path)}.cli', __package__)
        except ModuleNotFoundError as e:
            if e.name[:15] == 'song_classifier' and e.name[-3:] == 'cli':
                continue
            raise
        cmd = getattr(ctl, 'COMMANDS', [])
        for c in cmd:
            main.add_command(c)


@main.command()
def cosine():
    from .implementations.tfidf import run
    for i in range(1):
        print(f'Run #{i}')
        run()


@main.command()
def knn():
    from .implementations.knn import run
    for i in range(1):
        print(f'Run #{i}')
        run()


@main.command()
@click.pass_context
def master(ctx):
    from .mastertest import test
    test(ctx.obj['SEED'])


@main.command()
@click.option('-i', '--input-file', required=True)
def mbz_artists(input_file):
    from .web import collect_artists
    with open(input_file, 'r') as f:
        artists = json.load(f)
    asyncio.run(collect_artists(artists))


@main.command()
@click.option('-i', '--input-dir', required=True)
def nuclear(input_dir):
    from .web import collect_songs
    artists = []
    for p in Path(input_dir).iterdir():
        if p.suffix != '.json':
            continue
        with open(p) as f:
            data = json.load(f)
        artists.append((data['id'], data['name']))
    asyncio.run(collect_songs(artists))


@main.command()
def version():
    from . import __version__
    from .database import VERSION
    print('NYU Spring 2021 NLP Final Group #31\n'
          f'Application version {__version__}\n'
          f'Database version {VERSION}')


if __name__ == '__main__':
    # find_commands()
    main(prog_name='python -m song_classifier')
