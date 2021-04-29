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

from importlib import import_module
from pathlib import Path

import click

from .app import Application
from .util.importutil import iter_module_tree
from .util.logger import config_logging
from .util.settings import Settings


@click.group()
@click.option('-a', '--instance', default='instance', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('-v', '--verbose', is_flag=True, default=False)
@click.pass_context
def main(ctx, instance, verbose):
    ctx.ensure_object(dict)
    config_logging(level=10 if verbose else 20)
    settings = Settings()
    settings['DB_ECHO'] = verbose
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
def version():
    from . import __version__
    from .database import VERSION
    print('NYU Spring 2021 NLP Final Group #31\n'
          f'Application version {__version__}\n'
          f'Database version {VERSION}')


if __name__ == '__main__':
    find_commands()
    main(prog_name='python -m song_classifier')
