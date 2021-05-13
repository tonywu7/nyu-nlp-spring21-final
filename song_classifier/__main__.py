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

import logging
import random
import sys
from textwrap import dedent

import click
import nltk

from .algorithms import cosine, knn2, tfidf2
from .app import N_NEIGHBORS, TEXT_PROCESSORS, Application, get_settings
from .collector import samples
from .config import KEYWORDS_4
from .scoring import print_score, score, stats
from .util.logger import config_logging
from .util.settings import Settings


def init_nltk():
    """Initialize nltk"""
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


@click.group()
@click.option('-a', '--instance', required=True, type=click.Path(exists=True, file_okay=False),
              help='Directory containing the song database with name `index.db`')
@click.option('-v', '--verbose', is_flag=True, default=False)
@click.pass_context
def main(ctx, instance, verbose):
    ctx.ensure_object(dict)
    config_logging(level=10 if verbose else 20)
    settings = Settings()
    settings['DB_ECHO'] = verbose
    Application(instance)
    init_nltk()


ALGORITHMS = {
    'cosine': cosine.run,
    'knn': knn2.run,
}


@main.command()
@click.option('-s', '--seed', required=False, type=click.INT, default=None,
              help='Random seed controlling training/testing set creation')
@click.option('-r', '--training-set-ratio', required=False, type=click.FLOAT, default=.8,
              help='Size ratio of training to testing dataset; decimal between 0 and 1')
@click.option('-x', '--algorithm', required=False, type=click.Choice(['cosine', 'knn']), default='cosine',
              help='The kind of classification function to use')
@click.option('-p', '--preprocessors', required=False, type=click.Choice(tfidf2.PROCESSOR_PRESETS.keys()), default='lexical,syntactic',
              help='Text preprocessor presets')
@click.option('-m', '--min-playlists', required=False, type=click.INT, default=2,
              help='Minimum number of playlists a song must appear in for it to enter the sample set')
@click.option('-n', '--n-neighbors', required=False, type=click.INT, default=5,
              help='(For k-NN only) k-NN N neighbors')
def run_test(seed, training_set_ratio, algorithm, preprocessors, min_playlists, n_neighbors):
    log = logging.getLogger('main')
    if not seed:
        seed = random.randrange(sys.maxsize)
    RNG = random.Random(seed)
    runner = ALGORITHMS.get(algorithm)
    if not runner:
        log.error(f'No such algorithm {algorithm}')
        raise click.Abort()
    processors = tfidf2.PROCESSOR_PRESETS.get(preprocessors)
    if not processors:
        log.error(f'No such preprocessor preset {preprocessors}')
        raise click.Abort()
    config = dedent(f"""\
        Program config
        random seed         = {seed}
        algorithm           = {algorithm}
        processors          = {preprocessors}
        train/test ratio    = {training_set_ratio}
        min playlist        = {min_playlists}
        (kNN) N neighbors   = {n_neighbors}\
    """)
    log.info(config)
    settings = get_settings()
    settings[TEXT_PROCESSORS] = processors
    settings[N_NEIGHBORS] = n_neighbors
    training, testing = samples(RNG, training_set_ratio, KEYWORDS_4.keys(), KEYWORDS_4, min_playlists)
    print('Dataset stats:')
    for k in training.keys():
        print(f'{k}: training={len(training[k])} testing={len(testing[k])}')
    predictions, truths = runner(training, testing)
    stats(predictions, truths, KEYWORDS_4.keys())
    print_score(*score(predictions, truths, KEYWORDS_4.keys()))


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
