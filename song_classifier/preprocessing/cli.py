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

import click

from .digesters import digesters, driver


@click.command()
@click.option('-i', '--fin', required=True, type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--fout', required=True, type=click.Path())
@click.option('-t', '--datatype', required=True, type=click.Choice(digesters.keys()))
def preprocess(fin, fout, datatype):
    driver(digesters[datatype], in_dir=fin, output=fout)


COMMANDS = [preprocess]