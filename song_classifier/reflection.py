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

from typing import Dict

import pandas as pd

from .implementations.tfidf import Vector, get_similarity


def vector_distances(category_vectors: Dict[str, Vector]) -> pd.DataFrame:
    dist = pd.DataFrame(index=category_vectors.keys(), columns=category_vectors.keys())
    for k, v in category_vectors.items():
        similarities = get_similarity(v, category_vectors)
        for x, y in similarities:
            dist.loc[k, x] = y
    return dist
