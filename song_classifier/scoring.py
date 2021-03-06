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

from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def stats(predictions: Dict[str, str], ground_truths: Dict[str, str], categories: List[str]):
    predictions = np.array([*predictions.values()])
    ground_truths = np.array([*ground_truths.values()])
    print('# results')
    for k in categories:
        print(f'{k}: {np.count_nonzero(predictions == k)} predicted {np.count_nonzero(ground_truths == k)} expected')


def score(predictions: Dict[str, str], ground_truths: Dict[str, str], categories: List[str]) -> Dict[str, Tuple[float, float, float]]:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    gt = np.array([*ground_truths.values()])
    counts = {k: np.count_nonzero(gt == k) for k in categories}

    confusion = pd.DataFrame(index=categories, columns=categories, data=0)

    for k, prediction in predictions.items():
        truth = ground_truths[k]
        confusion.loc[truth, prediction] += 1
        if prediction == truth:
            tp[truth] += 1
        else:
            fp[prediction] += 1
            fn[truth] += 1

    for k in confusion:
        confusion.loc[k, :] /= confusion.loc[k, :].sum()

    precisions = defaultdict(int)
    recall = defaultdict(int)
    fscore = defaultdict(int)

    for c in categories:
        try:
            precisions[c] = tp[c] / (tp[c] + fp[c])
        except ZeroDivisionError:
            precisions[c] = -1

    for c in categories:
        try:
            recall[c] = tp[c] / (tp[c] + fn[c])
        except ZeroDivisionError:
            recall[c] = -1

    for c in categories:
        try:
            fscore[c] = tp[c] / (tp[c] + .5 * (fp[c] + fn[c]))
        except ZeroDivisionError:
            fscore[c] = -1

    percat = {}
    for c in categories:
        p, r, f = precisions[c], recall[c], fscore[c]
        percat[c] = (p, r, f)

    try:
        accuracy = sum(tp.values()) / (sum(tp.values()) + .5 * (sum(fp.values()) + sum(fn.values())))
    except ZeroDivisionError:
        accuracy = float('nan')

    macro = {
        'precision': mean(precisions.values()),
        'recall': mean(recall.values()),
        'f-score': mean(fscore.values()),
    }

    weighted = {}
    for name, stat in zip(('precisions', 'recall', 'fscore'), (precisions, recall, fscore)):
        try:
            weighted[name] = sum(stat[k] * counts[k] for k in categories) / sum(counts.values()),
        except ZeroDivisionError:
            weighted[name] = float('nan')

    return percat, macro, weighted, accuracy, confusion


def export(predictions: Dict[str, str], ground_truths: Dict[str, str]):
    df_pred = pd.DataFrame(columns=['title', 'prediction'], data=predictions.items())
    df_pred = df_pred.set_index(['title'])

    df_truth = pd.DataFrame(columns=['title', 'truth'], data=ground_truths.items())
    df_truth = df_truth.set_index(['title'])

    df = df_pred.join(df_truth)
    filename = f'predictions.{datetime.now().strftime("%Y%m%d.%H%M%S")}.csv'
    df.to_csv(filename, index=True, header=True)


def print_score(percat, macro, weighted, accuracy, confusion):
    print('  Per category:')
    for k, (p, r, f) in percat.items():
        print(f'    {k}: precision={p:.3f} recall={r:.3f} f-score={f:.3f}')
    print('  Macro:')
    for k, v in macro.items():
        print(f'    {k}: {v}')
    print('  Weighted:')
    for k, v in weighted.items():
        print(f'    {k}: {v}')
    print(f'  Accuracy: {accuracy}')
    print('  Confusion')
    print(confusion)
