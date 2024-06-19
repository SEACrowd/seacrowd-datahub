# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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

from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from pandas import read_excel

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@inproceedings{koto-koto-2020-towards,
    title = "Towards Computational Linguistics in {M}inangkabau Language:
    Studies on Sentiment Analysis and Machine Translation",
    author = "Koto, Fajri  and
        Koto, Ikhwan",
    editor = "Nguyen, Minh Le  and
        Luong, Mai Chi  and
        Song, Sanghoun",
    booktitle = "Proceedings of the 34th Pacific Asia Conference on Language,
    Information and Computation",
    month = oct,
    year = "2020",
    address = "Hanoi, Vietnam",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.paclic-1.17",
    pages = "138--148",
}
"""

_DATASETNAME = "minang_senti"

_DESCRIPTION = """\
We release the Minangkabau corpus for sentiment analysis by manually translating
5,000 sentences of Indonesian sentiment analysis corpora. In this work, we
conduct a binary sentiment classification on positive and negative sentences by
first manually translating the Indonesian sentiment analysis corpus to the
Minangkabau language (Agam-Tanah Datar dialect)
"""

_HOMEPAGE = "https://github.com/fajri91/minangNLP"

_LANGUAGES = ["ind", "min"]

_LICENSE = Licenses.MIT.value

_LOCAL = False

_BASE_URL = "https://github.com/fajri91/minangNLP/raw/master/sentiment/data/folds/{split}{index}.xlsx"

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # text

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MinangSentiDataset(datasets.GeneratorBasedBuilder):
    """Binary sentiment classification on manually translated Minangkabau corpus."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in _LANGUAGES:
        BUILDER_CONFIGS += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} {subset} source schema",
                schema="source",
                subset_id=subset,
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset}_{_SEACROWD_SCHEMA}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {subset} SEACrowd schema",
                schema=_SEACROWD_SCHEMA,
                subset_id=subset,
            ),
        ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{_LANGUAGES[0]}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "minang": datasets.Value("string"),
                    "indo": datasets.Value("string"),
                    "sentiment": datasets.ClassLabel(names=["positive", "negative"]),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = schemas.text_features(label_names=["positive", "negative"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        train_urls = [_BASE_URL.format(split="train", index=i) for i in range(5)]
        test_urls = [_BASE_URL.format(split="test", index=i) for i in range(5)]
        dev_urls = [_BASE_URL.format(split="dev", index=i) for i in range(5)]

        train_paths = [Path(dl_manager.download(url)) for url in train_urls]
        test_paths = [Path(dl_manager.download(url)) for url in test_urls]
        dev_paths = [Path(dl_manager.download(url)) for url in dev_urls]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_paths,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_paths,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dev_paths,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        key = 0
        for file in filepath:
            data = read_excel(file)
            for _, row in data.iterrows():
                if self.config.schema == "source":
                    yield key, {
                        "minang": row["minang"],
                        "indo": row["indo"],
                        "sentiment": row["sentiment"],
                    }
                elif self.config.schema == _SEACROWD_SCHEMA:
                    yield key, {
                        "id": str(key),
                        "text": row["minang"] if self.config.subset_id == "min" else row["indo"],
                        "label": row["sentiment"],
                    }
                key += 1
