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

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{Astuti2023,
title = {Code-Mixed Sentiment Analysis using Transformer for Twitter Social Media Data},
journal = {International Journal of Advanced Computer Science and Applications},
doi = {10.14569/IJACSA.2023.0141053},
url = {http://dx.doi.org/10.14569/IJACSA.2023.0141053},
year = {2023},
publisher = {The Science and Information Organization},
volume = {14},
number = {10},
author = {Laksmita Widya Astuti and Yunita Sari and Suprapto}
}
"""

_DATASETNAME = "indonglish"
_DESCRIPTION = """\
Indonglish-dataset was constructed based on keywords derived from the
sociolinguistic phenomenon observed among teenagers in South Jakarta. The
dataset was designed to tackle the semantic task of sentiment analysis,
incorporating three distinct label categories: positive, negative, and
neutral. The annotation of the dataset was carried out by a panel of five
annotators, each possessing expertise language and data science.
"""

_HOMEPAGE = "https://github.com/laksmitawidya/indonglish-dataset"
_LANGUAGES = ["ind"]
_LICENSE = Licenses.UNKNOWN.value
_LOCAL = False

_URLS = {
    "skenario-orig": {
        "train": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario-ori/train.csv",
        "validation": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario-ori/validation.csv",
        "test": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario-ori/test.csv",
    },
    "skenario1": {
        "train": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario1/training.csv",
        "validation": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario1/validation.csv",
        "test": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario1/test.csv",
    },
    "skenario2": {
        "train": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario2/training.csv",
        "validation": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario2/validation.csv",
        "test": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario2/test.csv",
    },
    "skenario3": {
        "train": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario3/training.csv",
        "validation": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario3/validation.csv",
        "test": "https://raw.githubusercontent.com/laksmitawidya/indonglish-dataset/master/skenario3/test.csv",
    },
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class Indonglish(datasets.GeneratorBasedBuilder):
    """Indonglish dataset for sentiment analysis from https://github.com/laksmitawidya/indonglish-dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "text"
    _LABELS = ["Positif", "Negatif", "Netral"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
    ]
    for i in range(1, 4):
        BUILDER_CONFIGS += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_skenario{i}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} source schema",
                schema="source",
                subset_id=f"{_DATASETNAME}_skenario{i}",
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_skenario{i}_seacrowd_{SEACROWD_SCHEMA_NAME}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} SEACrowd schema",
                schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
                subset_id=f"{_DATASETNAME}_skenario{i}",
            ),
        ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tweet": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=self._LABELS),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(self._LABELS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        if "skenario" in self.config.name:
            setting = self.config.name.split("_")[1]
        else:
            setting = "skenario-orig"

        data_paths = {
            setting: {
                "train": Path(dl_manager.download_and_extract(_URLS[setting]["train"])),
                "validation": Path(dl_manager.download_and_extract(_URLS[setting]["validation"])),
                "test": Path(dl_manager.download_and_extract(_URLS[setting]["test"])),
            }
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_paths[setting]["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_paths[setting]["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_paths[setting]["validation"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        # read csv file
        with open(filepath, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            csv_data = [row for row in csv_reader]
        csv_data = csv_data[1:]  # remove header

        num_sample = len(csv_data)

        for i in range(num_sample):
            if self.config.schema == "source":
                example = {
                    "id": str(i),
                    "tweet": csv_data[i][0],
                    "label": csv_data[i][1],
                }
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                example = {
                    "id": str(i),
                    "text": csv_data[i][0],
                    "label": csv_data[i][1],
                }

            yield i, example
