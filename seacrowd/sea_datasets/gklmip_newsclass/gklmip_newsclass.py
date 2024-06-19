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

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import numpy as np
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{,
author="Jiang, Shengyi
and Fu, Sihui
and Lin, Nankai
and Fu, Yingwen",
title="Pre-trained Models and Evaluation Data for the Khmer Language",
year="2021",
publisher="Tsinghua Science and Technology",
}
"""

_DATASETNAME = "gklmip_newsclass"

_DESCRIPTION = """\
The GKLMIP Khmer News Dataset is scraped from the Voice of America Khmer website. \
The news articles in the dataset are categorized into 8 categories: culture, economics, education, \
environment, health, politics, rights and science.
"""

_HOMEPAGE = "https://github.com/GKLMIP/Pretrained-Models-For-Khmer"
_LANGUAGES = ["khm"]

_LICENSE = Licenses.UNKNOWN.value
_LOCAL = False

_URLS = {
    _DATASETNAME: "https://github.com/GKLMIP/Pretrained-Models-For-Khmer/raw/main/NewsDataset.zip",
}

_SUPPORTED_TASKS = [Tasks.TOPIC_MODELING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_TAGS = ["culture", "economic", "education", "environment", "health", "politics", "right", "science"]


class GklmipNewsclass(datasets.GeneratorBasedBuilder):
    """\
    The GKLMIP Khmer News Dataset is scraped from the Voice of America Khmer website. \
    The news articles in the dataset are categorized into 8 categories: culture, economics, education, \
    environment, health, politics, rights and science.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "text"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "culture": datasets.Value("bool"),
                    "economic": datasets.Value("bool"),
                    "education": datasets.Value("bool"),
                    "environment": datasets.Value("bool"),
                    "health": datasets.Value("bool"),
                    "politics": datasets.Value("bool"),
                    "right": datasets.Value("bool"),
                    "science": datasets.Value("bool"),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(_TAGS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.csv"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.csv"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        dataset = pd.read_csv(filepath)
        reverse_encoding = dict(zip(range(len(_TAGS)), _TAGS))
        if self.config.schema == "source":
            for i, row in dataset.iterrows():
                yield i, {
                    "text": row["text"],
                    "culture": row["culture"],
                    "economic": row["economic"],
                    "education": row["education"],
                    "environment": row["environment"],
                    "health": row["health"],
                    "politics": row["politics"],
                    "right": row["right"],
                    "science": row["science"],
                }

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for i, row in dataset.iterrows():
                yield i, {"id": i, "text": row["text"], "label": reverse_encoding[np.argmax(row[_TAGS])]}
