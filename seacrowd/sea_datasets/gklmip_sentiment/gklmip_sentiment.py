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

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@InProceedings{,
author="Jiang, Shengyi
and Huang, Xiuwen
and Cai, Xiaonan
and Lin, Nankai",
title="Pre-trained Models and Evaluation Data for the Myanmar Language",
booktitle="The 28th International Conference on Neural Information Processing",
year="2021",
publisher="Springer International Publishing",
address="Cham",
}
"""

_DATASETNAME = "gklmip_sentiment"
_DESCRIPTION = """\
The GKLMIP Product Sentiment Dataset is a Burmese dataset for sentiment analysis. \
It was created by crawling comments on an e-commerce website. The sentiment labels range \
from 1 to 5, with 1 and 2 being negative, 3 and 4 being neutral, and 5 being positive.
"""

_HOMEPAGE = "https://github.com/GKLMIP/Pretrained-Models-For-Myanmar/tree/main"
_LANGUAGES = ["mya"]
_LICENSE = Licenses.UNKNOWN.value
_LOCAL = False

_URLS = {
    _DATASETNAME: "https://github.com/GKLMIP/Pretrained-Models-For-Myanmar/raw/main/Product%20Sentiment%20Dataset.zip",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_LABELS = [1, 2, 3, 4, 5]


class GklmipSentimentDataset(datasets.GeneratorBasedBuilder):
    """The GKLMIP Product Sentiment Dataset is a Burmese dataset for sentiment analysis."""

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
            features = datasets.Features({"bpe": datasets.Value("string"), "text": datasets.Value("string"), "label": datasets.Value("string")})

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(_LABELS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "product_sentiment_dataset_train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "product_sentiment_dataset_test.json"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "product_sentiment_dataset_dev.json"),
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        with open(filepath) as file:
            dataset = json.load(file)

        if self.config.schema == "source":
            for i, line in enumerate(dataset):
                yield i, {"bpe": line["bpe"], "text": line["text"], "label": line["label"]}

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for i, line in enumerate(dataset):
                yield i, {"id": i, "text": line["text"], "label": line["label"]}
