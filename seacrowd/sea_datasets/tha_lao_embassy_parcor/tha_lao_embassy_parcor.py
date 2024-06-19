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
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
Wannaphong Phatthiyaphaibun. (2021). PyThaiNLP/Thai-Lao-Parallel-Corpus: \
Thai Lao Parallel corpus v0.7 (v0.7). Zenodo \
https://doi.org/10.5281/zenodo.5807093"""

_DATASETNAME = "tha_lao_embassy_parcor"

_DESCRIPTION = """\
Thai-Lao Parallel Corpus contains equivalent Thai and Lao sentence pairs \
derived from the website of the Royal Thai Embassy in Vientiane, Laos.
"""

_HOMEPAGE = "https://github.com/PyThaiNLP/Thai-Lao-Parallel-Corpus/tree/master"
_LANGUAGES = ["tha", "lao"]
_LICENSE = Licenses.CC0_1_0.value

_LOCAL = False
_URLS = {_DATASETNAME: "https://github.com/PyThaiNLP/Thai-Lao-Parallel-Corpus/raw/master/vientiane-thaiembassy-sent.csv"}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "0.7.0"
_SEACROWD_VERSION = "2024.06.20"


class ThaLaoEmbassyParcorDataset(datasets.GeneratorBasedBuilder):
    """Thai-Lao Parallel Corpus contains equivalent Thai and Lao sentence pairs \
    derived from the website of the Royal Thai Embassy in Vientiane, Laos."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "t2t"

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
                    "lao_sent": datasets.Value("string"),
                    "thai_sent": datasets.Value("string"),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        filename = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(filename),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        dataset = pd.read_csv(filepath)

        if self.config.schema == "source":
            for i, row in dataset.iterrows():
                yield i, {"lao_sent": row["lao_sent"], "thai_sent": row["thai_sent"]}

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for i, row in dataset.iterrows():
                yield i, {
                    "id": i,
                    "text_1": row["lao_sent"],
                    "text_2": row["thai_sent"],
                    "text_1_name": "lao",
                    "text_2_name": "tha",
                }
