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
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses

_CITATION = """\
@inproceedings{maxwelll-smith-foley-2023-automated,
title = "Automated speech recognition of {I}ndonesian-{E}nglish language lessons on {Y}ou{T}ube using transfer learning",
author = "Maxwell-Smith, Zara and Foley, Ben",
editor = "Serikov, Oleg
        and Voloshina, Ekaterina
        and Postnikova, Anna
        and Klyachko, Elena
        and Vylomova, Ekaterina
        and Shavrina, Tatiana
        and Le Ferrand, Eric
        and Malykh, Valentin
        and Tyers, Francis
        and Arkhangelskiy, Timofey
        and Mikhailov, Vladislav",
    booktitle = "Proceedings of the Second Workshop on NLP Applications to Field Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.fieldmatters-1.1",
    doi = "10.18653/v1/2023.fieldmatters-1.1",
    pages = "1--16",
    abstract = "Experiments to fine-tune large multilingual models with limited data from a specific domain or setting has potential
    to improve automatic speech recognition (ASR) outcomes. This paper reports on the use of the Elpis ASR pipeline to fine-tune two
    pre-trained base models, Wav2Vec2-XLSR-53 and Wav2Vec2-Large-XLSR-Indonesian, with various mixes of data from 3 YouTube channels
    teaching Indonesian with English as the language of instruction. We discuss our results inferring new lesson audio (22-46%
    word error rate) in the context of speeding data collection in diverse and specialised settings. This study is an example of how
    ASR can be used to accelerate natural language research, expanding ethically sourced data in low-resource settings.",
}
"""

_DATASETNAME = "oil"

_DESCRIPTION = """\
The Online Indonesian Learning (OIL) dataset or corpus currently contains lessons from three Indonesian teachers who have posted content on YouTube.
"""

_HOMEPAGE = "https://huggingface.co/datasets/ZMaxwell-Smith/OIL"

_LANGUAGES = ["eng", "ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_NC_ND_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {"train": "https://huggingface.co/api/datasets/ZMaxwell-Smith/OIL/parquet/default/train/0.parquet"},
}

_SUPPORTED_TASKS = []
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class OIL(datasets.GeneratorBasedBuilder):
    """The Online Indonesian Learning (OIL) dataset or corpus currently contains lessons from three Indonesian teachers who have posted content on YouTube."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "audio": datasets.Audio(decode=False),
                    "label": datasets.ClassLabel(num_classes=98),
                }
            )

        else:
            raise ValueError(f"Invalid config: {self.config.name}")

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
        train_path = dl_manager.download_and_extract(urls["train"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":

            df = pd.read_parquet(filepath)

            for index, row in df.iterrows():
                yield index, row.to_dict()

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
