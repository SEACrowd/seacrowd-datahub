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

from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@inproceedings{PhoNER_COVID19,
title     = {{COVID-19 Named Entity Recognition for Vietnamese}},
author    = {Thinh Hung Truong and Mai Hoang Dao and Dat Quoc Nguyen},
booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
year      = {2021}
}
"""

_DATASETNAME = "pho_ner_covid"

_DESCRIPTION = """\
A named entity recognition dataset for Vietnamese with 10 newly-defined entity types in the context of the COVID-19 pandemic.
Data is extracted from news articles and manually annotated. In total, there are 34 984 entities over 10 027 sentences.
"""

_HOMEPAGE = "https://github.com/VinAIResearch/PhoNER_COVID19/tree/main"

_LANGUAGES = ["vie"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "word_level": {
            "dev": "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/dev_word.json",
            "train": "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/train_word.json",
            "test": "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/test_word.json",
        },
        "syllable_level": {
            "dev": "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/dev_syllable.json",
            "train": "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/train_syllable.json",
            "test": "https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/syllable/test_syllable.json",
        },
    }
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SUPPORTED_SCHEMA_STRING_MAP: Dict[Tasks, str] = {}

for task, schema_string in zip(_SUPPORTED_TASKS, _SUPPORTED_SCHEMA_STRINGS):
    _SUPPORTED_SCHEMA_STRING_MAP[task] = schema_string

_SUBSETS = ["word_level", "syllable_level"]
_SPLITS = ["train", "dev", "test"]
_TAGS = [
    "O",
    "B-ORGANIZATION",
    "I-ORGANIZATION",
    "B-SYMPTOM_AND_DISEASE",
    "I-SYMPTOM_AND_DISEASE",
    "B-LOCATION",
    "B-DATE",
    "B-PATIENT_ID",
    "B-AGE",
    "B-NAME",
    "I-DATE",
    "B-JOB",
    "I-LOCATION",
    "B-TRANSPORTATION",
    "B-GENDER",
    "I-TRANSPORTATION",
    "I-JOB",
    "I-NAME",
    "I-AGE",
    "I-PATIENT_ID",
    "I-GENDER",
]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class PhoNerCovidDataset(datasets.GeneratorBasedBuilder):
    """A named entity recognition dataset for Vietnamese with 10 newly-defined entity types in the context of the COVID-19 pandemic."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []

    for subset_id in _SUBSETS:
        BUILDER_CONFIGS.append(
            SEACrowdConfig(
                name=f"{subset_id}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} source schema",
                schema="source",
                subset_id=subset_id,
            )
        )

        seacrowd_schema_config: list[SEACrowdConfig] = []

        for seacrowd_schema in _SUPPORTED_SCHEMA_STRINGS:

            seacrowd_schema_config.append(
                SEACrowdConfig(
                    name=f"{subset_id}_{seacrowd_schema}",
                    version=SEACROWD_VERSION,
                    description=f"{_DATASETNAME} {seacrowd_schema} schema",
                    schema=f"{seacrowd_schema}",
                    subset_id=subset_id,
                )
            )

        BUILDER_CONFIGS.extend(seacrowd_schema_config)

    DEFAULT_CONFIG_NAME = f"{_SUBSETS[0]}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "words": datasets.Sequence(datasets.Value("string")),
                    "tags": datasets.Sequence(datasets.ClassLabel(names=_TAGS)),
                }
            )

        elif self.config.schema == _SUPPORTED_SCHEMA_STRING_MAP[Tasks.NAMED_ENTITY_RECOGNITION]:
            features = schemas.seq_label_features(label_names=_TAGS)

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

        split_generators = []

        for split in _SPLITS:
            path = dl_manager.download_and_extract(_URLS[_DATASETNAME][self.config.subset_id][split])

            split_generators.append(
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "path": path,
                    },
                )
            )

        return split_generators

    def _generate_examples(self, path: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        idx = 0
        df = pd.read_json(path, lines=True)

        if self.config.schema == "source":
            for _, row in df.iterrows():
                yield idx, row.to_dict()
                idx += 1

        elif self.config.schema == _SUPPORTED_SCHEMA_STRING_MAP[Tasks.NAMED_ENTITY_RECOGNITION]:
            df["id"] = df.index
            df = df.rename(columns={"words": "tokens", "tags": "labels"})

            for _, row in df.iterrows():
                yield idx, row.to_dict()
                idx += 1

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
