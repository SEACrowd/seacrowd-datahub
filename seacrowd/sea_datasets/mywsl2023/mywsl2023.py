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
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@article{JOHARI2023109338,
title = {MyWSL: Malaysian words sign language dataset},
journal = {Data in Brief},
volume = {49},
pages = {109338},
year = {2023},
issn = {2352-3409},
doi = {https://doi.org/10.1016/j.dib.2023.109338},
url = {https://www.sciencedirect.com/science/article/pii/S2352340923004560},
author = {Rina Tasia Johari and Rizauddin Ramli and Zuliani Zulkoffli and Nizaroyani Saibani},
keywords = {Dataset, Hand gestures, Sign language, Image data},
}
"""

_DATASETNAME = "mywsl2023"

_DESCRIPTION = """\
This dataset contains pictures of hand gestures corresponding to ten commonly-used Malaysian Sign Language (XML) words.
Gestures are performed by five university students who belong to different ethnic groups and are proficient in XML.
Each gesture class contains 350 instances.
"""

_HOMEPAGE = "https://data.mendeley.com/datasets/zvk55p7ktd/1"

_LANGUAGES = ["xml"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {_DATASETNAME: "https://data.mendeley.com/public-files/datasets/zvk55p7ktd/files/7f11b8a0-24e4-45df-af3d-e861f41435ea/file_downloaded"}

_SUPPORTED_TASKS = [Tasks.SIGN_LANGUAGE_RECOGNITION]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SPLITS = [datasets.Split.TRAIN, datasets.Split.TEST]

_LABELS = ["air", "demam", "dengar", "makan", "minum", "salah", "saya", "senyap", "tidur", "waktu"]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MyWsl2023(datasets.GeneratorBasedBuilder):
    """This dataset contains pictures of hand gestures corresponding to ten commonly-used Malaysian Sign Language (XML) words."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    subset_id = _DATASETNAME

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{subset_id}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=subset_id,
        )
    ]

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

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "image_paths": datasets.Sequence(datasets.Value("string")),
                    "texts": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SIGN_LANGUAGE_RECOGNITION]).lower()}":
            features = schemas.image_text_features(label_names=_LABELS)

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

        path = dl_manager.download_and_extract(_URLS[_DATASETNAME])

        for split in _SPLITS:
            split_generators.append(
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "path": os.path.join(path, "MyWSL2023 RAW DATA", split._name),
                    },
                )
            )

        return split_generators

    def _generate_examples(self, path: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        image_folder_paths = [os.path.join(path, folder) for folder in os.listdir(path)]

        for idx, image_folder_path in enumerate(image_folder_paths):
            image_paths = os.listdir(image_folder_path)

            if self.config.schema == "source":
                yield idx, {
                    "image_paths": [os.path.join(image_folder_path, image_path) for image_path in image_paths],
                    "texts": os.path.basename(image_folder_path),
                }

            elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SIGN_LANGUAGE_RECOGNITION]).lower()}":
                yield idx, {
                    "id": os.path.basename(image_folder_path),
                    "image_paths": [os.path.join(image_folder_path, image_path) for image_path in image_paths],
                    "texts": os.path.basename(image_folder_path),
                    "metadata": {
                        "context": "Malaysian Sign Language (XML)",
                        "labels": [os.path.basename(image_folder_path)],
                    },
                }

            else:
                raise ValueError(f"Invalid config: {self.config.name}")
