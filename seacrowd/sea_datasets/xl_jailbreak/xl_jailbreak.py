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
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@misc{deng2023multilingual,
title={Multilingual Jailbreak Challenges in Large Language Models},
author={Yue Deng and Wenxuan Zhang and Sinno Jialin Pan and Lidong Bing},
year={2023},
eprint={2310.06474},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
"""

_DATASETNAME = "xl_jailbreak"

_DESCRIPTION = """\
This dataset contains the data for the paper "Multilingual Jailbreak Challenges in Large Language Models".
"""

_HOMEPAGE = "https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail"

_LANGUAGES = ["jav", "vie", "tha"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.MIT.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {"train": "https://huggingface.co/api/datasets/DAMO-NLP-SG/MultiJail/parquet/default/train/0.parquet"},
}

_SUPPORTED_TASKS = [Tasks.PROMPTING]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class XlJailbreak(datasets.GeneratorBasedBuilder):
    """This dataset contains the data for the paper "Multilingual Jailbreak Challenges in Large Language Models"."""

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

    seacrowd_schema_config: list[SEACrowdConfig] = []

    for seacrowd_schema in _SUPPORTED_SCHEMA_STRINGS:

        seacrowd_schema_config.append(
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{seacrowd_schema}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {seacrowd_schema} schema",
                schema=f"{seacrowd_schema}",
                subset_id=f"{_DATASETNAME}",
            )
        )

    BUILDER_CONFIGS.extend(seacrowd_schema_config)

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value(dtype="int64"),
                    "source": datasets.Value(dtype="string"),
                    "tags": datasets.Value(dtype="string"),
                    "en": datasets.Value(dtype="string"),
                    "zh": datasets.Value(dtype="string"),
                    "it": datasets.Value(dtype="string"),
                    "vi": datasets.Value(dtype="string"),
                    "ar": datasets.Value(dtype="string"),
                    "ko": datasets.Value(dtype="string"),
                    "th": datasets.Value(dtype="string"),
                    "bn": datasets.Value(dtype="string"),
                    "sw": datasets.Value(dtype="string"),
                    "jv": datasets.Value(dtype="string"),
                }
            )

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.PROMPTING]).lower()}":
            features = schemas.ssp_features

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

        is_schema_found = False

        if self.config.schema == "source":
            is_schema_found = True

            df = pd.read_parquet(filepath)

            for index, row in df.iterrows():
                yield index, row.to_dict()

        else:
            for seacrowd_schema in _SUPPORTED_SCHEMA_STRINGS:
                if self.config.schema == seacrowd_schema:
                    is_schema_found = True

                    df = pd.read_parquet(filepath)

                    def row_to_json(row):
                        # Create a dictionary excluding the id column
                        row_dict = {col: row[col] for col in df.columns if col != "id"}
                        # Convert the dictionary to a JSON string
                        return json.dumps(row_dict)

                    # Apply the function to each row and create a new column with the JSON string
                    df["text"] = df.apply(row_to_json, axis=1)

                    df = df[["id", "text"]]

                    print(df)

                    for index, row in df.iterrows():
                        yield index, row.to_dict()

        if not is_schema_found:
            raise ValueError(f"Invalid config: {self.config.name}")