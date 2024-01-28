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
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
{@inproceedings{Maxwell-Smith_Foley_2023_Automated,
  title     = {{Automated speech recognition of Indonesian-English language lessons on YouTube using transfer learning}},
  author    = {Maxwell-Smith, Zara and Foley, Ben},
  booktitle = {Proceedings of the {Second Workshop on NLP Applications to Field Linguistics (EACL)}},
  pages     = {},
  year      = {forthcoming}
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

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


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
                    "audio": datasets.Audio(decode=False),
                    "label": datasets.ClassLabel(num_classes=98),
                }
            )

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SPEECH_RECOGNITION]).lower()}":
            features = schemas.speech_text_features

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

                    base_folder = os.path.dirname(filepath)
                    base_folder = os.path.join(base_folder, split)

                    if not os.path.exists(base_folder):
                        os.makedirs(base_folder)

                    audio_paths = []

                    for _, row in df.iterrows():
                        audio_dict = row["audio"]
                        file_name = audio_dict["path"]

                        path = os.path.join(base_folder, file_name)

                        audio_dict["path"] = path

                        with open(path, "wb") as f:
                            f.write(audio_dict["bytes"])

                        audio_paths.append(path)

                    df.rename(columns={"label": "text"}, inplace=True)

                    df["path"] = audio_paths

                    df["id"] = df.index
                    df = df.assign(speaker_id="").astype({"speaker_id": "str"})
                    df = df.assign(metadata=[{"speaker_age": 0, "speaker_gender": ""}] * len(df)).astype({"metadata": "object"})

                    for index, row in df.iterrows():
                        yield index, row.to_dict()

        if not is_schema_found:
            raise ValueError(f"Invalid config: {self.config.name}")
