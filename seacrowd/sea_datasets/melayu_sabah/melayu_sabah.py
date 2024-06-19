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

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_DATASETNAME = "melayu_sabah"


_DESCRIPTION = """\
Korpus Variasi Bahasa Melayu: Sabah is a language corpus sourced from various folklores in Melayu Sabah dialect.
"""

_CITATION = """\
@misc{melayusabah,
  author = {Hiroki Nomoto},
  title = {Melayu_Sabah},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/matbahasa/Melayu_Sabah}},
  commit = {90a46c8268412ccc1f29cdcbbd47354474f12d50}
}
"""

_HOMEPAGE = "https://github.com/matbahasa/Melayu_Sabah"


_LANGUAGES = ["msi"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    "sabah201701": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201701.txt",
    "sabah201702": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201702.txt",
    "sabah201901": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201901.txt",
    "sabah201902": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201902.txt",
    "sabah201903": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201903.txt",
    "sabah201904": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201904.txt",
    "sabah201905": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201905.txt",
    "sabah201906": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201906.txt",
    "sabah201907": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201907.txt",
    "sabah201908": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201908.txt",
    "sabah201909": "https://raw.githubusercontent.com/matbahasa/Melayu_Sabah/master/Sabah201909.txt",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MelayuSabah(datasets.GeneratorBasedBuilder):
    """Korpus Variasi Bahasa Melayu:
    Sabah is a language corpus sourced from various folklores in Melayu Sabah dialect."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()

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
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_ssp":
            features = schemas.self_supervised_pretraining.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = [_URLS[key] for key in _URLS.keys()]
        data_path = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_path[0], "split": "train", "other_path": data_path[1:]},
            )
        ]

    def _generate_examples(self, filepath: Path, split: str, other_path: List) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        filepaths = [filepath] + other_path
        data = []
        for filepath in filepaths[:2]:
            with open(filepath, "r") as f:
                sentences = [line.rstrip() for line in f.readlines()]
                sentences = [sentence.split("\t")[-1] for sentence in sentences]
                data.append("\n".join(sentences))

        for filepath in filepaths[2:]:
            with open(filepath, "r") as f:
                data.append([line.rstrip() for line in f.readlines()])

        for id, text in enumerate(data):
            yield id, {"id": id, "text": text}
