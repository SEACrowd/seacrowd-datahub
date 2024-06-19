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

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@misc{nomoto2018melayustandardlisan,
	author = {Hiroki Nomoto},
	title = {Korpus Variasi Bahasa Melayu: Standard Lisan},
	year = {2018},
	url = {https://github.com/matbahasa/Melayu_Standard_Lisan}
}
"""

_DATASETNAME = "melayu_standard_lisan"


_DESCRIPTION = """\
Korpus Variasi Bahasa Melayu: Standard Lisan is a language corpus sourced from monologues of various melayu folklores.
"""


_HOMEPAGE = "https://github.com/matbahasa/Melayu_Standard_Lisan"


_LANGUAGES = ["zlm"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    "kl201701": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201701.txt",
    "kl201702": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201702.txt",
    "kl201703": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201703.txt",
    "kl201704": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201704.txt",
    "kl201705": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201705.txt",
    "kl201706": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201706.txt",
    "kl201707": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201707.txt",
    "kl201708": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201708.txt",
    "kl201709": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201709.txt",
    "kl201710": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201710.txt",
    "kl201711": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201711.txt",
    "kl201712": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201712.txt",
    "kl201713": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201713.txt",
    "kl201714": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201714.txt",
    "kl201715": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201715.txt",
    "kl201716": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201716.txt",
    "kl201717": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201717.txt",
    "kl201718": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201718.txt",
    "kl201719": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201719.txt",
    "kl201720": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201720.txt",
    "kl201721": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201721.txt",
    "kl201722": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201722.txt",
    "kl201723": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201723.txt",
    "kl201724": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201724.txt",
    "kl201725": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201725.txt",
    "kl201726": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201726.txt",
    "kl201727": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201727.txt",
    "kl201728": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201728.txt",
    "kl201729": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201729.txt",
    "kl201730": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201730.txt",
    "kl201731": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201731.txt",
    "kl201732": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201732.txt",
    "kl201733": "https://raw.githubusercontent.com/matbahasa/Melayu_Standard_Lisan/master/KL201733.txt",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MelayuStandardLisan(datasets.GeneratorBasedBuilder):
    """Korpus Variasi Bahasa Melayu:
    Standard Lisan is a language corpus sourced from monologues of various melayu folklores."""

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
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.self_supervised_pretraining.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE
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
        for filepath in filepaths:
            with open(filepath, "r") as f:
                data.append(" ".join([line.rstrip() for line in f.readlines()]))

        for id, text in enumerate(data):
            yield id, {"id": id, "text": text}
