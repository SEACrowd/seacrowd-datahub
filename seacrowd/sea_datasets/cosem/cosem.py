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
import re
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@article{gonzales_corpus_2021,
    title = {The {Corpus} of {Singapore} {English} {Messages} ({CoSEM})},
    issn = {0883-2919, 1467-971X},
    url = {https://onlinelibrary.wiley.com/doi/10.1111/weng.12534},
    doi = {10.1111/weng.12534},
    language = {en},
    urldate = {2022-02-19},
    journal = {World Englishes},
    author = {Gonzales, Wilkinson Daniel Wong and Hiramoto, Mie and R. E. Leimgruber, Jakob and Lim, Jun Jie},
    month = feb,
    year = {2021},
}
"""

_DATASETNAME = "cosem"

_DESCRIPTION = """\
The CoSEM dataset consists of over 900,000 lines of online messages from the messaging platform WhatsApp collected from personal chat
logs of students enrolled in an advanced sociolinguistics class from the National University of Singapore. Messages collected were
from 2016 to 2019. The dataset is in .txt format, where each line of utterance is tagged with a unique identifier that includes its
metadata such as line number, year message was sent, and age and nationality of sender.
"""

_HOMEPAGE = "https://github.com/wdwgonzales/CoSEM/blob/main/Corpus/COSEM_v4_publicrelease_SEP172023.zip"

_LANGUAGES = ["eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC0_1_0.value

_LOCAL = False

_URLS = {_DATASETNAME: "https://github.com/wdwgonzales/CoSEM/raw/main/Corpus/COSEM_v4_publicrelease_SEP172023.zip"}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class CoSEMDataset(datasets.GeneratorBasedBuilder):
    """The CoSEM dataset consists of over 900,000 lines of online messages from the messaging platform WhatsApp collected from
    personal chat logs of students enrolled in an advanced sociolinguistics class from the National University of Singapore."""

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
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SELF_SUPERVISED_PRETRAINING]).lower()}":
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

        split_generators = []

        path = dl_manager.download_and_extract(_URLS[_DATASETNAME])

        split_generators.append(
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(path, "COSEM_v4_publicrelease_SEP172023"),
                },
            )
        )

        return split_generators

    def _generate_examples(self, path: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        files = os.listdir(path)
        file_paths = [os.path.join(path, file) for file in files]
        pattern = r"<(COSEM:.*?)>(.*?)(?=<COSEM:|$)"

        s = {}

        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    key = match[0].strip()
                    value = match[1].strip()

                    if key in s:
                        continue
                    s[key] = value

                    if self.config.schema == "source" or self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SELF_SUPERVISED_PRETRAINING]).lower()}":
                        yield key, {"id": key, "text": value}

                    else:
                        raise ValueError(f"Invalid config: {self.config.name}")
