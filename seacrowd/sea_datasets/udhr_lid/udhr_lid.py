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

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = r"""\
@inproceedings{
  kargaran2023glotlid,
  title={{GlotLID}: Language Identification for Low-Resource Languages},
  author={Kargaran, Amir Hossein and Imani, Ayyoob and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
  booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023},
  url={https://openreview.net/forum?id=dl4e3EBz5j}
}
"""

_LANGUAGES = [
    "sun",
    "ace",
    "mad",
    "lao",
    "cfm",
    "hnj",
    "min",
    "zlm",
    "tha",
    "blt",
    "hni",
    "jav",
    "tdt",
    "cnh",
    "khm",
    "ban",
    "ind",
    "mya",
    "ccp",
    "duu",
    "tet",
    "kkh",
    "bug",
    "vie",
]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "udhr_lid"

_DESCRIPTION = """\
The UDHR-LID dataset is a refined version of the Universal Declaration of Human Rights, tailored for language identification tasks.
It removes filler texts, repeated phrases, and inaccuracies from the original UDHR, focusing only on cleaned paragraphs.
Each entry in the dataset is associated with a specific language, providing long, linguistically rich content.
This dataset is particularly useful for non-parallel, language-specific text analysis in natural language processing.
"""

_HOMEPAGE = "https://huggingface.co/datasets/cis-lmu/udhr-lid"

_LICENSE = Licenses.CC0_1_0.value

_URL = "https://huggingface.co/datasets/cis-lmu/udhr-lid/raw/main/udhr-lid.csv"

_SUPPORTED_TASKS = [Tasks.LANGUAGE_IDENTIFICATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class UDHRLID(datasets.GeneratorBasedBuilder):
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
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd Schema",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "iso639-3": datasets.Value("string"),
                    "iso15924": datasets.Value("string"),
                    "language": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(_LANGUAGES)

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_path = dl_manager.download(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_path,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:

        datas = pd.read_csv(filepath)

        for i, row in datas.iterrows():
            if row["iso639-3"] in _LANGUAGES:
                if self.config.schema == "source":
                    yield i, {"id": str(i), "sentence": row["sentence"], "iso639-3": row["iso639-3"], "iso15924": row["iso15924"], "language": row["language"]}
                elif self.config.schema == "seacrowd_text":
                    yield i, {"id": str(i), "text": row["sentence"], "label": row["iso639-3"]}
                else:
                    raise ValueError(f"Invalid config: {self.config.name}")
