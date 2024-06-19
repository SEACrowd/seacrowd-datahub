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

"""
The Thai Romanization dataset contains 648,241 Thai words \
    that were transliterated into English, making Thai \
    pronounciation easier for non-native Thai speakers. \
    This is a valuable dataset for Thai language learners \
    and researchers working on Thai language processing task. \
    Each word in the Thai Romanization dataset is paired with \
    its English phonetic representation, enabling accurate \
    pronunciation guidance. This facilitates the learning and \
    practice of Thai pronunciation for individuals who may not \
    be familiar with the Thai script. The dataset aids in improving \
    the accessibility and usability of Thai language resources, \
    supporting applications such as speech recognition, text-to-speech \
    synthesis, and machine translation. It enables the development of \
    Thai language tools that can benefit Thai learners, tourists, \
    and those interested in Thai culture and language.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

# There are no citation available for this dataset.
_CITATION = ""

_DATASETNAME = "thai_romanization"

_DESCRIPTION = """
The Thai Romanization dataset contains 648,241 Thai words \
    that were transliterated into English, making Thai \
    pronounciation easier for non-native Thai speakers. \
    This is a valuable dataset for Thai language learners \
    and researchers working on Thai language processing task. \
    Each word in the Thai Romanization dataset is paired with \
    its English phonetic representation, enabling accurate \
    pronunciation guidance. This facilitates the learning and \
    practice of Thai pronunciation for individuals who may not \
    be familiar with the Thai script. The dataset aids in improving \
    the accessibility and usability of Thai language resources, \
    supporting applications such as speech recognition, text-to-speech \
    synthesis, and machine translation. It enables the development of \
    Thai language tools that can benefit Thai learners, tourists, \
    and those interested in Thai culture and language.
"""

_HOMEPAGE = "https://www.kaggle.com/datasets/wannaphong/thai-romanization/data"

_LANGUAGES = ["tha"]

_LICENSE = Licenses.CC_BY_SA_3_0.value

_LOCAL = False

_URLS = {_DATASETNAME: "https://raw.githubusercontent.com/wannaphong/thai-romanization/master/dataset/data.csv"}

_SUPPORTED_TASKS = [Tasks.TRANSLITERATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class ThaiRomanizationDataset(datasets.GeneratorBasedBuilder):
    """
    Thai Romanization dataloader from Kaggle (Phong et al., 2018)
    """

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
            features = datasets.Features({"word": datasets.Value("string"), "romanization": datasets.Value("string")})

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
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir),
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = pd.read_csv(filepath, delimiter="	")
        df.columns = ["word", "romanization"]

        for index, row in df.iterrows():

            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                example = {
                    "id": str(index),
                    "text_1": str(row["word"]),
                    "text_2": str(row["romanization"]),
                    "text_1_name": "word",
                    "text_2_name": "romanization",
                }

            yield index, example
