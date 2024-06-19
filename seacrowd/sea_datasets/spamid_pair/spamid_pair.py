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
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{Chrismanto2022,
title = {SPAMID-PAIR: A Novel Indonesian Post–Comment Pairs Dataset Containing Emoji},
journal = {International Journal of Advanced Computer Science and Applications},
doi = {10.14569/IJACSA.2022.0131110},
url = {http://dx.doi.org/10.14569/IJACSA.2022.0131110},
year = {2022},
publisher = {The Science and Information Organization},
volume = {13},
number = {11},
author = {Antonius Rachmat Chrismanto and Anny Kartika Sari and Yohanes Suyanto}
}
"""

_DATASETNAME = "spamid_pair"


_DESCRIPTION = """\
SPAMID-PAIR is data post-comment pairs collected from 13 selected Indonesian public figures (artists) / public accounts
with more than 15 million followers and categorized as famous artists.
It was collected from Instagram using an online tool and Selenium.
Two persons labeled all pair data as an expert in a total of 72874 data.
The data contains Unicode text (UTF-8) and emojis scrapped in posts and comments without account profile information.
"""

_HOMEPAGE = "https://data.mendeley.com/datasets/fj5pbdf95t/1"

_LANGUAGES = ["ind"]


_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False


_URLS = {
    _DATASETNAME: "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/fj5pbdf95t-1.zip",
}

_SUPPORTED_TASKS = [Tasks.INTENT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class SpamidPairDataset(datasets.GeneratorBasedBuilder):
    """SPAMID-PAIR is data post-comment pairs collected from 13 selected Indonesian public figures (artists) / public accounts with more than 15 million followers and categorized as famous artists."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    LABEL_CLASSES = [1, 0]

    SEACROWD_SCHEMA_NAME = "text"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "igid": datasets.Value("string"),
                    "comment": datasets.Value("string"),
                    "posting": datasets.Value("string"),
                    "spam": datasets.ClassLabel(names=self.LABEL_CLASSES),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(self.LABEL_CLASSES)

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
        data_dir = Path(dl_manager.download_and_extract(urls))
        data_dir = os.path.join(os.path.join(os.path.join(data_dir, "SPAMID-PAIR"), "Raw"), "dataset-raw.xlsx")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = pd.read_excel(filepath)

        if self.config.schema == "source":
            for i, row in data.iterrows():
                yield i, {
                    "igid": str(row["igid"]),
                    "comment": str(row["comment"]),
                    "posting": str(row["posting"]),
                    "spam": row["spam"],
                }

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for i, row in data.iterrows():
                yield i, {
                    "id": str(i),
                    "text": str(row["comment"]) + "\n" + str(row["posting"]),
                    "label": int(row["spam"]),
                }
