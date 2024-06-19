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
@inproceedings{fujita2021empirical,
  title={An Empirical Investigation of Online News Classification on an Open-Domain, Large-Scale and High-Quality Dataset in Vietnamese},
  author={Fujita, H and Perez-Meana, H},
  booktitle={New Trends in Intelligent Software Methodologies, Tools and Techniques: Proceedings of the 20th International Conference on New Trends in Intelligent Software Methodologies, Tools and Techniques (SoMeT_21)},
  volume={337},
  pages={367},
  year={2021},
  organization={IOS Press}
}
"""

_DATASETNAME = "uit_vion"


_DESCRIPTION = """\
UIT-ViON (Vietnamese Online Newspaper) is a dataset collected from well-known online newspapers in Vietnamese.
The UIT-ViON is an open-domain, large-scale, and high-quality dataset consisting of 260,000 textual data
points annotated with 13 different categories for evaluating Vietnamese short text classification.
The dataset is split into training, validation, and test sets, each containing 208000, 26000,
and 26000 pieces of text, respectively.
"""

_HOMEPAGE = "https://github.com/kh4nh12/UIT-ViON-Dataset"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://github.com/kh4nh12/UIT-ViON-Dataset/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [Tasks.TOPIC_MODELING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class UitVion(datasets.GeneratorBasedBuilder):
    """UIT-ViON (Vietnamese Online Newspaper) is a dataset collected from well-known online newspapers in Vietnamese."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    LABEL_CLASSES = [i for i in range(13)]

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
                    "title": datasets.Value("string"),
                    "link": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=self.LABEL_CLASSES),
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
        data_dir = dl_manager.download_and_extract(urls)
        file_dir = os.path.join("UIT-ViON-Dataset-main", "data.zip")
        data_dir = os.path.join(data_dir, file_dir)
        data_dir = dl_manager.download_and_extract(data_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "UIT-ViON_train.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "UIT-ViON_test.csv"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "UIT-ViON_dev.csv"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = pd.read_csv(filepath)

        if self.config.schema == "source":
            for i, row in data.iterrows():
                yield i, {
                    "title": str(row["title"]),
                    "link": str(row["link"]),
                    "label": row["label"],
                }

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for i, row in data.iterrows():
                yield i, {
                    "id": str(i),
                    "text": str(row["title"]),
                    "label": int(row["label"]),
                }
