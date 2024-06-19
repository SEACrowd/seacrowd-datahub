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

_CITATION = """\
@article{Tuhenay2021,
  title = {Perbandingan Klasifikasi Bahasa Menggunakan Metode Naïve Bayes Classifier (NBC) Dan Support Vector Machine (SVM)},
  volume = {4},
  ISSN = {2656-1948},
  url = {http://dx.doi.org/10.33387/jiko.v4i2.2958},
  DOI = {10.33387/jiko.v4i2.2958},
  number = {2},
  journal = {JIKO (Jurnal Informatika dan Komputer)},
  publisher = {LPPM Universitas Khairun},
  author = {Tuhenay,  Deglorians},
  year = {2021},
  month = aug,
  pages = {105-111}
}
"""

_DATASETNAME = "identifikasi_bahasa"

_DESCRIPTION = """\
The identifikasi-bahasa dataset includes text samples in Indonesian, Ambonese, and Javanese. \
Each entry is comprised of cleantext, representing the sentence content, and a label identifying the language. \
The manual input process involved grouping the data by language categories, \
with labels for language identification and cleantext representing sentence content. The dataset, excluding punctuation and numbers, \
consists of a minimum of 3,000 Ambonese, 10,000 Javanese, \
and 3,500 Indonesian language entries, meeting the research's minimum standard for effective language identification.
"""

_HOMEPAGE = "https://github.com/joanitolopo/identifikasi-bahasa"
_LANGUAGES = ["ind", "jav", "abs"]

_LICENSE = Licenses.APACHE_2_0.value
_LOCAL = False

_URLS = {
    _DATASETNAME: "https://github.com/joanitolopo/identifikasi-bahasa/raw/main/DataKlasifikasi.xlsx",
}

_SUPPORTED_TASKS = [Tasks.LANGUAGE_IDENTIFICATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"
_TAGS = ["Ambon", "Indo", "Jawa"]


class IdentifikasiBahasaDataset(datasets.GeneratorBasedBuilder):
    """The "identifikasi-bahasa" dataset, manually grouped by language, \
    contains labeled Indonesian, Ambonese, and Javanese text entries, excluding \
    punctuation and numbers, with a minimum of 3,000 Ambonese, 10,000 Javanese, \
    and 3,500 Indonesian entries for effective language identification."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
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
            features = datasets.Features({"cleanText": datasets.Value("string"), "label": datasets.Value("string")})
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(_TAGS)

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
        data_dir = dl_manager.download(urls)
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
        dataset = pd.read_excel(filepath)

        if self.config.schema == "source":
            for i, row in dataset.iterrows():
                yield i, {"cleanText": row["cleanText"], "label": row["label"]}

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for i, row in dataset.iterrows():
                yield i, {"id": i, "text": row["cleanText"], "label": row["label"]}
