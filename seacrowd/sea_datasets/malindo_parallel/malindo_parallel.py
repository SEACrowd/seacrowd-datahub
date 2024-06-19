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
This template serves as a starting point for contributing a dataset to the SEACrowd Datahub repo.


Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (DEFAULT_SEACROWD_VIEW_NAME,
                                      DEFAULT_SOURCE_VIEW_NAME, Tasks)

_CITATION = """\
@misc{MALINDO-parallel,
  title = "MALINDO-parallel",
  howpublished = "https://github.com/matbahasa/MALINDO_Parallel/blob/master/README.md",
  note = "Accessed: 2023-01-27",
}
"""

_DATASETNAME = "malindo_parallel"


_DESCRIPTION = """\
Teks ini adalah skrip video untuk Kampus Terbuka Universiti Bahasa Asing Tokyo pada tahun 2020. Tersedia parallel sentences dalam Bahasa Melayu/Indonesia dan Bahasa Jepang
"""


_HOMEPAGE = "https://github.com/matbahasa/MALINDO_Parallel/tree/master/OpenCampusTUFS"


_LANGUAGES = ["zlm", "jpn"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)


_LICENSE = "Creative Commons Attribution 4.0 (cc-by-4.0)" 


_LOCAL = False


_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/matbahasa/MALINDO_Parallel/master/OpenCampusTUFS/OCTUFS2020.txt",
}


_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]


_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"



class MalindoParallelDataset(datasets.GeneratorBasedBuilder):
    """Data terjemahan bahasa Melayu/Indonesia"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)


    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="malindo_parallel_source",
            version=SOURCE_VERSION,
            description="malindo_parallel source schema",
            schema="source",
            subset_id="malindo_parallel",
        ),
        SEACrowdConfig(
            name="malindo_parallel_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description="malindo_parallel SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id="malindo_parallel",
        ),
    ]

    DEFAULT_CONFIG_NAME = "malindo_parallel_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "text": datasets.Value("string")})

        elif self.config.schema == "seacrowd_t2t":
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
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:

        rows = []
        temp_cols = None
        with open(filepath) as file:
            while line := file.readline():
                if temp_cols is None:
                    cols = []
                    for col in line.split('\t'):
                        if len(col.strip('\n'))>0:
                            cols.append(col)
                    if len(cols) > 2:
                        correct_line = line.rstrip()
                        rows.append(correct_line)
                    else:
                        temp_cols = cols
                else:
                    temp_cols.append(line)
                    correct_line = "\t".join(temp_cols).rstrip()
                    temp_cols = None
                    rows.append(correct_line)

        if self.config.schema == "source":

            for i, row in enumerate(rows):
                t1idx = row.find("\t") + 1
                t2idx = row[t1idx:].find("\t")
                row_id = row[:t1idx]
                row_melayu = row[t1idx : t1idx + t2idx]
                row_japanese = row[t1idx + t2idx + 1 : -1]
                ex = {"id": row_id.rstrip(),
                      "text": row_melayu + "\t" + row_japanese}
                yield i, ex

        elif self.config.schema == "seacrowd_t2t":

            for i, row in enumerate(rows):
                t1idx = row.find("\t") + 1
                t2idx = row[t1idx:].find("\t")
                row_id = row[:t1idx]
                row_melayu = row[t1idx : t1idx + t2idx]
                row_japanese = row[t1idx + t2idx + 1 : -1]
                ex = {
                    "id": row_id.rstrip(),
                    "text_1": row_melayu,
                    "text_2": row_japanese,
                    "text_1_name": "zlm",
                    "text_2_name": "jpn",
                }
                yield i, ex
