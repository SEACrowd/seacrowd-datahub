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
ViCon, comprises pairs of synonyms and antonymys across \
    noun, verb, and adjective classes, offerring data to \
    distinguish between similarity and dissimilarity.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{nguyen-etal-2018-introducing,
    title = "Introducing Two {V}ietnamese Datasets for Evaluating Semantic Models of (Dis-)Similarity and Relatedness",
    author = "Nguyen, Kim Anh  and
      Schulte im Walde, Sabine  and
      Vu, Ngoc Thang",
    editor = "Walker, Marilyn  and
      Ji, Heng  and
      Stent, Amanda",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-2032",
    doi = "10.18653/v1/N18-2032",
    pages = "199--205",
    }
"""

_DATASETNAME = "vicon"

_DESCRIPTION = """\
ViCon, comprises pairs of synonyms and antonymys across \
    noun, verb, and adjective classes, offerring data to \
    distinguish between similarity and dissimilarity.
"""

_HOMEPAGE = "https://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/vnese-sem-datasets/"

_LANGUAGES = ["vie"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_NC_SA_2_0.value  # example: Licenses.MIT.value, Licenses.CC_BY_NC_SA_4_0.value, Licenses.UNLICENSE.value, Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "noun": "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/ViData.zip",
    "adj": "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/ViData.zip",
    "verb": "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/ViData.zip",
}

# This task is more suitable for TEXTUAL_ENTAILMENT
# because the labels (antonym, synonym) roughly correlates to (contradiction, entailment)
_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class ViConDataset(datasets.GeneratorBasedBuilder):
    """
    ViCon, comprises pairs of synonyms and antonymys across \
    noun, verb, and adjective classes, offerring data to \
    distinguish between similarity and dissimilarity.
    """

    POS_TAGS = ["noun", "adj", "verb"]

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [SEACrowdConfig(name=f"{_DATASETNAME}_{POS_TAG}_source", version=_SOURCE_VERSION, description=f"{_DATASETNAME}_{POS_TAG} source schema", schema="source", subset_id=f"{_DATASETNAME}_{POS_TAG}",) for POS_TAG in POS_TAGS] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{POS_TAG}_seacrowd_pairs",
            version=_SEACROWD_VERSION,
            description=f"{_DATASETNAME}_{POS_TAG} SEACrowd schema",
            schema="seacrowd_pairs",
            subset_id=f"{_DATASETNAME}_{POS_TAG}",
        )
        for POS_TAG in POS_TAGS
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_noun_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "Word1": datasets.Value("string"),
                    "Word2": datasets.Value("string"),
                    "Relation": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_pairs":
            features = schemas.pairs_features(["ANT", "SYN"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        POS_TAG = self.config.name.split("_")[1]
        if POS_TAG == "noun" or POS_TAG == "verb":
            number = 400
        elif POS_TAG == "adj":
            number = 600

        if POS_TAG in self.POS_TAGS:
            data_dir = dl_manager.download_and_extract(_URLS[POS_TAG])

        else:
            data_dir = [dl_manager.download_and_extract(_URLS[POS_TAG]) for POS_TAG in self.POS_TAGS]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"ViData/ViCon/{number}_{POS_TAG}_pairs.txt"),
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()

        data = []
        for line in lines:
            columns = line.strip().split("\t")
            data.append(columns)

        df = pd.DataFrame(data[1:], columns=data[0])

        for index, row in df.iterrows():

            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == "seacrowd_pairs":

                example = {
                    "id": str(index),
                    "text_1": str(row["Word1"]),
                    "text_2": str(row["Word2"]),
                    "label": str(row["Relation"]),
                }

            yield index, example
