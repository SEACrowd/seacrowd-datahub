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
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{romadhona-etal-2022-brcc,
  author    = {Romadhona, Nanda Putri and Lu, Sin-En and Lu, Bo-Han and Tsai, Richard Tzong-Han},
  title     = {BRCC and SentiBahasaRojak: The First Bahasa Rojak Corpus for Pretraining and Sentiment Analysis Dataset},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
  publisher = {International Committee on Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.coling-1.389/},
  pages     = {4418--4428},
}
"""

_LOCAL = False
_LANGUAGES = ["zlm", "eng", "cmn"]
_DATASETNAME = "brcc"
_DESCRIPTION = """
The Bahasa Rojak Crawled Corpus (BRCC) is a code-mixed dataset for the Bahasa Rojak dialect in Malaysia.
Passages are generated through data augmentation from English and Malay Wikipedia pages using a modified CoSDA-ML method.
The quality of generated passages is evaluated by two native Malay speakers.
"""
_HOMEPAGE = "https://data.depositar.io/dataset/brcc_and_sentibahasarojak"
_LICENSE = Licenses.UNKNOWN.value
_URL = "https://data.depositar.io/dataset/304d1572-27d6-4549-8292-b1c8f5e9c086/resource/8a558f64-98ff-4922-a751-0ce2ce8447bd/download/BahasaRojak_Datasets.zip"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class BRCCDataset(datasets.GeneratorBasedBuilder):
    """Dataset of Bahasa Rojak passages generated from English and Malay Wikipedia pages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_ssp",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd ssp schema",
            schema="seacrowd_ssp",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        # Source schema = SeaCrowd schema because file only contains lines of text
        if self.config.schema in ("source", "seacrowd_ssp"):
            features = schemas.ssp_features
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "BahasaRojak Datasets", "BRCC", "mix.train"),
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding="utf-8") as f:
           for idx, line in enumerate(f):
            example = {"id": str(idx), "text": line.strip()}
            yield idx, example
