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
This dataset is collected from electronic newspapers published on the web and provided by VLSP organization.\
It consists of approximately 15k sentences, each of which contain NE information in the IOB annotation format\
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{nguyen-et-al-2019-vlsp-ner,
author = {Nguyen, Huyen and Ngo, Quyen and Vu, Luong and Mai, Vu and Nguyen, Hien},
year = {2019},
month = {01},
pages = {283-294},
title = {VLSP Shared Task: Named Entity Recognition},
volume = {34},
journal = {Journal of Computer Science and Cybernetics},
doi = {10.15625/1813-9663/34/4/13161}
}
"""

_DATASETNAME = "vlsp2016_ner"

_DESCRIPTION = """\
This dataset is collected from electronic newspapers published on the web and provided by VLSP organization. \
It consists of approximately 15k sentences, each of which contain NE information in the IOB annotation format
"""

_HOMEPAGE = "https://huggingface.co/datasets/datnth1709/VLSP2016-NER-data"

_LANGUAGES = ["vie"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_NC_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "train": "https://huggingface.co/datasets/datnth1709/VLSP2016-NER-data/resolve/main/data/train-00000-of-00001-b0417886a268b83a.parquet?download=true",
        "test": "https://huggingface.co/datasets/datnth1709/VLSP2016-NER-data/resolve/main/data/valid-00000-of-00001-846411c236133ba3.parquet?download=true",
    },
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class Visp2016NER(datasets.GeneratorBasedBuilder):
    """This dataset is collected from electronic newspapers published on the web and provided by VLSP organization.
    It consists of approximately 15k sentences, each of which contain NE information in the IOB annotation format"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="vlsp2016_ner_source",
            version=SOURCE_VERSION,
            description="vlsp2016_ner source schema",
            schema="source",
            subset_id="vlsp2016_ner",
        ),
        SEACrowdConfig(
            name="vlsp2016_ner_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="vlsp2016_ner SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id="vlsp2016_ner",
        ),
    ]

    DEFAULT_CONFIG_NAME = "vlsp2016_ner_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.Value("int64")),
                }
            )
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label.features([x for x in range(9)])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        train_url = _URLS[_DATASETNAME]["train"]
        train_path = dl_manager.download_and_extract(train_url)

        test_url = _URLS[_DATASETNAME]["test"]
        test_path = dl_manager.download_and_extract(test_url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_path,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        df = pd.read_parquet(filepath)
        if self.config.schema == "source":
            for i in range(len(df)):
                row = df.iloc[i]
                yield (
                    i,
                    {
                        "tokens": row["tokens"],
                        "ner_tags": row["ner_tags"],
                    },
                )
        elif self.config.schema == "seacrowd_seq_label":
            for i in range(len(df)):
                row = df.iloc[i]
                yield (
                    i,
                    {
                        "id": i,
                        "tokens": row["tokens"],
                        "labels": row["ner_tags"],
                    },
                )
