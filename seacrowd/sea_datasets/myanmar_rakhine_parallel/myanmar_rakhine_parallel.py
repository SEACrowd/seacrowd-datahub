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

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{myint-oo-etal-2019-neural,
    title = "Neural Machine Translation between {M}yanmar ({B}urmese) and {R}akhine ({A}rakanese)",
    author = "Myint Oo, Thazin and
        Kyaw Thu, Ye and
        Mar Soe, Khin",
    editor = {Zampieri, Marcos and
        Nakov, Preslav and
        Malmasi, Shervin and
        Ljube{\v{s}}i{\'c}, Nikola and
        Tiedemann, J{\"o}rg and
        Ali, Ahmed},
    booktitle = "Proceedings of the Sixth Workshop on {NLP} for Similar Languages, Varieties and Dialects",
    month = jun,
    year = "2019",
    address = "Ann Arbor, Michigan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-1408",
    doi = "10.18653/v1/W19-1408",
    pages = "80--88",
}
"""

_DATASETNAME = "myanmar_rakhine_parallel"
_DESCRIPTION = """\
The data contains 18,373 Myanmar sentences of the ASEAN-MT Parallel Corpus,
which is a parallel corpus in the travel domain. It contains six main
categories: people (greeting, introduction, and communication), survival
(transportation, accommodation, and finance), food (food, beverages, and
restaurants), fun (recreation, traveling, shopping, and nightlife), resource
(number, time, and accuracy), special needs (emergency and health). Manual
translation into the Rakhine language was done by native Rakhine students from
two Myanmar universities, and the translated corpus was checked by the editor
of a Rakhine newspaper. Word segmentation for Rakhine was done manually, and
there are exactly 123,018 words in total.
"""

_HOMEPAGE = "https://github.com/ye-kyaw-thu/myPar/tree/master/my-rk"
_LANGUAGES = ["mya", "rki"]
_LICENSE = Licenses.GPL_3_0.value
_LOCAL = False
_URLS = {
    "train_mya": "https://raw.githubusercontent.com/ye-kyaw-thu/myPar/master/my-rk/ver-0.1/train.my",
    "dev_mya": "https://raw.githubusercontent.com/ye-kyaw-thu/myPar/master/my-rk/ver-0.1/dev.my",
    "test_mya": "https://raw.githubusercontent.com/ye-kyaw-thu/myPar/master/my-rk/ver-0.1/test.my",
    "train_rki": "https://raw.githubusercontent.com/ye-kyaw-thu/myPar/master/my-rk/ver-0.1/train.rk",
    "dev_rki": "https://raw.githubusercontent.com/ye-kyaw-thu/myPar/master/my-rk/ver-0.1/dev.rk",
    "test_rki": "https://raw.githubusercontent.com/ye-kyaw-thu/myPar/master/my-rk/ver-0.1/test.rk",
}
_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "0.1.0"
_SEACROWD_VERSION = "2024.06.20"


class MyanmarRakhineParallel(datasets.GeneratorBasedBuilder):
    """Myanmar-Rakhine Parallel dataset from https://github.com/ye-kyaw-thu/myPar/tree/master/my-rk"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "t2t"

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

        if self.config.schema == "source" or self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text2text_features
        else:
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        data_paths = {
            "train_mya": Path(dl_manager.download_and_extract(_URLS["train_mya"])),
            "dev_mya": Path(dl_manager.download_and_extract(_URLS["dev_mya"])),
            "test_mya": Path(dl_manager.download_and_extract(_URLS["test_mya"])),
            "train_rki": Path(dl_manager.download_and_extract(_URLS["train_rki"])),
            "dev_rki": Path(dl_manager.download_and_extract(_URLS["dev_rki"])),
            "test_rki": Path(dl_manager.download_and_extract(_URLS["test_rki"])),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "mya_filepath": data_paths["train_mya"],
                    "rki_filepath": data_paths["train_rki"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "mya_filepath": data_paths["test_mya"],
                    "rki_filepath": data_paths["test_rki"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "mya_filepath": data_paths["dev_mya"],
                    "rki_filepath": data_paths["dev_rki"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, mya_filepath: Path, rki_filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        # read mya file
        with open(mya_filepath, "r", encoding="utf-8") as mya_file:
            mya_data = mya_file.readlines()
        mya_data = [s.strip("\n") for s in mya_data]

        # read rki file
        with open(rki_filepath, "r", encoding="utf-8") as rki_file:
            rki_data = rki_file.readlines()
        rki_data = [s.strip("\n") for s in rki_data]

        num_sample = len(mya_data)

        for i in range(num_sample):
            if self.config.schema == "source" or self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                example = {"id": str(i), "text_1": mya_data[i], "text_2": rki_data[i], "text_1_name": "mya", "text_2_name": "rki"}
            yield i, example
