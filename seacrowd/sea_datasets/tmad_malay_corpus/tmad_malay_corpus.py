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
The Towards Malay Abbreviation Disambiguation (TMAD) Malay Corpus includes sentences from Malay news sites with abbreviations and their meanings. Only abbreviations with more than one possible meaning are included.
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{article,
author = {Ciosici, Manuel and Sommer, Tobias},
year = {2019},
month = {04},
pages = {},
title = {Unsupervised Abbreviation Disambiguation Contextual disambiguation using word embeddings}
}
"""

_DATASETNAME = "tmad_malay_corpus"

_DESCRIPTION = """\
The Towards Malay Abbreviation Disambiguation (TMAD) Malay Corpus includes sentences from Malay news sites with abbreviations and their meanings. Only abbreviations with more than one possible meaning are included.
"""

_HOMEPAGE = "https://github.com/bhysss/TMAD-CUM/tree/master"

_LANGUAGES = ["zlm"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "train": "https://raw.githubusercontent.com/bhysss/TMAD-CUM/master/data/Malay/data_train.csv",
    "dev": "https://raw.githubusercontent.com/bhysss/TMAD-CUM/master/data/Malay/data_dev.csv",
    "test": "https://raw.githubusercontent.com/bhysss/TMAD-CUM/master/data/Malay/data_test.csv",
    "dict": "https://raw.githubusercontent.com/bhysss/TMAD-CUM/master/data/Malay/May_dic.json",
}
_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class TMADMalayCorpusDataset(datasets.GeneratorBasedBuilder):
    """Abbreviation disambiguation dataset from Malay news sites."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_qa",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_qa",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"abbr": datasets.Value("string"), "definition": datasets.Value("string"), "sentence": datasets.Value("string"), "choices": datasets.Sequence(datasets.Value("string"))})

        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        data_dirs = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_dirs["train"], "dictpath": data_dirs["dict"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dirs["test"], "dictpath": data_dirs["dict"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_dirs["dev"], "dictpath": data_dirs["dict"]},
            ),
        ]

    def _generate_examples(self, filepath: Path, dictpath: Path) -> Tuple[int, Dict]:

        with open(dictpath) as f:
            may_dict = json.load(f)

        if self.config.schema == "source":
            with open(filepath, encoding="utf-8") as f:
                for row_idx, row in enumerate(csv.DictReader(f)):
                    yield row_idx, {"abbr": row["Abbr"], "definition": row["Definition"], "sentence": row["Sentence"], "choices": may_dict[row["Abbr"]]}

        elif self.config.schema == "seacrowd_qa":
            with open(filepath, encoding="utf-8") as f:
                for row_idx, row in enumerate(csv.DictReader(f)):
                    yield row_idx, {"id": row_idx, "question_id": 0, "document_id": 0, "question": row["Abbr"], "type": "multiple_choice", "choices": may_dict[row["Abbr"]], "context": row["Sentence"], "answer": [row["Definition"]], "meta": {}}
