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


import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{rizqullah2023qasina,
title={QASiNa: Religious Domain Question Answering using Sirah Nabawiyah},
author={Muhammad Razif Rizqullah and Ayu Purwarianti and Alham Fikri Aji},
year={2023},
eprint={2310.08102},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
"""

_DATASETNAME = "qasina"

_DESCRIPTION = """\
Question Answering Sirah Nabawiyah Dataset (QASiNa) is Extractive \
QA Dataset which build to perform QA task in Sirah Nabawiyah domain.
"""

_HOMEPAGE = "https://github.com/rizquuula/QASiNa"

_LANGUAGES = ["ind"]

_LICENSE = Licenses.MIT.value
_LOCAL = False

_URLS = {
    _DATASETNAME: "https://github.com/rizquuula/QASiNa/raw/main/QASiNa.json",
}


_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class QasinaDataset(datasets.GeneratorBasedBuilder):
    """Question Answering Sirah Nabawiyah Dataset (QASiNa) is \
    Extractive QA Dataset which build to perform QA task in Sirah Nabawiyah domain."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "qa"

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
            features = datasets.Features(
                {
                    "context_id": datasets.Value("int32"),
                    "context": datasets.Value("string"),
                    "question_answers": datasets.Sequence({"type": datasets.Value("string"), "question": datasets.Value("string"), "answer": datasets.Value("string"), "answer_start": datasets.Value("int32"), "question_id": datasets.Value("int32")}),
                    "context_length": datasets.Value("int32"),
                    "context_title": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.qa.features
            features["meta"] = {"context_title": datasets.Value("string"), "answer_start": datasets.Value("int32"),"context_length": datasets.Value("int32"), "type": datasets.Value("string")}

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        filepath = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        with open(filepath) as file:
            dataset = json.load(file)

        if self.config.schema == "source":
            for i, line in enumerate(dataset):
                yield i, {
                    "context_id": line["context_id"],
                    "context": line["context"],
                    "question_answers": [
                        {
                            "type": subline["type"],
                            "question": subline["question"],
                            "answer": subline["answer"],
                            "answer_start": subline["answer_start"],
                            "question_id": subline["question_id"],
                        }
                        for subline in line["question_answers"]
                    ],
                    "context_length": line["context_length"],
                    "context_title": line["context_title"],
                }

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for line in dataset:
                for question_answer in line["question_answers"]:
                    id = question_answer["question_id"]

                    yield id, {
                        "id": id,
                        "question_id": question_answer["question_id"],
                        "document_id": line["context_id"],
                        "question": question_answer["question"],
                        "type": "extractive",
                        "choices": [],
                        "context": line["context"],
                        "answer": [question_answer["answer"]],
                        "meta": {
                            "context_title": line["context_title"],
                            "answer_start": question_answer["answer_start"],
                            "context_length": line["context_length"],
                            "type": question_answer["type"],
                        },
                    }
