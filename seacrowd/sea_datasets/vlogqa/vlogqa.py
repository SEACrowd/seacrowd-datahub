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

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@inproceedings{ngo-etal-2024-vlogqa,
    title = "{V}log{QA}: Task, Dataset, and Baseline Models for {V}ietnamese Spoken-Based Machine Reading Comprehension",
    author = "Ngo, Thinh  and
        Dang, Khoa  and
        Luu, Son  and
        Nguyen, Kiet  and
        Nguyen, Ngan",
    editor = "Graham, Yvette  and
        Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.79",
    pages = "1310--1324",
}
"""

_DATASETNAME = "vlogqa"

_DESCRIPTION = """\
VlogQA is a Vietnamese spoken language corpus for machine reading comprehension. It
consists of 10,076 question-answer pairs based on 1,230 transcript documents sourced from
YouTube videos around food and travel.
"""

_HOMEPAGE = "https://github.com/sonlam1102/vlogqa"

_LANGUAGES = ["vie"]

_LICENSE = f"""{Licenses.OTHERS.value} |
The user of VlogQA developed by the NLP@UIT research group must respect the following
terms and conditions:
1. The dataset is only used for non-profit research for natural language processing and
   education.
2. The dataset is not allowed to be used in commercial systems.
3. Do not redistribute the dataset. This dataset may be modified or improved to serve a
   research purpose better, but the edited dataset may not be distributed.
4. Summaries, analyses, and interpretations of the properties of the dataset may be
   derived and published, provided it is not possible to reconstruct the information from
   these summaries.
5. Published research works that use the dataset must cite the following paper:
   Thinh Ngo, Khoa Dang, Son Luu, Kiet Nguyen, and Ngan Nguyen. 2024. VlogQA: Task,
   Dataset, and Baseline Models for Vietnamese Spoken-Based Machine Reading Comprehension.
   In Proceedings of the 18th Conference of the European Chapter of the Association for
   Computational Linguistics (Volume 1: Long Papers), pages 1310–1324, St. Julian’s,
   Malta. Association for Computational Linguistics.
"""

_LOCAL = True  # need to signed a user agreement, see _HOMEPAGE

_URLS = {}  # local dataset

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # qa

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class VlogQADataset(datasets.GeneratorBasedBuilder):
    """Vietnamese spoken language corpus around food and travel for machine reading comprehension"""

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
            name=f"{_DATASETNAME}_{_SEACROWD_SCHEMA}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=_SEACROWD_SCHEMA,
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]]  # qa_features
            features["meta"] = {
                "answers_start": datasets.Sequence(datasets.Value("int32")),
            }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the `data_dir` kwarg (where the .json is located) to load_dataset.")
        else:
            data_dir = Path(self.config.data_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "file_path": data_dir / "train.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "file_path": data_dir / "dev.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "file_path": data_dir / "test.json",
                },
            ),
        ]

    def _generate_examples(self, file_path: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        key = 0
        for example in data["data"]:

            if self.config.schema == "source":
                for paragraph in example["paragraphs"]:
                    for qa in paragraph["qas"]:
                        yield key, {
                            "id": qa["id"],
                            "title": example["title"],
                            "context": paragraph["context"],
                            "question": qa["question"],
                            "answers": qa["answers"],
                        }
                        key += 1

            elif self.config.schema == _SEACROWD_SCHEMA:
                for paragraph in example["paragraphs"]:
                    for qa in paragraph["qas"]:
                        yield key, {
                            "id": str(key),
                            "question_id": qa["id"],
                            "document_id": example["title"],
                            "question": qa["question"],
                            "type": None,
                            "choices": [],  # escape multiple_choice qa seacrowd test, can't be None
                            "context": paragraph["context"],
                            "answer": [answer["text"] for answer in qa["answers"]],
                            "meta": {
                                "answers_start": [answer["answer_start"] for answer in qa["answers"]],
                            },
                        }
                        key += 1
