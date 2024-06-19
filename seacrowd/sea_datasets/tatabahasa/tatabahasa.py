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
This test is a general test for Malay grammar. Contains 349 questions that may be reinforced with instructions.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses, TASK_TO_SCHEMA

_CITATION = None

_DATASETNAME = "tatabahasa"

_DESCRIPTION = """\
This test is a general test for Malay grammar. Contains 349 questions.
"""

_HOMEPAGE = "https://github.com/mesolitica/malaysian-dataset/tree/master/llm-benchmark/tatabahasabm.tripod.com"

_LANGUAGES = ["zlm"]  

_LICENSE = Licenses.UNLICENSE.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/mesolitica/malaysian-dataset/master/llm-benchmark/tatabahasabm.tripod.com/quiz-tatabahasa.jsonl",
}

_SUPPORTED_TASKS = [Tasks.COMMONSENSE_REASONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class TatabahasaDataset(datasets.GeneratorBasedBuilder):
    """This test is a general test for Malay grammar. Contains 349 questions."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA}",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            multi_choice = {"text" : datasets.Value("string"), "answer": datasets.Value("bool")}
            features = datasets.Features({
                "question" : datasets.Value("string"),
                "instruction": datasets.Value("string"),
                "choices": {
                    "A": multi_choice,
                    "B": multi_choice,
                    "C": multi_choice,
                    "D": multi_choice,
                },
                "website": datasets.Value("string")
            })

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA}":
            features = schemas.qa_features
            features["meta"] = {"website": datasets.Value("string")}

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        with open(filepath ,'r') as f:
            data = [json.loads(line) for line in f]

        if self.config.schema == "source":
            for i in range(len(data)):
                out = {
                    "question": data[i]["question"],
                    "instruction": data[i]["instruction"],
                    "choices": data[i]["choices"],
                    "website": data[i]["website"]
                }
                yield i, out

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA}":
            for i in range(len(data)):
                out = {
                    "id": i + 1,
                    "question_id": None,
                    "document_id": None,
                    "question": data[i]["question"],
                    "type": "multiple_choice",
                    "choices": [
                        data[i]["choices"]["A"]["text"],
                        data[i]["choices"]["B"]["text"],
                        data[i]["choices"]["C"]["text"],
                        data[i]["choices"]["D"]["text"],
                    ],
                    "context": data[i]["instruction"],
                    "answer": [choice["text"] for choice in data[i]["choices"].values() if choice["answer"]],
                    "meta": {"website": data[i]["website"]},
                }
                yield i, out
