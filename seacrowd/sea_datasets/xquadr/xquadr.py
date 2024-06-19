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
from seacrowd.utils.constants import Tasks, Licenses, TASK_TO_SCHEMA, SCHEMA_TO_FEATURES

_CITATION = """\
@article{,@inproceedings{roy-etal-2020-lareqa,
    title = "{LAR}e{QA}: Language-Agnostic Answer Retrieval from a Multilingual Pool",
    author = "Roy, Uma  and
        Constant, Noah  and
        Al-Rfou, Rami  and
        Barua, Aditya  and
        Phillips, Aaron  and
        Yang, Yinfei",
    editor = "Webber, Bonnie  and
        Cohn, Trevor  and
        He, Yulan  and
        Liu, Yang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.477",
    doi = "10.18653/v1/2020.emnlp-main.477",
    pages = "5919--5930",
}
"""

_DATASETNAME = "xquadr"

_DESCRIPTION = """\
XQuAD-R is a retrieval version of the XQuAD dataset (a cross-lingual extractive
QA dataset) that is a part of the LAReQA benchmark. Like XQuAD, XQUAD-R is an
11-way parallel dataset, where each question (out of around 1200) appears in 11
different languages and has 11 parallel correct answers across the languages. It
is designed so as to include parallel QA pairs across languages, allowing
questions to be matched with answers from different languages. The span-tagging
task in XQuAD is converted into a retrieval task by breaking up each contextual
paragraph into sentences, and treating each sentence as a possible target
answer. There are around 1000 candidate answers in each language.
"""

_HOMEPAGE = "https://github.com/google-research-datasets/lareqa"

_LANGUAGES = ["tha", "vie"]

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

_URLS = {
    "tha": "https://github.com/google-research-datasets/lareqa/raw/master/xquad-r/th.json",
    "vie": "https://github.com/google-research-datasets/lareqa/raw/master/xquad-r/vi.json",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING_RETRIEVAL]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # qa

_SOURCE_VERSION = "1.1.0"  # inside the dataset

_SEACROWD_VERSION = "2024.06.20"


class XquadRDataset(datasets.GeneratorBasedBuilder):
    """A retrieval version of the XQuAD dataset (a cross-lingual extractive QA dataset)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in _LANGUAGES:
        BUILDER_CONFIGS += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} {subset} source schema",
                schema="source",
                subset_id=subset,
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset}_{_SEACROWD_SCHEMA}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {subset} SEACrowd schema",
                schema=_SEACROWD_SCHEMA,
                subset_id=subset,
            ),
        ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{_LANGUAGES[0]}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "paragraphs": datasets.Sequence(
                        {
                            "context": datasets.Value("string"),
                            "qas": datasets.Sequence(
                                {
                                    "answers": datasets.Sequence(
                                        {
                                            "answer_start": datasets.Value("int32"),
                                            "text": datasets.Value("string"),
                                        }
                                    ),
                                    "id": datasets.Value("string"),
                                    "question": datasets.Value("string"),
                                }
                            ),
                            "sentence_breaks": datasets.Sequence(
                                datasets.Sequence(datasets.Value("int32"))
                            ),
                            "sentences": datasets.Sequence(datasets.Value("string")),
                        }
                    ),
                    "title": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[
                TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]
            ]  # qa_features
            features["meta"] = {
                "title": datasets.Value("string"),
                "answers_start": datasets.Sequence(datasets.Value("int32")),
                "answers_text": datasets.Sequence(datasets.Value("string")),
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
        url = _URLS[self.config.subset_id]
        data_path = Path(dl_manager.download(url))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_path": data_path,
                },
            ),
        ]

    def _generate_examples(self, data_path: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        key = 0
        for example in data["data"]:

            if self.config.schema == "source":
                yield key, example
                key += 1

            elif self.config.schema == _SEACROWD_SCHEMA:
                for paragraph in example["paragraphs"]:
                    # get sentence breaks (sentences' string stop index)
                    break_list = [breaks[1] for breaks in paragraph["sentence_breaks"]]

                    for qa in paragraph["qas"]:
                        # get answers' string start index
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]

                        # retrieve answers' relevant sentence
                        answers = []
                        for start in answer_starts:
                            for i, end in enumerate(break_list):
                                if start < end:
                                    answers.append(paragraph["sentences"][i])
                                    break

                        yield key, {
                            "id": str(key),
                            "question_id": qa["id"],
                            # "document_id": None,
                            "question": qa["question"],
                            "type": "retrieval",
                            "choices": [],  # escape multiple choice qa seacrowd test
                            "context": paragraph["context"],
                            "answer": answers,
                            "meta": {
                                "title": example["title"],
                                "answers_start": answer_starts,
                                "answers_text": [answer["text"] for answer in qa["answers"]],
                            },
                        }
                        key += 1
