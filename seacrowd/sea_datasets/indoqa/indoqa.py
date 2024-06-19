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
@misc{IndoQA,
  author = {{Jakarta Artificial Intelligence Research}}
  title = {IndoQA: Building Indonesian QA dataset},
  year = {2023}
  url = {https://huggingface.co/datasets/jakartaresearch/indoqa}
}
"""

_DATASETNAME = "indoqa"

_DESCRIPTION = """\
IndoQA is a monolingual question-answering dataset of Indonesian language (ind).
It comprises 4,413 examples with 3:1 split of training and validation sets.
The datasets consists of a context paragraph along with an associated question-answer pair.
"""

_HOMEPAGE = "https://jakartaresearch.com/"
_LICENSE = Licenses.CC_BY_ND_4_0.value

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_URLS = {
    _DATASETNAME: {
        "train": "https://drive.google.com/uc?id=1ND893H5x2gaPRRMJVajQ4hgqpopHoD0u",
        "validation": "https://drive.google.com/uc?id=1mq_foV72riXb1KVBirJzTFZEe7oa8f4f",
    },
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class IndoQADataset(datasets.GeneratorBasedBuilder):
    """IndoQA: A monolingual Indonesian question-answering dataset comprises 4,413 instances of QA-pair with context."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
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
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "category": datasets.Value("string"),
                    "span_start": datasets.Value("int32"),
                    "span_end": datasets.Value("int32"),
                }
            )

        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features
            features["meta"]["span_start"] = datasets.Value("int32")
            features["meta"]["span_end"] = datasets.Value("int32")

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
        data_paths = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_paths["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_paths["validation"]},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, "r", encoding="utf-8") as file:
            datas = json.load(file)

        if self.config.schema == "source":
            for key, data in enumerate(datas):
                yield key, data

        elif self.config.schema == "seacrowd_qa":
            for key, data in enumerate(datas):
                yield key, {
                    "id": f'{data["id"]}',
                    "question_id": data["id"],
                    "document_id": "",
                    "question": data["question"],
                    "type": data["category"],
                    "choices": [],
                    "context": data["context"],
                    "answer": [data["answer"]],
                    "meta": {
                        "span_start": data["span_start"],
                        "span_end": data["span_end"],
                    },
                }
