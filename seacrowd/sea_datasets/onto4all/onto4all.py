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
Onto4All is a subsample of other open source performant conversational datasets. We start with a carefully curated subset of the OpenHermes-2.5-Viet dataset, co-created by @qnguyen3 and @Teknium. This dataset is specifically designed to support the training and evaluation of Multilingual language models, such as Vistral-7B-chat and VinaLlama-7B-chat, and is derived from our Supervised Fine-Tuning (SFT) data. We have included Vietnamese here, but will add more languages.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd
import json

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@article{Onto4All2024,
  title={Onto4All: Enhancing Multilingual Conversational AI},
  author={Nguyen, Q., },
  journal={GitHub repository},
  year={2024},
  publisher={HuggingFace Datasets}
}
"""

_DATASETNAME = "onto4all"

_DESCRIPTION = """\
Onto4All is a subsample of other open source performant conversational datasets. We start with a carefully curated subset of the OpenHermes-2.5-Viet dataset, co-created by @qnguyen3 and @Teknium. This dataset is specifically designed to support the training and evaluation of Multilingual language models, such as Vistral-7B-chat and VinaLlama-7B-chat, and is derived from our Supervised Fine-Tuning (SFT) data. We have included Vietnamese here, but will add more languages.
"""

_HOMEPAGE = "https://huggingface.co/datasets/ontocord/onto4all"

_LANGUAGES = ["vie"] 

_LICENSE = Licenses.CC0_1_0.value

_LOCAL = False

_URLS = "https://huggingface.co/datasets/ontocord/onto4all/resolve/main/data/train-00000-of-00001.parquet?download=true"

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

class Onto4AllDataset(datasets.GeneratorBasedBuilder):
    """Onto4All is a subsample of other open source performant conversational datasets. We start with a carefully curated subset of the OpenHermes-2.5-Viet dataset, co-created by @qnguyen3 and @Teknium. This dataset is specifically designed to support the training and evaluation of Multilingual language models, such as Vistral-7B-chat and VinaLlama-7B-chat, and is derived from our Supervised Fine-Tuning (SFT) data. We have included Vietnamese here, but will add more languages."""

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
                    "id": datasets.Value("int32"),
                    "type": datasets.Value("string"),
                    "conversation": datasets.Sequence({
                        "from": datasets.Value("string"),
                        "value": datasets.Value("string"),
                        "weight": datasets.Value("int32"),
                    })
                }
            )

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
        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_parquet(filepath)

        if self.config.schema == "source":
            for i, row in df.iterrows():
                conversation = [{
                        "from": item["from"],
                        "value": item["value"],
                        "weight": item["weight"],
                    } for item in row["conversation"]
                ]

                yield i, {
                    "id": row["id"],
                    "type": row["type"],
                    "conversation": conversation,
                }
                break

        elif self.config.schema == "seacrowd_qa":
            for i, row in df.iterrows():
                context = ""
                question = ""
                answer = ""

                for item in row["conversation"]:
                    if item["from"] == "system":
                        context = item["value"]
                    elif item["from"] == "human":
                        question = item["value"]
                    elif item["from"] == "gpt":
                        answer = item["value"]

                yield i, {
                    "id": row["id"],
                    "question_id": row["id"],
                    "document_id": "",
                    "question": question,
                    "type": row["type"],
                    "choices": [],
                    "context": context,
                    "answer": [answer],
                    "meta": {},
                }
