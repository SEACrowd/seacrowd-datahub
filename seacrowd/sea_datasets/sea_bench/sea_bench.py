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
@article{damonlpsg2023seallm,
  author = {Xuan-Phi Nguyen*, Wenxuan Zhang*, Xin Li*, Mahani Aljunied*,
            Qingyu Tan, Liying Cheng, Guanzheng Chen, Yue Deng, Sen Yang,
            Chaoqun Liu, Hang Zhang, Lidong Bing},
  title = {SeaLLMs - Large Language Models for Southeast Asia},
  year = 2023,
  Eprint = {arXiv:2312.00738},
  url = {https://arxiv.org/pdf/2312.00738.pdf},
}
"""

_DATASETNAME = "sea_bench"

_DESCRIPTION = """\
Sea-bench is a multilingual benchmark for assistant-style models annotated by native linguists
covering 8 Southeast Asian languages. The linguists sourced such data by manually translating
open-source English test sets, collecting real user questions from local forums and websites,
collecting real math and reasoning questions from reputable sources, as well as writing test
instructions and questions themselves. The Sea-bench test set contains 20 questions per task
(5 tasks for 3 languages, 4 tasks for other 5 languages).
"""

_HOMEPAGE = "https://huggingface.co/datasets/SeaLLMs/Sea-bench"

_LANGUAGES = ["eng", "ind", "khm", "lao", "mya", "tgl", "tha", "vie", "zlm"]

_LICENSE = Licenses.APACHE_2_0.value

_LOCAL = False

_URLS = "https://huggingface.co/datasets/SeaLLMs/Sea-bench/raw/main/question.jsonl"

_SUPPORTED_TASKS = [Tasks.INSTRUCTION_TUNING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class SeaBenchDataset(datasets.GeneratorBasedBuilder):
    """
    Sea-bench is a multilingual benchmark from https://huggingface.co/datasets/SeaLLMs/Sea-bench.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    LANGUAGES_DICT = {"tgl": "tl", "khm": "km", "vie": "vi", "tha": "th", "lao": "lo", "mya": "my", "ind": "id", "zlm": "ms", "eng": "en"}

    BUILDER_CONFIGS = (
        [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} source schema for all 8 languages",
                schema="source",
                subset_id=f"{_DATASETNAME}",
            )
        ]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME}_{lang} source schema",
                schema="source",
                subset_id=f"{_DATASETNAME}_{lang}",
            )
            for lang in LANGUAGES_DICT
        ]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_t2t",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} SEACrowd schema for T2T for all 8 languages",
                schema="seacrowd_t2t",
                subset_id=f"{_DATASETNAME}",
            )
        ]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang}_seacrowd_t2t",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME}_{lang} SEACrowd schema for T2T",
                schema="seacrowd_t2t",
                subset_id=f"{_DATASETNAME}_{lang}",
            )
            for lang in LANGUAGES_DICT
        ]
    )

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "question_id": datasets.Value("int64"),
                    "category": datasets.Value("string"),
                    "lang": datasets.Value("string"),
                    "turns": datasets.Sequence(datasets.Value("string")),
                    "chatgpt_response": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features
        else:
            raise ValueError(f"Invalid schema: '{self.config.schema}'")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """
        Returns SplitGenerators.
        """

        train_path = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """
        subset_id = self.config.subset_id.split("_")
        if len(subset_id) > 2:
            language_list = subset_id[2]
            if language_list in self.LANGUAGES_DICT:
                language_list = [self.LANGUAGES_DICT[language_list]]
        else:
            language_list = list(self.LANGUAGES_DICT.values())

        idx = 0
        with open(filepath, "r") as f:
            data = list(map(json.loads, f))
            if self.config.schema == "source":
                for d in data:
                    if d["lang"] in language_list:
                        x = {k: v if v != "" and k in self.info.features else None for k, v in d.items()}
                        if "chatgpt_response" not in x:
                            x["chatgpt_response"] = ""
                        yield idx, x
                        idx += 1
            elif self.config.schema == "seacrowd_t2t":
                for d in data:
                    if d["lang"] in language_list:
                        x = {
                            "id": idx,
                            "text_1": d["turns"][0] if "turns" in d else "",
                            "text_2": d["chatgpt_response"] if "chatgpt_response" in d else "",
                            "text_1_name": "turns",
                            "text_2_name": "chatgpt_response",
                        }
                        yield idx, x
                        idx += 1
            else:
                raise ValueError(f"Invalid schema: '{self.config.schema}'")
