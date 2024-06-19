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
MULTISPIDER, the largest multilingual text-to-SQL dataset which covers \
    seven languages (English, German, French, Spanish, Japanese, \
    Chinese, and Vietnamese). Upon MULTISPIDER, we further identify \
    the lexical and structural challenges of text-to-SQL (caused by \
    specific language properties and dialect sayings) and their \
    intensity across different languages.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@inproceedings{Dou2022MultiSpiderTB,
  title={MultiSpider: Towards Benchmarking Multilingual Text-to-SQL Semantic Parsing},
  author={Longxu Dou and Yan Gao and Mingyang Pan and Dingzirui Wang and Wanxiang Che and Dechen Zhan and Jian-Guang Lou},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2023},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/26499/26271}
}
"""

_DATASETNAME = "multispider"

_DESCRIPTION = """\
MULTISPIDER, the largest multilingual text-to-SQL dataset which covers \
    seven languages (English, German, French, Spanish, Japanese, \
    Chinese, and Vietnamese). Upon MULTISPIDER, we further identify \
    the lexical and structural challenges of text-to-SQL (caused by \
    specific language properties and dialect sayings) and their \
    intensity across different languages.
"""

_HOMEPAGE = "https://github.com/longxudou/multispider"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    "train": "https://huggingface.co/datasets/dreamerdeo/multispider/resolve/main/dataset/multispider/with_original_value/train_vi.json?download=true",
    "dev": "https://huggingface.co/datasets/dreamerdeo/multispider/raw/main/dataset/multispider/with_original_value/dev_vi.json",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MultispiderDataset(datasets.GeneratorBasedBuilder):
    """
    MULTISPIDER, the largest multilingual text-to-SQL dataset which covers \
    seven languages (English, German, French, Spanish, Japanese, \
    Chinese, and Vietnamese). Upon MULTISPIDER, we further identify \
    the lexical and structural challenges of text-to-SQL (caused by \
    specific language properties and dialect sayings) and their \
    intensity across different languages.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "t2t"

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
                    "db_id": datasets.Value("string"),
                    "query": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "query_toks": datasets.Sequence(feature=datasets.Value("string")),
                    "query_toks_no_value": datasets.Sequence(feature=datasets.Value("string")),
                    "question_toks": datasets.Sequence(feature=datasets.Value("string")),
                    "sql": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        data_path_train = Path(dl_manager.download_and_extract(_URLS["train"]))
        data_path_dev = Path(dl_manager.download_and_extract(_URLS["dev"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path_train,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_path_dev,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = pd.read_json(filepath)

        for index, row in df.iterrows():

            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                example = {
                    "id": str(index),
                    "text_1": str(row["question"]),
                    "text_2": str(row["query"]),
                    "text_1_name": "question",
                    "text_2_name": "query",
                }

            yield index, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
