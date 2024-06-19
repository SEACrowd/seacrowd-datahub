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
Filipino-Tagalog Product Reviews Sentiment Analysis
This is a machine learning dataset that can be used to analyze the sentiment of product reviews in Filipino-Tagalog.
The data is scraped from lazada Philippines.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses


_CITATION = """@misc{github,
  author={Eric Echemane},
  title={Filipino-Tagalog-Product-Reviews-Sentiment-Analysis},
  year={2022},
  url={https://github.com/EricEchemane/Filipino-Tagalog-Product-Reviews-Sentiment-Analysis/tree/main},
}
"""

_DATASETNAME = "lazada_review_filipino"


_DESCRIPTION = """Filipino-Tagalog Product Reviews Sentiment Analysis
This is a machine learning dataset that can be used to analyze the sentiment of product reviews in Filipino-Tagalog.
The dataset contains over 900+ weakly annotated Filipino reviews scraped from the Lazada Philippines platform. 
Each review is associated with a five star point rating where one is the lowest and five is the highest.
"""


_HOMEPAGE = "https://github.com/EricEchemane/Filipino-Tagalog-Product-Reviews-Sentiment-Analysis"

_LANGUAGES = ['fil', 'tgl']

_LICENSE = Licenses.UNKNOWN.value


_LOCAL = False



_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/EricEchemane/Filipino-Tagalog-Product-Reviews-Sentiment-Analysis/main/data/reviews.json",
}


_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]


_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"



class LazadaReviewFilipinoDataset(datasets.GeneratorBasedBuilder):
    """The dataset contains over 900+ weakly annotated Filipino reviews scraped from the Lazada Philippines platform"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)


    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="lazada_review_filipino_source",
            version=SOURCE_VERSION,
            description="lazada reviews in filipino source schema",
            schema="source",
            subset_id="lazada_review_filipino",
        ),
        SEACrowdConfig(
            name="lazada_review_filipino_seacrowd_text",
            version=SEACROWD_VERSION,
            description="lazada reviews in filipino SEACrowd schema",
            schema="seacrowd_text",
            subset_id="lazada_review_filipino",
        ),
    ]

    DEFAULT_CONFIG_NAME = "lazada_review_filipino_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "review": datasets.Value("string"),
                                          "rating": datasets.Value("string")})

        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(label_names=["1", "2", "3", "4", "5"])

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
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,

                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, 'r') as file:
            data = json.load(file)

        if self.config.schema == "source":
            for i in range(len(data)):
                yield i, {"index": str(i), "review": data[i]['review'], "rating": data[i]['rating']}

        elif self.config.schema == "seacrowd_text":
            for i in range(len(data)):
                yield i, {"id": str(i), "text": data[i]['review'], "label": str(data[i]['rating'])}
