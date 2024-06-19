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
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{hernandez-devaraj-2019-phfakenews,
  author    = {Fernandez, Aaron Carl T. and Devaraj, Madhavi},
  title     = {Computing the Linguistic-Based Cues of Fake News in the Philippines Towards its Detection},
  booktitle = {Proceedings of the 9th International Conference on Web Intelligence, Mining and Semantics},
  publisher = {Association for Computing Machinery},
  year      = {2019},
  url       = {https://dl.acm.org/doi/abs/10.1145/3326467.3326490},
  doi       = {10.1145/3326467.3326490},
  pages     = {1-9},
}
"""

_LOCAL = False
_LANGUAGES = ["eng"]
_DATASETNAME = "ph_fake_news_corpus"
_DESCRIPTION = """
The Philippine Fake News Corpus consists of news headlines and content from various "credible" and "non-credible"
national news outlets. "Credible" sources were national broadsheets available in the National Library of the
Philippines, while "non-credible" sources were sources included in lists of websites with fake or unverified content
provided by government and private institutions.
"""

_HOMEPAGE = "https://github.com/aaroncarlfernandez/Philippine-Fake-News-Corpus"
_LICENSE = Licenses.UNKNOWN.value
_URL = "https://github.com/aaroncarlfernandez/Philippine-Fake-News-Corpus/raw/master/Philippine%20Fake%20News%20Corpus.zip/"

_SUPPORTED_TASKS = [Tasks.FACT_CHECKING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class PhilippineFakeNewsDataset(datasets.GeneratorBasedBuilder):
    """
    Dataset of English news articles from the Philippines manually annotated as "credible" or
    "non-credible" based on source.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "Headline": datasets.Value("string"),
                "Content": datasets.Value("string"),
                "Authors": datasets.Value("string"),
                "Date": datasets.Value("string"),
                "URL": datasets.Value("string"),
                "Brand": datasets.Value("string"),
                "Label": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "Philippine Fake News Corpus.csv"),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(filepath, index_col=None, header="infer", encoding="utf-8")
        for index, example in df.iterrows():
            yield index, example.to_dict()