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
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@article{gonzales_broadening_2023,
  author    = {Gonzales, Wilkinson Daniel Wong},
  title     = {Broadening horizons in the diachronic and sociolinguisstic study of
  Philippine Englishes with the Twitter Corpus of Philippine Englishes (TCOPE)},
  journal   = {English World-Wide},
  year      = {2023},
  url       = {https://osf.io/k3qzx},
  doi       = {10.17605/OSF.IO/3Q5PW},
}
"""

_LOCAL = False
_LANGUAGES = ["eng", "fil"]
_DATASETNAME = "tcope"
_DESCRIPTION = """
The TCOPE dataset consists of public tweets (amounting to about 13.5 million words) collected from 13 major cities from the Philippines.
Tweets are either purely in English or involve code-switching between English and Filipino.
Tweets are tagged for part-of-speech and dependency parsing using spaCy. Tweets collected are from 2010 to 2021.
The publicly available dataset is only a random sample (10%) from the whole TCOPE dataset, which consist of roughly 27 million tweets
(amounting to about 135 million words) collected from 29 major cities during the same date range.
"""

_HOMEPAGE = "https://osf.io/3q5pw/wiki/home/"
_LICENSE = Licenses.CC0_1_0.value
_URL = "https://files.osf.io/v1/resources/3q5pw/providers/osfstorage/63737a5b0e715d3616a998f7"

_SUPPORTED_TASKS = [Tasks.POS_TAGGING, Tasks.DEPENDENCY_PARSING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class TCOPEDataset(datasets.GeneratorBasedBuilder):
    """TCOPE is a dataset of Philippine English tweets by Gonzales (2023)."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    # Actual data has invalid "labels" likely due to coding errors,
    # such as "BODY", "BIRTHDAY", "HAVAIANAS", etc. Only valid
    # POS tags are included here and in loaded data.
    POS_LABELS = ["NOUN", "PUNCT", "PROPN", "VERB", "PRON", "ADP", "ADJ", "ADV", "DET", "AUX", "PART", "CCONJ", "INTJ", "SPACE", "SCONJ", "NUM", "X", "SYM"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd sequence labeling schema",
            schema="seacrowd_seq_label",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = "tcope_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "copeid": datasets.Value("string"),
                    "userid": datasets.Value("int64"),
                    "divided_tweet": datasets.Value("string"),
                    "postag": datasets.Value("string"),
                    "deptag": datasets.Value("string"),
                    "citycode": datasets.Value("string"),
                    "year": datasets.Value("int64"),
                    "extendedcope": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(label_names=self.POS_LABELS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # First ZIP contains second ZIP
        # Second ZIP has spreadsheet data
        folder_zip_dir = dl_manager.download_and_extract(_URL)
        spreadsheet_zip_dir = dl_manager.extract(f"{folder_zip_dir}/public_v1/spreadsheet_format.zip")
        spreadsheet_fp = f"{spreadsheet_zip_dir}/spreadsheet_format/tcope_v1_public_sample.csv"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": spreadsheet_fp,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema not in ("source", "seacrowd_seq_label"):
            raise ValueError(f"Received unexpected config schema {self.config.schema}")

        df = pd.read_csv(filepath, index_col=None)
        df = df.rename(columns={"divided.tweet": "divided_tweet"}).query("divided_tweet.notna()")

        for index, row in df.iterrows():
            if self.config.schema == "source":
                example = row.to_dict()
            elif self.config.schema == "seacrowd_seq_label":
                tokens, tags = self.split_token_and_tag(row["postag"], valid_tags=self.POS_LABELS)
                example = {
                    "id": str(index),
                    "tokens": tokens,
                    "labels": tags,
                }
            yield index, example

    def split_token_and_tag(self, tweet: str, valid_tags: List[str]) -> Tuple[List[str], List[str]]:
        """Split tweet into two separate lists of tokens and tags."""
        tokens_with_tags = tweet.split()
        tokens = []
        tags = []
        for indiv_token_with_tag in tokens_with_tags:
            token, tag = indiv_token_with_tag.rsplit("_", 1)
            tokens.append(token)
            if tag in valid_tags:
                tags.append(tag)
            else:  # Use "X"/other spaCy tag for invalid POS tags
                tags.append("X")
        return tokens, tags
