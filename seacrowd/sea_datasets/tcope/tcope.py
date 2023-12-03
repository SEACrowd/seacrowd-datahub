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
This template serves as a starting point for contributing a dataset to the SEACrowd Datahub repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

# TODO: Add BibTeX citation
_CITATION = """\
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
_DESCRIPTION = """\
The TCOPE dataset consists of 1,048,576 public tweets (amounting to about 13.5 million words) collected from 13 major cities from the Philippines.
Tweets are tagged for part-of-speech and dependency parsing using spaCy.Tweets collected are from 2010 to 2021.
The publicly available dataset is only a random sample (10%) from the whole TCOPE dataset, which consist of roughly 27 million tweets
(amounting to about 135 million words) collected from 29 major cities during the same date range.
"""

_HOMEPAGE = "https://osf.io/3q5pw/wiki/home/"
_LICENSE = "Licenses.CC0_1_0.value"
_URL = "https://files.osf.io/v1/resources/3q5pw/providers/osfstorage/63737a5b0e715d3616a998f7"

_SUPPORTED_TASKS = [Tasks.DEPENDENCY_PARSING, Tasks.POS_TAGGING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class TCOPEDataset(datasets.GeneratorBasedBuilder):
    """TCOPE is a dataset of Philippine English tweets by Gonzales (2023)."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    # Actual data has invalid "labels" likely due to coding errors,
    # such as "BODY", "BIRTHDAY", "HAVAIANAS", etc. Only valid
    # POS tags are included here and in loaded data.
    pos_labels = [
        "NOUN", "PUNCT", "PROPN", "VERB", "PRON", "ADP",
        "ADJ", "ADV", "DET", "AUX", "PART", "CCONJ", "INTJ",
        "SPACE", "SCONJ", "NUM", "X", "SYM"
    ]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_kb",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd knowledge base schema",
            schema="seacrowd_kb",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd sequence labeling schema",
            schema="seacrowd_seq_label",
            subset_id=_DATASETNAME,
        )
    ]

    DEFAULT_CONFIG_NAME = "tcope_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
           features = datasets.Features(
               {
                   "copeid": datasets.Value("string"),
                   "userid": datasets.Value("int64"),
                   "divided.tweet": datasets.Value("string"),
                   "postag": datasets.Value("string"),
                   "deptag": datasets.Value("string"),
                   "citycode": datasets.Value("string"),
                   "year": datasets.Value("int64"),
                   "extendedcope": datasets.Value("string"),
               }
           )

        elif self.config.schema == "seacrowd_kb":
            #features = schemas.kb_features
        
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(label_names=self.pos_labels)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # First ZIP contains second ZIP which has spreadsheet data
        folder_zip_dir = dl_manager.download_and_extract(_URL)
        spreadsheet_zip_dir = dl_manager.extract(
            f"{folder_zip_dir}/public_v1/spreadsheet_format.zip"
        )
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

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(filepath, index_col=None)
        if self.config.schema == "source":
            # TODO: yield (key, example) tuples in the original dataset schema
            #for key, example in thing:
                #yield key, example

        elif self.config.schema == "seacrowd_seq_label":
            # TODO: yield (key, example) tuples in the seacrowd schema
            for key, example in thing:
                yield key, example

    def split_token_and_tag(self, tweet: str, valid_tags: List[str]) -> Tuple[List[str], List[str]]:
        tokens_with_tags = tweet.split()

        tokens = []
        tags = []
        for indiv_token_with_tag in tokens_with_tags:
            token, tag = indiv_token_with_tag.split("_")
            if tag in valid_tags:
                tokens.append(token)
                tags.append(tags)
        return tokens, tags

# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
