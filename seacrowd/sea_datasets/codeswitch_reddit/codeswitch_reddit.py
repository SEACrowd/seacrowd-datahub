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
import html
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproccedings{rabinovich-2019-codeswitchreddit,
  author    = {Rabinovich, Ella and Sultani, Masih and Stevenson, Suzanne},
  title     = {CodeSwitch-Reddit: Exploration of Written Multilingual Discourse in Online Discussion Forums},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing},
  publisher = {Association for Computational Linguistics},
  year      = {2019},
  url       = {https://aclanthology.org/D19-1484},
  doi       = {10.18653/v1/D19-1484},
  pages     = {4776--4786},
}
"""

_LOCAL = False
_LANGUAGES = ["eng", "ind", "tgl"]
_DATASETNAME = "codeswitch_reddit"
_DESCRIPTION = """
This corpus consists of monolingual English and multilingual (English and one other language) posts
from country-specific subreddits, including r/indonesia, r/philippines and r/singapore for Southeast Asia.
Posts were manually classified whether they contained code-switching or not.
"""

_HOMEPAGE = "https://github.com/ellarabi/CodeSwitch-Reddit"
_LICENSE = Licenses.UNKNOWN.value
_URL = "http://www.cs.toronto.edu/~ella/code-switch.reddit.tar.gz"

_SUPPORTED_TASKS = [Tasks.CODE_SWITCHING_IDENTIFICATION, Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class CodeSwitchRedditDataset(datasets.GeneratorBasedBuilder):
    """Dataset of monolingual English and multilingual comments from country-specific subreddits."""

    SUBSETS = ["cs", "eng_monolingual"]
    INCLUDED_SUBREDDITS = ["indonesia", "Philippines", "singapore"]
    INCLUDED_LANGUAGES = {"English": "eng", "Indonesian": "ind", "Tagalog": "tgl"}

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema for {subset} subset",
            schema="source",
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in SUBSETS
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_eng_monolingual_seacrowd_ssp",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd ssp schema for eng_monolingual subset",
            schema="seacrowd_ssp",
            subset_id=f"{_DATASETNAME}_eng_monolingual",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_cs_seacrowd_text_multi",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd text multilabel schema for cs subset",
            schema="seacrowd_text_multi",
            subset_id=f"{_DATASETNAME}_cs",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_cs_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            if "cs" in self.config.subset_id:
                features = datasets.Features(
                    {
                        "author": datasets.Value("string"),
                        "subreddit": datasets.Value("string"),
                        "country": datasets.Value("string"),
                        "date": datasets.Value("int32"),
                        "confidence": datasets.Value("int32"),
                        "lang1": datasets.Value("string"),
                        "lang2": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        "id": datasets.Value("string"),
                        "link_id": datasets.Value("string"),
                        "parent_id": datasets.Value("string"),
                    }
                )
            elif "eng_monolingual" in self.config.subset_id:
                features = datasets.Features(
                    {
                        "author": datasets.Value("string"),
                        "subreddit": datasets.Value("string"),
                        "country": datasets.Value("string"),
                        "date": datasets.Value("int32"),
                        "confidence": datasets.Value("int32"),
                        "lang": datasets.Value("string"),
                        "text": datasets.Value("string"),
                    }
                )

        elif self.config.schema == "seacrowd_ssp":
            features = schemas.ssp_features
        elif self.config.schema == "seacrowd_text_multi":
            features = schemas.text_multi_features(label_names=list(self.INCLUDED_LANGUAGES.values()))

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
        if "cs" in self.config.subset_id:
            filepath = os.path.join(data_dir, "cs_main_reddit_corpus.csv")
        elif "eng_monolingual" in self.config.subset_id:
            filepath = os.path.join(data_dir, "eng_monolingual_reddit_corpus.csv")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(filepath, index_col=None, header="infer", encoding="utf-8")
        df = df[df["Subreddit"].isin(self.INCLUDED_SUBREDDITS)]

        if self.config.subset_id.split("_")[-1] == "cs":
            df = df[(df["Lang1"].isin(self.INCLUDED_LANGUAGES)) & (df["Lang2"].isin(self.INCLUDED_LANGUAGES))]
            df.reset_index(drop=True, inplace=True)

            for index, row in df.iterrows():
                parsed_text = html.unescape(row["Text"])
                if self.config.schema == "source":
                    example = {
                        "author": row["Author"],
                        "subreddit": row["Subreddit"],
                        "country": row["Country"],
                        "date": row["Date"],
                        "confidence": row["confidence"],
                        "lang1": row["Lang1"],
                        "lang2": row["Lang2"],
                        "text": parsed_text,
                        "id": row["id"],
                        "link_id": row["link_id"],
                        "parent_id": row["parent_id"],
                    }

                elif self.config.schema == "seacrowd_text_multi":
                    lang_one, lang_two = self.INCLUDED_LANGUAGES[row["Lang1"]], self.INCLUDED_LANGUAGES[row["Lang2"]]
                    example = {
                        "id": str(index),
                        "text": parsed_text,
                        "labels": list(sorted([lang_one, lang_two])),  # Language order doesn't matter in original dataset; just arrange alphabetically for consistency
                    }
                yield index, example

        else:
            df.reset_index(drop=True, inplace=True)
            for index, row in df.iterrows():
                parsed_text = html.unescape(row["Text"])
                if self.config.schema == "source":
                    example = {
                        "author": row["Author"],
                        "subreddit": row["Subreddit"],
                        "country": row["Country"],
                        "date": row["Date"],
                        "confidence": row["confidence"],
                        "lang": row["Lang"],
                        "text": parsed_text,
                    }
                elif self.config.schema == "seacrowd_ssp":
                    example = {
                        "id": str(index),
                        "text": parsed_text,
                    }
                yield index, example
