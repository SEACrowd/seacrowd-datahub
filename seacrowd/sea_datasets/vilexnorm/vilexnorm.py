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
The ViLexNorm corpus is a collection of comment pairs in Vietnamese, designed for the task of lexical normalization. The corpus contains 10,467 comment pairs, carefully curated and annotated for lexical normalization purposes.
These comment pairs are partitioned into three subsets: training, development, and test, distributed in an 8:1:1 ratio.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{nguyen-etal-2024-vilexnorm,
    title = "{V}i{L}ex{N}orm: A Lexical Normalization Corpus for {V}ietnamese Social Media Text",
    author = "Nguyen, Thanh-Nhi  and
      Le, Thanh-Phong  and
      Nguyen, Kiet",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.85",
    pages = "1421--1437",
    abstract = "Lexical normalization, a fundamental task in Natural Language Processing (NLP), involves the transformation of words into their canonical forms. This process has been proven to benefit various downstream NLP tasks greatly.
    In this work, we introduce Vietnamese Lexical Normalization (ViLexNorm), the first-ever corpus developed for the Vietnamese lexical normalization task. The corpus comprises over 10,000 pairs of sentences meticulously annotated
    by human annotators, sourced from public comments on Vietnam{'}s most popular social media platforms. Various methods were used to evaluate our corpus, and the best-performing system achieved a result of 57.74% using
    the Error Reduction Rate (ERR) metric (van der Goot, 2019a) with the Leave-As-Is (LAI) baseline. For extrinsic evaluation, employing the model trained on ViLexNorm demonstrates the positive impact of the Vietnamese lexical normalization task
    on other NLP tasks. Our corpus is publicly available exclusively for research purposes.",
}
"""

_DATASETNAME = "vilexnorm"

_DESCRIPTION = """\
The ViLexNorm corpus is a collection of comment pairs in Vietnamese, designed for the task of lexical normalization. The corpus contains 10,467 comment pairs, carefully curated and annotated for lexical normalization purposes.
These comment pairs are partitioned into three subsets: training, development, and test, distributed in an 8:1:1 ratio.
"""

_HOMEPAGE = "https://github.com/ngxtnhi/ViLexNorm"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_LOCAL = False

_URLS = {
    "train": "https://raw.githubusercontent.com/ngxtnhi/ViLexNorm/main/data/train.csv",
    "dev": "https://raw.githubusercontent.com/ngxtnhi/ViLexNorm/main/data/dev.csv",
    "test": "https://raw.githubusercontent.com/ngxtnhi/ViLexNorm/main/data/test.csv",
}

_SUPPORTED_TASKS = [Tasks.MULTILEXNORM]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class VilexnormDataset(datasets.GeneratorBasedBuilder):
    """The ViLexNorm corpus is a collection of comment pairs in Vietnamese, designed for the task of lexical normalization. The corpus contains 10,467 comment pairs, carefully curated and annotated for lexical normalization purposes.
    These comment pairs are partitioned into three subsets: training, development, and test, distributed in an 8:1:1 ratio."""

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
            name=f"{_DATASETNAME}_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "original": datasets.Value("string"),
                    "normalized": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_t2t":
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

        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["dev"],
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = pd.read_csv(filepath)

        if self.config.schema == "source":
            for i, row in df.iterrows():
                yield i, {
                    "id": i,
                    "original": row["original"],
                    "normalized": row["normalized"],
                }

        elif self.config.schema == "seacrowd_t2t":
            for i, row in df.iterrows():
                yield i, {
                    "id": str(i),
                    "text_1": row["original"],
                    "text_2": row["normalized"],
                    "text_1_name": "original",
                    "text_2_name": "normalized",
                }
