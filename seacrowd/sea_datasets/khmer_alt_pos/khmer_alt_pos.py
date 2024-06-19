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

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@article{10.1145/3464378,
author = {Kaing, Hour and Ding, Chenchen and Utiyama, Masao and Sumita, Eiichiro and Sam, Sethserey and Seng, Sopheap and Sudoh, Katsuhito and Nakamura, Satoshi},
title = {Towards Tokenization and Part-of-Speech Tagging for Khmer: Data and Discussion},
year = {2021},
issue_date = {November 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {20},
number = {6},
issn = {2375-4699},
url = {https://doi.org/10.1145/3464378},
doi = {10.1145/3464378},
abstract = {As a highly analytic language, Khmer has considerable ambiguities in tokenization and part-of-speech (POS) tagging processing. This topic is investigated in this study. Specifically, a 20,000-sentence Khmer corpus with manual tokenization and POS-tagging annotation is released after a series of work over the last 4 years. This is the largest morphologically annotated Khmer dataset as of 2020, when this article was prepared. Based on the annotated data, experiments were conducted to establish a comprehensive benchmark on the automatic processing of tokenization and POS-tagging for Khmer. Specifically, a support vector machine, a conditional random field (CRF), a long short-term memory (LSTM)-based recurrent neural network, and an integrated LSTM-CRF model have been investigated and discussed. As a primary conclusion, processing at morpheme-level is satisfactory for the provided data. However, it is intrinsically difficult to identify further grammatical constituents of compounds or phrases because of the complex analytic features of the language. Syntactic annotation and automatic parsing for Khmer will be scheduled in the near future.},
journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
month = {sep},
articleno = {104},
numpages = {16},
keywords = {annotated data, tokenization, POS-tagging, Khmer, machine learning}
}
"""

_LANGUAGES = ["khm"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "khmer_alt_pos"

_DESCRIPTION = """\
The data contains 20,000-sentence Khmer corpus with manual tokenization and POS-tagging annotation.
"""

_HOMEPAGE = "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/km-nova-181101/README.txt"

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_URL = "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/km-nova-181101.zip"

_SUPPORTED_TASKS = [Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.1.0"

_SEACROWD_VERSION = "2024.06.20"


class KhmerAltPOS(datasets.GeneratorBasedBuilder):
    """The data contains 20,000-sentence Khmer corpus with manual tokenization and POS-tagging annotation."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    POS_TAGS = ['n', 'v-', 'v', 'o', 'n[n', 'n]n', 'v]n', '1]n', '.', 'a[n', '1]a', 'a', 'a[v', 'n]a', 'v[a', 'v]v', 'o[o', 'o]o', 'o-', 'a[1', 'v[v', 'a[o', '1', 'a]a', 'n-[n', 'n-]n-', 'a-', 'n]v', 'a]n', 'o]a', 'v]o', 'n-]o', 'a[a', 'n-', 'n[1', 'n]o', '1[1', '1]1', 'n[o', 'n[v', 'a]o', '.]a', 'o-[o-', 'o]o-', 'o[v', 'n[a', 'v-]n', 'v]a', 'o[a', 'o[n', 'o[n-', 'o-]o', 'a-]n', 'n[.', 'o]v', 'o]n', '.]o', 'a-]a', 'a-]o', 'v-[v-', 'o]v-', 'o-]o-', 'v-]o', 'a]v', 'v[n', '.]n', 'n-]a', 'v]o-', 'o[o-', 'n-]n', '+', 'v[o', '.]1', 'v[1', 'n]1', '.]v', 'o]1', 'o[1', '1]o', 'v]1', 'n-[n-', '1]v', '1[n', 'a[.', '.[n', '.].', '+[n', 'n]+', '1[.', '+]o', 'n]o-', 'o-]a', 'v-[v', '.]v-', '.[1', 'v-[o']

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd Seq Label schema",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({
                "id": datasets.Value("string"),
                "tokens": [datasets.Value("string")],
                "labels": [datasets.Value("string")],
            })
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(self.POS_TAGS)

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_path = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        data_labels = open(f"{filepath}/km-nova-181101/data_km.km-tag.nova", "r").readlines()

        data_tokens = open(f"{filepath}/km-nova-181101/data_km.km-tok.nova", "r").readlines()

        mapping_sentences = {}

        for line in data_tokens:
            id, tokens = line.split("\t")
            tokens, _ = tokens.split("\n")

            if id not in mapping_sentences:
                mapping_sentences[id] = {}
            
            mapping_sentences[id]["tokens"] = tokens.split(" ")

        for line in data_labels:
            id, labels = line.split("\t")
            labels, _ = labels.split("\n")
            
            if id not in mapping_sentences:
                mapping_sentences[id] = {}
            
            mapping_sentences[id]["labels"] = labels.split(" ")

        if self.config.schema == "source" or self.config.schema == "seacrowd_seq_label":
            for num, key in enumerate(mapping_sentences):
                yield num, {
                    "id": key,
                    "tokens": mapping_sentences[key]["tokens"],
                    "labels": mapping_sentences[key]["labels"]
                }

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")
