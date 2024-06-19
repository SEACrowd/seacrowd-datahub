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
Parallel and monolingual data for training machine translation systems translating English texts into Vietnamese, with a focus on news domain.
The data was crawled from high-quality bilingual or multilingual websites of news and one-speaker educational talks on various topics, mostly technology, entertainment, and design (hereby referred to as TED-like talks).
The dataset also includes noisy movie subtitles from the OpenSubtitle dataset.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{vlsp2020-mt,
title     = {{Goals, Challenges and Findings of the VLSP 2020 English-Vietnamese News Translation Shared Task}},
author    = {Thanh-Le Ha and Van-Khanh Tran and Kim-Anh Nguyen},
booktitle = {{Proceedings of the 7th International Workshop on Vietnamese Language and Speech Processing - VLSP 2020}},
year      = {2020}
}
"""

_DATASETNAME = "vlsp2020_mt_envi"

_DESCRIPTION = """\
Parallel and monolingual data for training machine translation systems translating English texts into Vietnamese, with a focus on news domain.
The data was crawled from high-quality bilingual or multilingual websites of news and one-speaker educational talks on various topics, mostly technology, entertainment, and design (hereby referred to as TED-like talks).
The dataset also includes noisy movie subtitles from the OpenSubtitle dataset.
"""

_HOMEPAGE = "https://github.com/thanhleha-kit/EnViCorpora"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = "https://github.com/thanhleha-kit/EnViCorpora/archive/refs/heads/master.zip"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class Vlsp2020MtEnviDataset(datasets.GeneratorBasedBuilder):
    """
    Parallel and monolingual data for training machine translation systems translating English texts into Vietnamese, with a focus on news domain.
    The data was crawled from high-quality bilingual or multilingual websites of news and one-speaker educational talks on various topics, mostly technology, entertainment, and design (hereby referred to as TED-like talks).
    The dataset also includes noisy movie subtitles from the OpenSubtitle dataset.
    """

    # Skipping openSub & mono-vi for future development (Large Drive file download bottleneck)
    subsets = {
        # key: subset_id, value: subset_filename
        "EVBCorpus": [
            ("bitext", datasets.Split.TRAIN),
        ],
        "VLSP20-official": [
            ("offi_test", datasets.Split.TEST),
        ],
        "basic": [
            ("data", datasets.Split.TRAIN),
        ],
        "indomain-news": [
            ("train", datasets.Split.TRAIN),
            ("dev", datasets.Split.VALIDATION),
            ("tst", datasets.Split.TEST),
        ],
        "iwslt15": [
            ("train", datasets.Split.TRAIN),
            ("dev", datasets.Split.VALIDATION),
            ("test", datasets.Split.TEST),
        ],
        "iwslt15-official": [
            ("IWSLT15.official_test", datasets.Split.TEST),
        ],
        "ted-like": [
            ("data", datasets.Split.TRAIN),
        ],
        "wiki-alt": [
            ("data", datasets.Split.TRAIN),
        ],
    }

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME}_{subset} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in list(subsets.keys())
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME}_{subset} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in list(subsets.keys())
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_VLSP20-official_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text_en": datasets.Value("string"),
                    "text_vi": datasets.Value("string"),
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
        subset_id = self.config.subset_id.split("_")[-1]

        filenames = self.subsets[subset_id]
        if "iwslt15" in subset_id:  # for iwslt15-official
            subset_id = "iwslt15"

        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=splitname,
                gen_kwargs={
                    "filepath": {
                        "en": os.path.join(data_dir, "EnViCorpora-master", subset_id, f"{filename}.en"),
                        "vi": os.path.join(data_dir, "EnViCorpora-master", subset_id, f"{filename}.vi"),
                    },
                },
            )
            for filename, splitname in filenames
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath["en"], "r") as f:
            en = f.readlines()
        with open(filepath["vi"], "r") as f:
            vi = f.readlines()

        if self.config.schema == "source":
            for i, (en_text, vi_text) in enumerate(zip(en, vi)):
                yield i, {
                    "id": str(i),
                    "text_en": en_text.strip(),
                    "text_vi": vi_text.strip(),
                }

        elif self.config.schema == "seacrowd_t2t":
            for i, (en_text, vi_text) in enumerate(zip(en, vi)):
                yield i, {
                    "id": str(i),
                    "text_1": en_text.strip(),
                    "text_2": vi_text.strip(),
                    "text_1_name": "en",
                    "text_2_name": "vi",
                }
