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
The GATITOS (Google's Additional Translations Into Tail-languages: Often Short) dataset is a high-quality, multi-way parallel dataset of tokens and short phrases.
This dataset consists in 4,000 English segments (4,500 tokens) that have been translated into each of 173 languages, 170 of which are low-resource, 23 are spoken in Southeast Asia.
This dataset contains primarily short segments: 93% single tokens, and only 23 sentences (0.6%) have over 5 tokens.
As such it is best thought of as a multilingual lexicon, rather than a parallel training corpus.
The source text is frequent words in the English Language, along with some common phrases and short sentences.
Care has been taken to ensure that they include good coverage of numbers, months, days of the week, swadesh words, and names of the languages themselves (including the endonym).
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{jones2023bilex,
      title={Bilex Rx: Lexical Data Augmentation for Massively Multilingual Machine Translation},
      author={Alex Jones and Isaac Caswell and Ishank Saxena and Orhan Firat},
      year={2023},
      eprint={2303.15265},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
}
"""

_DATASETNAME = "gatitos"

_DESCRIPTION = """\
The GATITOS (Google's Additional Translations Into Tail-languages: Often Short) dataset is a high-quality, multi-way parallel dataset of tokens and short phrases.
This dataset consists in 4,000 English segments (4,500 tokens) that have been translated into each of 173 languages, 170 of which are low-resource, 23 are spoken in Southeast Asia.
This dataset contains primarily short segments: 93% single tokens, and only 23 sentences (0.6%) have over 5 tokens.
As such it is best thought of as a multilingual lexicon, rather than a parallel training corpus.
The source text is frequent words in the English Language, along with some common phrases and short sentences.
Care has been taken to ensure that they include good coverage of numbers, months, days of the week, swadesh words, and names of the languages themselves (including the endonym).
"""

_HOMEPAGE = "https://github.com/google-research/url-nlp/blob/main/gatitos/README.md"

_LANGUAGES = ["ace", "ban", "bbc", "bew", "bjn", "bts", "btx", "bug", "cnh", "hil", "iba", "ilo", "kac", "lus", "mad", "mak", "meo", "min", "pag", "pam", "shn", "tet", "war"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLs = "https://raw.githubusercontent.com/google-research/url-nlp/main/gatitos/{src}_{tgt}.tsv"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class GATITOSDataset(datasets.GeneratorBasedBuilder):
    """The GATITOS (Google's Additional Translations Into Tail-languages: Often Short) dataset is a high-quality, multi-way parallel dataset of tokens and short phrases."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{src_lang}_{tgt_lang}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_{src_lang}_{tgt_lang}",
        )
        for (src_lang, tgt_lang) in [("eng", lang) for lang in _LANGUAGES] + [(lang, "eng") for lang in _LANGUAGES]
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{src_lang}_{tgt_lang}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_{src_lang}_{tgt_lang}",
        )
        for (src_lang, tgt_lang) in [("eng", lang) for lang in _LANGUAGES] + [(lang, "eng") for lang in _LANGUAGES]
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "src_text": datasets.Value("string"), "tgt_text": datasets.Value("string")})

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

        _, src_lang, tgt_lang = self.config.subset_id.split("_")

        filepath = dl_manager.download_and_extract(_URLs.format(src=src_lang.replace("eng", "en"), tgt=tgt_lang.replace("eng", "en")))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": filepath, "src_lang": src_lang, "tgt_lang": tgt_lang},
            )
        ]

    def _generate_examples(self, src_lang: str, tgt_lang: str, filepath: Path) -> Tuple[int, Dict]:
        if self.config.schema == "source":
            for row_id, row in enumerate(open(filepath)):
                src_text, tgt_text = row.strip().split("\t")
                yield row_id, {"id": row_id, "src_text": src_text, "tgt_text": tgt_text}

        elif self.config.schema == "seacrowd_t2t":
            for row_id, row in enumerate(open(filepath)):
                src_text, tgt_text = row.strip().split("\t")
                yield row_id, {"id": row_id, "text_1": src_text, "text_2": tgt_text, "text_1_name": src_lang, "text_2_name": tgt_lang}
