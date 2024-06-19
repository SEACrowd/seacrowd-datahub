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
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{palen-michel-lignos-2023-lr,
  author    = {Palen-Michel, Chester and Lignos, Constantine},
  title     = {LR - Sum: Summarization for Less-Resourced Languages},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  year      = {2023},
  publisher = {Association for Computational Linguistics},
  address   = {Toronto, Canada},
  doi       = {10.18653/v1/2023.findings-acl.427},
  pages     = {6829--6844},
}
"""

_LOCAL = False
_LANGUAGES = ["ind", "khm", "lao", "mya", "tha", "vie"]

_DATASETNAME = "lr_sum"
_DESCRIPTION = """
LR-Sum is a news abstractive summarization dataset focused on low-resource languages. It contains human-written summaries
for 39 languages and the data is based on the Multilingual Open Text corpus
(ultimately derived from the Voice of America website).
"""

_HOMEPAGE = "https://huggingface.co/datasets/bltlab/lr-sum"
_LICENSE = Licenses.CC_BY_4_0.value
_URL = "https://huggingface.co/datasets/bltlab/lr-sum"

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class LRSumDataset(datasets.GeneratorBasedBuilder):
    """Dataset of article-summary pairs for different low-resource languages."""

    # Config to load individual datasets per language
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema for {lang} language",
            schema="source",
            subset_id=f"{_DATASETNAME}_{lang}",
        )
        for lang in _LANGUAGES
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema for {lang} language",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_{lang}",
        )
        for lang in _LANGUAGES
    ]

    # Config to load all datasets
    BUILDER_CONFIGS.extend(
        [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} source schema for all languages",
                schema="source",
                subset_id=_DATASETNAME,
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_t2t",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} SEACrowd schema for all languages",
                schema="seacrowd_t2t",
                subset_id=_DATASETNAME,
            ),
        ]
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                    "text": datasets.Value("string"),
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
        # dl_manager not used since dataloader uses HF 'load_dataset'
        return [
            datasets.SplitGenerator(name=split, gen_kwargs={"split": split._name})
            for split in (
                datasets.Split.TRAIN,
                datasets.Split.VALIDATION,
                datasets.Split.TEST,
            )
        ]

    def _load_hf_data_from_remote(self, lang: str, split: str) -> datasets.DatasetDict:
        """Load dataset from HuggingFace."""
        hf_remote_ref = "/".join(_URL.split("/")[-2:])
        return datasets.load_dataset(hf_remote_ref, lang, split=split)

    def _generate_examples(self, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        lr_sum_datasets = []

        lang = self.config.subset_id.split("_")[-1]
        if lang in _LANGUAGES:
            lr_sum_datasets.append(self._load_hf_data_from_remote(lang, split))
        else:
            for lang in _LANGUAGES:
                lr_sum_datasets.append(self._load_hf_data_from_remote(lang, split))

        index = 0
        for lang_subset in lr_sum_datasets:
            for row in lang_subset:
                if self.config.schema == "source":
                    example = row

                elif self.config.schema == "seacrowd_t2t":
                    example = {
                        "id": str(index),
                        "text_1": row["text"],
                        "text_2": row["summary"],
                        "text_1_name": "document",
                        "text_2_name": "summary",
                    }
                yield index, example
                index += 1
