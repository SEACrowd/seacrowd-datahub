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
@inproceedings{bhattacharjee-etal-2023-crosssum,
  author    = {Bhattacharjee, Abhik and Hasan, Tahmid and Ahmad, Wasi Uddin and Li, Yuan-Fang and Kang, Yong-Bin and Shahriyar, Rifat},
  title     = {CrossSum: Beyond English-Centric Cross-Lingual Summarization for 1,500+ Language Pairs},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  publisher = {Association for Computational Linguistics},
  year      = {2023},
  url       = {https://aclanthology.org/2023.acl-long.143},
  doi       = {10.18653/v1/2023.acl-long.143},
  pages     = {2541--2564},
  }
"""

_LOCAL = False
_LANGUAGES = ["ind", "mya", "vie"]
_DATASETNAME = "crosssum"
_DESCRIPTION = """
This is a large-scale cross-lingual summarization dataset containing article-summary samples in 1,500+ language pairs,
including pairs with the Burmese, Indonesian and Vietnamese languages. Articles in the first language are assigned
summaries in the second language.
"""

_HOMEPAGE = "https://huggingface.co/datasets/csebuetnlp/CrossSum"
_LICENSE = Licenses.CC_BY_NC_SA_4_0.value
_URL = "https://huggingface.co/datasets/csebuetnlp/CrossSum"


_SUPPORTED_TASKS = [Tasks.CROSS_LINGUAL_SUMMARIZATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class CrossSumDataset(datasets.GeneratorBasedBuilder):
    """Dataset of cross-lingual article-summary samples."""

    SUBSETS = [
        "ind_mya",
        "ind_vie",
        "mya_ind",
        "mya_vie",
        "vie_mya",
        "vie_ind",
    ]
    LANG_CODE_MAPPER = {"ind": "indonesian", "mya": "burmese", "vie": "vietnamese"}

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
            name=f"{_DATASETNAME}_{subset}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema for {subset} subset",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in SUBSETS
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_ind_mya_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "source_url": datasets.Value("string"),
                    "target_url": datasets.Value("string"),
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

    def _load_hf_data_from_remote(self, split: str) -> datasets.DatasetDict:
        """Load dataset from HuggingFace."""
        source_lang = self.LANG_CODE_MAPPER[self.config.subset_id.split("_")[-2]]
        target_lang = self.LANG_CODE_MAPPER[self.config.subset_id.split("_")[-1]]
        HF_REMOTE_REF = "/".join(_URL.split("/")[-2:])
        _hf_dataset_source = datasets.load_dataset(HF_REMOTE_REF, f"{source_lang}-{target_lang}", split=split)
        return _hf_dataset_source

    def _generate_examples(self, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = self._load_hf_data_from_remote(split)
        for index, row in enumerate(data):
            if self.config.schema == "source":
                example = row
            elif self.config.schema == "seacrowd_t2t":
                example = {"id": str(index), "text_1": row["text"], "text_2": row["summary"], "text_1_name": "document", "text_2_name": "summary"}
            yield index, example
