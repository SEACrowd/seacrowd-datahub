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

from seacrowd.sea_datasets.mtop_intent_classification.labels import (
    DOMAIN_LABELS, INTENT_LABELS)
from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{li-etal-2021-mtop,
  author    = {Li, Haoran and Arora, Abhinav and Chen, Shuochi and Gupta, Anchit and Gupta, Sonal and Mehdad, Yashar},
  title     = {MTOP: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark},
  booktitle   = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://aclanthology.org/2021.eacl-main.257},
  doi       = {10.18653/v1/2021.eacl-main.257},
  pages    = {2950-2962},
}
"""
_LOCAL = False
_LANGUAGES = ["tha"]
_DATASETNAME = "mtop_intent_classification"
_DESCRIPTION = """
This dataset contains annotated utterances from 6 languages, including Thai,
for semantic parsing. Queries corresponding to the chosen domains are crowdsourced.
 Two subsets are included in this dataset: 'domain' (eg. 'news', 'people', 'weather')
 and 'intent' (eg. 'GET_MESSAGE', 'STOP_MUSIC', 'END_CALL')
"""

_HOMEPAGE = "https://huggingface.co/mteb"
_LICENSE = Licenses.CC_BY_SA_4_0.value  # Found in original dataset (not HF) linked in paper
_URL = "https://huggingface.co/datasets/mteb/"


_SUPPORTED_TASKS = [Tasks.INTENT_CLASSIFICATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class MTOPIntentClassificationDataset(datasets.GeneratorBasedBuilder):
    """Dataset of Thai sentences and their domains or intents."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SUBSETS = ["domain", "intent"]

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
            name=f"{_DATASETNAME}_{subset}_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema for {subset} subset",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in SUBSETS
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_domain_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                    "label_text": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_text":
            if self.config.subset_id == "domain":
                labels = DOMAIN_LABELS
            elif self.config.subset_id == "intent":
                labels = INTENT_LABELS
            else:
                raise ValueError(f"Received unexpected schema name {self.config.name}")
            features = schemas.text_features(label_names=labels)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # dl_manager not used since dataloader uses HF `load_dataset`
        return [datasets.SplitGenerator(name=split, gen_kwargs={"split": split._name}) for split in (datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST)]

    def _load_hf_data_from_remote(self, split: str) -> datasets.DatasetDict:
        """Load dataset from HuggingFace."""
        if self.config.subset_id not in ("domain", "intent"):
            raise ValueError(f"Received unexpected schema name {self.config.name}")
        HF_REMOTE_REF = "/".join(_URL.split("/")[-2:]) + f"mtop_{self.config.subset_id}"
        _hf_dataset_source = datasets.load_dataset(HF_REMOTE_REF, "th", split=split)
        return _hf_dataset_source

    def _generate_examples(self, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = self._load_hf_data_from_remote(split=split)
        for index, row in enumerate(data):
            if self.config.schema == "source":
                example = row

            elif self.config.schema == "seacrowd_text":
                example = {"id": str(index), "text": row["text"], "label": row["label_text"]}
            yield index, example
