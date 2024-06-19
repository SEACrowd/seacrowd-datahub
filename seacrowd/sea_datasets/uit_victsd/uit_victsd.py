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
@inproceedings{,
  author    = {Nguyen, Luan Thanh and Van Nguyen, Kiet and Nguyen,  Ngan Luu-Thuy},
  title     = {Constructive and Toxic Speech Detection for Open-domain Social Media Comments in Vietnamese},
  booktitle = {Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices},
  year      = {2021},
  publisher = {Springer International Publishing},
  address   = {Kuala Lumpur, Malaysia},
  pages     = {572--583},
}
"""

_LOCAL = False
_LANGUAGES = ["vie"]
_DATASETNAME = "uit_victsd"
_DESCRIPTION = """
The UIT-ViCTSD (Vietnamese Constructive and Toxic Speech Detection dataset) is a compilation of 10,000 human-annotated
comments intended for constructive and toxic comments detection. The dataset spans 10 domains, reflecting the diverse topics
and expressions found in social media interactions among Vietnamese users.
"""

_HOMEPAGE = "https://github.com/tarudesu/ViCTSD"
_LICENSE = Licenses.UNKNOWN.value
_URL = "https://huggingface.co/datasets/tarudesu/ViCTSD"


_SUPPORTED_TASKS = [Tasks.INTENT_CLASSIFICATION, Tasks.ABUSIVE_LANGUAGE_PREDICTION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class UiTViCTSDDataset(datasets.GeneratorBasedBuilder):
    """
    Dataset of Vietnamese social media comments annotated
    for constructiveness and toxicity.
    """

    SUBSETS = ["constructiveness", "toxicity"]
    CLASS_LABELS = [0, 1]

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

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_constructiveness_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "Unnamed: 0": datasets.Value("int64"),  # Column name missing in original dataset
                    "Comment": datasets.Value("string"),
                    "Constructiveness": datasets.ClassLabel(names=self.CLASS_LABELS),
                    "Toxicity": datasets.ClassLabel(names=self.CLASS_LABELS),
                    "Title": datasets.Value("string"),
                    "Topic": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(label_names=self.CLASS_LABELS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # dl_manager not used since dataloader uses HF 'load_dataset'
        return [datasets.SplitGenerator(name=split, gen_kwargs={"split": split._name}) for split in (datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST)]

    def _load_hf_data_from_remote(self, split: str) -> datasets.DatasetDict:
        """Load dataset from HuggingFace."""
        HF_REMOTE_REF = "/".join(_URL.split("/")[-2:])
        _hf_dataset_source = datasets.load_dataset(HF_REMOTE_REF, split=split)
        return _hf_dataset_source

    def _generate_examples(self, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = self._load_hf_data_from_remote(split=split)
        for index, row in enumerate(data):
            if self.config.schema == "source":
                example = row

            elif self.config.schema == "seacrowd_text":
                if "constructiveness" in self.config.name:
                    label = row["Constructiveness"]
                elif "toxicity" in self.config.name:
                    label = row["Toxicity"]
                example = {"id": str(index), "text": row["Comment"], "label": label}
            yield index, example
