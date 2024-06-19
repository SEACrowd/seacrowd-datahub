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


import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{limkonchotiwat-etal-2021-handling,
    title = "Handling Cross- and Out-of-Domain Samples in {T}hai Word Segmentation",
    author = "Limkonchotiwat, Peerat  and
      Phatthiyaphaibun, Wannaphong  and
      Sarwar, Raheem  and
      Chuangsuwanich, Ekapol  and
      Nutanong, Sarana",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.86",
    doi = "10.18653/v1/2021.findings-acl.86",
    pages = "1003--1016",
}
"""

_DATASETNAME = "vistec_tp_th_21"

_DESCRIPTION = """\
The largest social media domain datasets for Thai text processing (word segmentation, 
misspell correction and detection, and named-entity boundary) called "VISTEC-TP-TH-2021" or VISTEC-2021. 
VISTEC corpus contains 49,997 sentences with 3.39M words where the collection was manually annotated by 
linguists on four tasks, namely word segmentation, misspelling detection and correction, 
and named entity recognition.
"""

_HOMEPAGE = "https://github.com/mrpeerat/OSKut/tree/main/VISTEC-TP-TH-2021"


_LANGUAGES = ["tha"]


_LICENSE = Licenses.CC_BY_SA_3_0.value

_LOCAL = False

_URLS = {
    "train": "https://raw.githubusercontent.com/mrpeerat/OSKut/main/VISTEC-TP-TH-2021/train/VISTEC-TP-TH-2021_train_proprocessed.txt",
    "test": "https://raw.githubusercontent.com/mrpeerat/OSKut/main/VISTEC-TP-TH-2021/test/VISTEC-TP-TH-2021_test_proprocessed.txt",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class VISTEC21Dataset(datasets.GeneratorBasedBuilder):
    """
    The largest social media domain datasets for Thai text processing (word segmentation,
    misspell correction and detection, and named-entity boundary) called "VISTEC-TP-TH-2021" or VISTEC-2021.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "seq_label"
    LABEL_CLASSES = ["0", "1"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.features.ClassLabel(names=self.LABEL_CLASSES)),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.seq_label_features(self.LABEL_CLASSES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_files = {
            "train": Path(dl_manager.download_and_extract(_URLS["train"])),
            "test": Path(dl_manager.download_and_extract(_URLS["test"])),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"], "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        label_key = "ner_tags" if self.config.schema == "source" else "labels"

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            id = 0
            for line in lines:
                tokens = line.split("|")
                token_list = []
                ner_tag = []
                for token in tokens:
                    if "<ne>" in token:
                        token = token.replace("<ne>", "")
                        token = token.replace("</ne>", "")
                        token_list.append(token)
                        ner_tag.append(1)
                        continue
                    if "</msp>" in token and "<msp value=" in token:
                        token_list.append(re.findall(r"<msp value=([^>]*)>", token)[0])
                        ner_tag.append(0)
                        continue
                    if "<compound>" in token or "</compound>" in token:
                        token = token.replace("<compound>", "")
                        token = token.replace("</compound>", "")
                        token_list.append(token)
                        ner_tag.append(0)
                        continue
                    token_list.append(token)
                    ner_tag.append(0)
                id += 1
                yield id, {
                    "id": str(id),
                    "tokens": token_list,
                    label_key: ner_tag,
                }
