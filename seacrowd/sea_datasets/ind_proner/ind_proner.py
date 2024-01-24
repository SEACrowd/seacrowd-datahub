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
from seacrowd.utils.common_parser import load_conll_data
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@INPROCEEDINGS{9212879,
  author={Akmal, Muhammad and Romadhony, Ade},
  booktitle={2020 International Conference on Data Science and Its Applications (ICoDSA)},
  title={Corpus Development for Indonesian Product Named Entity Recognition Using Semi-supervised Approach},
  year={2020},
  volume={},
  number={},
  pages={1-5},
  keywords={Feature extraction;Labeling;Buildings;Semisupervised learning;Training data;Text recognition;Manuals;proner;semi-supervised learning;crf},
  doi={10.1109/ICoDSA50139.2020.9212879}
}
"""

_DATASETNAME = "ind_proner"

_DESCRIPTION = """\
Indonesian PRONER is a corpus for Indonesian product named entity recognition . It contains data was labeled manually
and data that was labeled automatically through a semi-supervised learning approach of conditional random fields (CRF).
"""

_HOMEPAGE = "https://github.com/dziem/proner-labeled-text"

_LANGUAGES = {"ind": "id"}

_LANGUAGE_CODES = list(_LANGUAGES.values())

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = [
    "https://raw.githubusercontent.com/dziem/proner-labeled-text/master/automatically_labeled.tsv",
    "https://raw.githubusercontent.com/dziem/proner-labeled-text/master/manually_labeled.tsv",
]

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

logger = datasets.logging.get_logger(__name__)


class IndPRONERDataset(datasets.GeneratorBasedBuilder):
    """
    Indonesian PRONER is a product named entity recognition dataset from https://github.com/dziem/proner-labeled-text.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    label_classes = [
        "B-PRO|B-BRA",
        "B-PRO|B-TYP",
        "B-PRO|I-PRO",
        "B-PRO|I-BRA",
        "B-PRO|I-TYP",
        "B-BRA|B-PRO",
        "B-BRA|B-TYP",
        "B-BRA|I-PRO",
        "B-BRA|I-BRA",
        "B-BRA|I-TYP",
        "B-TYP|B-PRO",
        "B-TYP|B-BRA",
        "B-TYP|I-PRO",
        "B-TYP|I-BRA",
        "B-TYP|I-TYP",
        "I-PRO|B-PRO",
        "I-PRO|B-BRA",
        "I-PRO|B-TYP",
        "I-PRO|I-BRA",
        "I-PRO|I-TYP",
        "I-BRA|B-PRO",
        "I-BRA|B-BRA",
        "I-BRA|B-TYP",
        "I-BRA|I-PRO",
        "I-BRA|I-TYP",
        "I-TYP|B-PRO",
        "I-TYP|B-BRA",
        "I-TYP|B-TYP",
        "I-TYP|I-PRO",
        "I-TYP|I-BRA",
        "B-PRO",
        "B-BRA",
        "B-TYP",
        "I-PRO",
        "I-BRA",
        "I-TYP",
        "O",
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(label_names=self.label_classes)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """
        Returns SplitGenerators.
        """

        train_paths = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": train_paths,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepaths: Path, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """
        idx = 0
        for filepath in filepaths:
            conll_dataset = load_conll_data(filepath)
            if self.config.schema == "source":
                for _, row in enumerate(conll_dataset):
                    x = {"id": str(idx), "tokens": row["sentence"], "ner_tags": row["label"]}
                    yield idx, x
                    idx += 1
            elif self.config.schema == "seacrowd_seq_label":
                for _, row in enumerate(conll_dataset):
                    x = {"id": str(idx), "tokens": row["sentence"], "labels": row["label"]}
                    yield idx, x
                    idx += 1
