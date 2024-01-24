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

import conllu
import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@INPROCEEDINGS{10053062,
  author={Samsuri, Mukhlizar Nirwan and Yuliawati, Arlisa and Alfina, Ika},
  booktitle={2022 5th International Seminar on Research of Information Technology and Intelligent Systems (ISRITI)},
  title={A Comparison of Distributed, PAM, and Trie Data Structure Dictionaries in Automatic Spelling Correction for Indonesian Formal Text},
  year={2022},
  pages={525-530},
  keywords={Seminars;Dictionaries;Data structures;Intelligent systems;Information technology;automatic spelling correction;distributed dictionary;non-word error;trie data structure;Partition Around Medoids},
  doi={10.1109/ISRITI56927.2022.10053062},
  url = {https://ieeexplore.ieee.org/document/10053062},
}
"""

_DATASETNAME = "etos"

_DESCRIPTION = """\
ETOS (Ejaan oTOmatiS) is a dataset for automatic spelling correction for formal Indonesian text.
It consists of 200 sentences with each sentence contains at least one typo. It has 4,323
tokens with 288 of them are non-word errors.
"""

_HOMEPAGE = "https://github.com/ir-nlp-csui/etos"

_LANGUAGES = {"ind": "id"}

_LANGUAGE_CODES = list(_LANGUAGES.values())

_LICENSE = Licenses.AGPL_3_0.value

_LOCAL = False

_URLS = "https://raw.githubusercontent.com/ir-nlp-csui/etos/main/gold_standard.conllu"

_SUPPORTED_TASKS = [Tasks.ERROR_SPELLING_CORRECTION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

logger = datasets.logging.get_logger(__name__)


class ETOSDataset(datasets.GeneratorBasedBuilder):
    """
    ETOS is an Indonesian error spelling correction dataset from https://github.com/ir-nlp-csui/etos.
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

    def _info(self) -> datasets.DatasetInfo:
        labels = []
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sent_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(label_names=labels)

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

        train_path = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """

        idx = 0
        with open(filepath, "r", encoding="utf-8") as data_file:
            tokenlist = list(conllu.parse_incr(data_file))
            for sent in tokenlist:
                if "sent_id" in sent.metadata:
                    sent_id = sent.metadata["sent_id"]
                else:
                    sent_id = idx

                tokens = [token["form"] for token in sent]

                if "text" in sent.metadata:
                    txt = sent.metadata["text"]
                else:
                    txt = " ".join(tokens)

                if self.config.schema == "source":
                    yield idx, {
                        "sent_id": str(sent_id),
                        "text": txt,
                        "tokens": [token["form"] for token in sent],
                    }

                elif self.config.schema == "seacrowd_seq_label":
                    yield idx, {
                        "id": str(sent_id),
                        "tokens": [token["form"] for token in sent],
                        "labels": [None for token in sent],
                    }
                idx += 1
