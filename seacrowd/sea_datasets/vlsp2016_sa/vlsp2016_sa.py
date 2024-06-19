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
@article{nguyen2018vlsp,
  title={VLSP shared task: sentiment analysis},
  author={Nguyen, Huyen TM and Nguyen, Hung V and Ngo, \
Quyen T and Vu, Luong X and Tran, Vu Mai and Ngo, Bach X and Le, Cuong A},
  journal={Journal of Computer Science and Cybernetics},
  volume={34},
  number={4},
  pages={295--310},
  year={2018}
}
"""
_DATASETNAME = "vlsp2016_sa"

_DESCRIPTION = """\
The SA-VLSP2016 dataset were collected from three source sites which are tinhte.vn, \
vnexpress.net and Facebook, and used for the sentiment analysis task. The data consists \
of comments of technical articles on those sites. Each comment is given one of \
four labels: POS (positive), NEG (negative), NEU (neutral) and USELESS (filter-out).
"""

_HOMEPAGE = "https://vlsp.org.vn/resources-vlsp2016"
_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value
_LOCAL = True

_URLS = {}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"
_TAGS = ["POS", "NEG", "NEU"]


class VLSP2016SADataset(datasets.GeneratorBasedBuilder):
    """The SA-VLSP2016 dataset, used for sentiment analysis, comprises comments from technical \
    articles on tinhte.vn, vnexpress.net, and Facebook, each labeled as positive, negative, neutral, or filter-out."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "text"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_tokenized_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_tokenized",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=_TAGS),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(_TAGS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "SA2016-training_data"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "SA2016-TestData-Ans"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        if split == "dev":
            if self.config.schema in ["source", f"seacrowd_{self.SEACROWD_SCHEMA_NAME}"]:
                labelfile = "test_raw_ANS.txt"
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}_tokenized":
                labelfile = "test_tokenized_ANS.txt"

            with open(os.path.join(filepath, labelfile)) as file:
                data = file.read()

            pattern = re.compile("(?P<sentence>.+)\n(?P<label>(POS|NEG|NEU))\n")

            if self.config.schema in ["source", f"seacrowd_{self.SEACROWD_SCHEMA_NAME}", f"seacrowd_{self.SEACROWD_SCHEMA_NAME}_tokenized"]:
                for i, match in enumerate(pattern.finditer(data)):
                    yield i, {"id": i, "text": match.group("sentence").replace("\xa0", " "), "label": match.group("label")}

        else:
            labeltext = {"POS": [], "NEG": [], "NEU": []}
            if self.config.schema in ["source", f"seacrowd_{self.SEACROWD_SCHEMA_NAME}"]:
                positive = "SA-training_positive.txt"
                negative = "SA-training_negative.txt"
                neutral = "SA-training_neutral.txt"
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}_tokenized":
                positive = "train_positive_tokenized.txt"
                negative = "train_negative_tokenized.txt"
                neutral = "train_neutral_tokenized.txt"

            for labelsplit, labelfile in zip(labeltext.keys(), [positive, negative, neutral]):
                with open(os.path.join(filepath, labelfile)) as file:
                    data = file.read()
                labeltext[labelsplit] = data.split("\n\n")[:-1]

            if self.config.schema in ["source", f"seacrowd_{self.SEACROWD_SCHEMA_NAME}", f"seacrowd_{self.SEACROWD_SCHEMA_NAME}_tokenized"]:
                idcounter = 0
                for label, sentences in labeltext.items():
                    for sentence in sentences:
                        yield idcounter, {"id": idcounter, "text": sentence, "label": label}
                        idcounter = idcounter + 1
