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
UIT-ViSFD is the Vietnamese Smartphone Feedback Dataset.
It is an aspect-based sentiment analysis dataset.
It consists of 11,122 human-annotated comments for mobile e-commerce.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
  @InProceedings{10.1007/978-3-030-82147-0_53,
  author="Luc Phan, Luong
  and Huynh Pham, Phuc
  and Thi-Thanh Nguyen, Kim
  and Khai Huynh, Sieu
  and Thi Nguyen, Tham
  and Thanh Nguyen, Luan
  and Van Huynh, Tin
  and Van Nguyen, Kiet",
  editor="Qiu, Han
  and Zhang, Cheng
  and Fei, Zongming
  and Qiu, Meikang
  and Kung, Sun-Yuan",
  title="SA2SL: From Aspect-Based Sentiment Analysis to Social Listening System for Business Intelligence",
  booktitle="Knowledge Science, Engineering and Management ",
  year="2021",
  publisher="Springer International Publishing",
  address="Cham",
  pages="647--658",
  isbn="978-3-030-82147-0"
  }
"""

_DATASETNAME = "uit_visfd"

_DESCRIPTION = """
UIT-ViSFD is the Vietnamese Smartphone Feedback Dataset.
It is an aspect-based sentiment analysis dataset.
It consists of 11,122 human-annotated comments for mobile e-commerce.
"""

_HOMEPAGE = "https://github.com/LuongPhan/UIT-ViSFD"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {_DATASETNAME: "https://github.com/LuongPhan/UIT-ViSFD/raw/main/UIT-ViSFD.zip"}

_SUPPORTED_TASKS = [Tasks.ASPECT_BASED_SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class UITViSFDDataset(datasets.GeneratorBasedBuilder):
    """
    Crawled textual feedback from customers about smartphones on a large e-commerce website in Vietnam.
    The label of the dataset is ten aspects and three polarities.
    Please read the guidelines in the paper for more information.
    We randomly divide the dataset into three sets:
        - Train: 7,786.
        - Dev: 1,112.
        - Test: 2,224.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text_multi",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text_multi",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    _LABELS = [
        "BATTERY#Positive",
        "BATTERY#Neutral",
        "BATTERY#Negative",
        "GENERAL#Positive",
        "GENERAL#Neutral",
        "GENERAL#Negative",
        "CAMERA#Positive",
        "CAMERA#Neutral",
        "CAMERA#Negative",
        "FEATURES#Positive",
        "FEATURES#Neutral",
        "FEATURES#Negative",
        "PRICE#Positive",
        "PRICE#Neutral",
        "PRICE#Negative",
        "SER&ACC#Positive",
        "SER&ACC#Neutral",
        "SER&ACC#Negative",
        "PERFORMANCE#Positive",
        "PERFORMANCE#Neutral",
        "PERFORMANCE#Negative",
        "SCREEN#Positive",
        "SCREEN#Neutral",
        "SCREEN#Negative",
        "DESIGN#Positive",
        "DESIGN#Neutral",
        "DESIGN#Negative",
        "STORAGE#Positive",
        "STORAGE#Neutral",
        "STORAGE#Negative",
        "OTHERS",
    ]

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {"index": datasets.Value("int64"), "comment": datasets.Value("string"), "n_star": datasets.Value("int64"), "date_time": datasets.Value("string"), "label": datasets.Sequence(feature=datasets.ClassLabel(names=self._LABELS))}
            )

        elif self.config.schema == "seacrowd_text_multi":
            features = schemas.text_multi_features(self._LABELS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        data_dir = dl_manager.download_and_extract(_URLS[_DATASETNAME])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "Train.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "Test.csv"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "Dev.csv"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = pd.read_csv(filepath, index_col=None)

        def transform_label(label_string):
            label_string = label_string.strip("{}")
            label_pairs = label_string.split(";")
            label_array = []
            for pair in label_pairs:
                pair = pair.strip("{}")
                if pair:
                    label_array.append(pair)
            return label_array

        df["label"] = df["label"].apply(transform_label)

        for index, row in df.iterrows():

            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == "seacrowd_text_multi":

                example = {
                    "id": str(row["index"]),
                    "text": str(row["comment"]),
                    "labels": row["label"],
                }

            yield index, example
