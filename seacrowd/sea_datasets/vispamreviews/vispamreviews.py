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
from typing import Dict, List, Tuple

import datasets
import pandas

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@InProceedings{10.1007/978-3-031-21743-2_48,
author="Van Dinh, Co
and Luu, Son T.
and Nguyen, Anh Gia-Tuan",
editor="Nguyen, Ngoc Thanh
and Tran, Tien Khoa
and Tukayev, Ualsher
and Hong, Tzung-Pei
and Trawi{\'{n}}ski, Bogdan
and Szczerbicki, Edward",
title="Detecting Spam Reviews on Vietnamese E-Commerce Websites",
booktitle="Intelligent Information and Database Systems",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="595--607",
abstract="The reviews of customers play an essential role in online shopping.
People often refer to reviews or comments of previous customers to decide whether
to buy a new product. Catching up with this behavior, some people create untruths and
illegitimate reviews to hoax customers about the fake quality of products. These are called
spam reviews, confusing consumers on online shopping platforms and negatively affecting online
shopping behaviors. We propose the dataset called ViSpamReviews, which has a strict annotation
procedure for detecting spam reviews on e-commerce platforms. Our dataset consists of two tasks:
the binary classification task for detecting whether a review is spam or not and the multi-class
classification task for identifying the type of spam. The PhoBERT obtained the highest results on
both tasks, 86.89%, and 72.17%, respectively, by macro average F1 score.",
isbn="978-3-031-21743-2"
}
"""

_LOCAL = False
_LANGUAGES = ["vie"]
_DATASETNAME = "vispamreviews"
_DESCRIPTION = """
The dataset was collected from leading online shopping platforms in Vietnam. Some of the most recent
selling products for each product category were selected and up to 15 reviews per product were collected.
Each review was then labeled as either NO-SPAM, SPAM-1 (fake review), SPAM-2 (review on brand only), or
SPAM-3 (irrelevant content).
"""

_HOMEPAGE = "https://github.com/sonlam1102/vispamdetection/"
_LICENSE = Licenses.CC_BY_NC_4_0.value
_URL = "https://raw.githubusercontent.com/sonlam1102/vispamdetection/main/dataset/vispamdetection_dataset.zip"

_Split_Path = {
    "train": "dataset/train.csv",
    "validation": "dataset/dev.csv",
    "test": "dataset/test.csv",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]  # Text Classification
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ViSpamReviewsDataset(datasets.GeneratorBasedBuilder):
    """
    The SeaCrowd dataloader for the review dataset shopping platforms in Vietnam (ViSpamReviews).
    """

    CLASS_LABELS = [0, 1]
    SPAM_TYPE_LABELS = [0, 1, 2, 3]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_spam_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema ",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_spam_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema ",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}_spam",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.name.endswith("source"):
            features = (datasets.Features
                (
                {"id": datasets.Value("int32"),
                 "text": datasets.Value("string"),
                 "label": datasets.Value("string"),
                 "spam_label": datasets.Value("string"),
                 "rating": datasets.Value("int32")
                 }
            ))

        elif self.config.name == "vispamreviews_seacrowd_text":
            features = schemas.text_features(label_names=self.CLASS_LABELS)
        elif self.config.name == "vispamreviews_spam_seacrowd_text":
            features = schemas.text_features(label_names=self.SPAM_TYPE_LABELS)
        else:
            raise ValueError(f"Invalid schema {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_paths = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(file_paths, _Split_Path["train"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(file_paths, _Split_Path["validation"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(file_paths, _Split_Path["test"])},
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data_lines = pandas.read_csv(filepath)
        for rid, row in enumerate(data_lines.itertuples()):
            if self.config.name.endswith("source"):
                example = {"id": str(rid), "text": row.Comment, "label": row.Label, "spam_label": row.SpamLabel,
                           "rating": row.Rating}
            elif self.config.name == "vispamreviews_seacrowd_text":
                example = {"id": str(rid), "text": row.Comment, "label": row.Label}
            elif self.config.name == "vispamreviews_spam_seacrowd_text":
                example = {"id": str(rid), "text": row.Comment, "label": row.SpamLabel}
            else:
                raise ValueError(f"Invalid schema {self.config.schema}")
            yield rid, example
