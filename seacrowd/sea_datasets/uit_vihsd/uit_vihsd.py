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
@InProceedings{10.1007/978-3-030-79457-6_35,
author="Luu, Son T.
and Nguyen, Kiet Van
and Nguyen, Ngan Luu-Thuy",
editor="Fujita, Hamido
and Selamat, Ali
and Lin, Jerry Chun-Wei
and Ali, Moonis",
title="A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts",
booktitle="Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="415--426",
abstract="In recent years, Vietnam witnesses the mass development of social network users on different social
platforms such as Facebook, Youtube, Instagram, and Tiktok. On social media, hate speech has become a critical
problem for social network users. To solve this problem, we introduce the ViHSD - a human-annotated dataset for
automatically detecting hate speech on the social network. This dataset contains over 30,000 comments, each comment
in the dataset has one of three labels: CLEAN, OFFENSIVE, or HATE. Besides, we introduce the data creation process
for annotating and evaluating the quality of the dataset. Finally, we evaluate the dataset by deep learning and transformer models.",
isbn="978-3-030-79457-6"
}
"""

_LOCAL = False
_LANGUAGES = ["vie"]
_DATASETNAME = "uit_vihsd"
_DESCRIPTION = """
The ViHSD dataset consists of comments collected from Facebook pages and YouTube channels that have a
high-interactive rate, and do not restrict comments. This dataset is used for hate speech detection on
Vietnamese language. Data is anonymized, and labeled as either HATE, OFFENSIVE, or CLEAN.
"""

_HOMEPAGE = "https://github.com/sonlam1102/vihsd/"
_LICENSE = Licenses.UNKNOWN.value
_URL = "https://raw.githubusercontent.com/sonlam1102/vihsd/main/data/vihsd.zip"

_Split_Path = {
    "train": "vihsd/train.csv",
    "validation": "vihsd/dev.csv",
    "test": "vihsd/test.csv",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class UiTVihsdDataset(datasets.GeneratorBasedBuilder):
    """
    The SeaCrowd dataloader for the dataset Vietnamese Hate Speech Detection (UIT-ViHSD).
    """

    CLASS_LABELS = ["CLEAN", "OFFENSIVE", "HATE"]  # 0:CLEAN, 1:OFFENSIVE, 2:HATE
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
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
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
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
        for row in data_lines.itertuples():
            if self.config.schema == "source":
                example = {"id": str(row.Index), "text": row.free_text, "label": row.label_id}
            if self.config.schema == "seacrowd_text":
                example = {"id": str(row.Index), "text": row.free_text, "label": self.CLASS_LABELS[int(row.label_id)]}
            yield row.Index, example

