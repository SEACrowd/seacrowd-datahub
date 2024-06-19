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
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@misc{ridife2019idsa,
  author = {Fe, Ridi},
  title = {Indonesia Sentiment Analysis Dataset},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/ridife/dataset-idsa}}
}
"""

_DATASETNAME = "id_sentiment_analysis"

_DESCRIPTION = """\
This dataset consists of 10806 labeled Indonesian tweets with their corresponding sentiment analysis: positive, negative, and neutral, up to 2019.
This dataset was developed in Cloud Experience Research Group, Gadjah Mada University.
There is no further explanation of the dataset. Contributor found this dataset after skimming through "Sentiment analysis of Indonesian datasets based on a hybrid deep-learning strategy" (Lin CH and Nuha U, 2023).
"""

_HOMEPAGE = "https://ridi.staff.ugm.ac.id/2019/03/06/indonesia-sentiment-analysis-dataset/"

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/ridife/dataset-idsa/master/Indonesian%20Sentiment%20Twitter%20Dataset%20Labeled.csv",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class IdSentimentAnalysis(datasets.GeneratorBasedBuilder):
    """This dataset consists of 10806 labeled Indonesian tweets with their corresponding sentiment analysis: positive, negative, and neutral, up to 2019."""

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
    ]

    seacrowd_schema_config: List[SEACrowdConfig] = []

    for seacrowd_schema in _SUPPORTED_SCHEMA_STRINGS:

        seacrowd_schema_config.append(
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{seacrowd_schema}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {seacrowd_schema} schema",
                schema=f"{seacrowd_schema}",
                subset_id=f"{_DATASETNAME}",
            )
        )

    BUILDER_CONFIGS.extend(seacrowd_schema_config)

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sentimen": datasets.Value("int32"),
                    "tweet": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SENTIMENT_ANALYSIS]).lower()}":
            features = schemas.text_features(label_names=[1, -1, 0])

        else:
            raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        path = dl_manager.download_and_extract(_URLS[_DATASETNAME])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": path,
                },
            ),
        ]

    def _generate_examples(self, path: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        idx = 0

        if self.config.schema == "source":
            df = pd.read_csv(path, delimiter="\t")

            df.rename(columns={"Tweet": "tweet"}, inplace=True)

            for _, row in df.iterrows():
                yield idx, row.to_dict()
                idx += 1

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SENTIMENT_ANALYSIS]).lower()}":
            df = pd.read_csv(path, delimiter="\t")

            df["id"] = df.index
            df.rename(columns={"sentimen": "label"}, inplace=True)
            df.rename(columns={"Tweet": "text"}, inplace=True)

            for _, row in df.iterrows():
                yield idx, row.to_dict()
                idx += 1

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
