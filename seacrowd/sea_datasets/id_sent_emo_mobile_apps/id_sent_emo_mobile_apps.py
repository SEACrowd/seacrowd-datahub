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
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@article{riccosan2023,
  author    = {Riccosan and Saputra, Karen Etania},
  title     = {Multilabel multiclass sentiment and emotion dataset from indonesian mobile application review},
  journal   = {Data in Brief},
  volume    = {50},
  year      = {2023},
  doi       = {10.1016/j.dib.2023.109576},
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]
_DATASETNAME = "id_sent_emo_mobile_apps"
_DESCRIPTION = """
This dataset contains manually annotated public reviews of mobile applications in Indonesia.
Each review is given a sentiment label (positive, negative, neutral) and
an emotion label (anger, sadness, fear, happiness, love, neutral).
"""
_HOMEPAGE = "https://github.com/Ricco48/Multilabel-Sentiment-and-Emotion-Dataset-from-Indonesian-" "Mobile-Application-Review/tree/CreateCodeForPaper"
_LICENSE = Licenses.CC_BY_NC_ND_4_0.value
_URL = (
    "https://github.com/Ricco48/Multilabel-Sentiment-and-Emotion-Dataset-from-Indonesian-Mobile-Application-Review/raw/CreateCodeForPaper/"
    "Multilabel%20Sentiment%20and%20Emotion%20Dataset%20from%20Indonesian%20Mobile%20Application%20Review/Multilabel%20Sentiment%20and%20Emotion"
    "%20Dataset%20from%20Indonesian%20Mobile%20Application%20Review.csv"
)

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS, Tasks.EMOTION_CLASSIFICATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class EmoSentIndMobile(datasets.GeneratorBasedBuilder):
    """Dataset of Indonesian mobile application reviews manually annotated for emotion and sentiment."""

    SUBSETS = ["emotion", "sentiment"]
    EMOTION_CLASS_LABELS = ["Sad", "Anger", "Fear", "Happy", "Love", "Neutral"]
    SENTIMENT_CLASS_LABELS = ["Negative", "Positive", "Neutral"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME
        )
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

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "content": datasets.Value("string"),
                    "sentiment": datasets.Value("string"),
                    "emotion": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_text":
            if "emotion" in self.config.subset_id:
                labels = self.EMOTION_CLASS_LABELS
            elif "sentiment" in self.config.subset_id:
                labels = self.SENTIMENT_CLASS_LABELS
            features = schemas.text_features(label_names=labels)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        fp = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": fp},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(filepath, sep="\t", index_col=None)
        for index, row in df.iterrows():
            if self.config.schema == "source":
                example = {
                    "content": row["content"],
                    "sentiment": row["Sentiment"].title(),
                    "emotion": row["Emotion"].title(),
                }
            elif self.config.schema == "seacrowd_text":
                if "emotion" in self.config.subset_id:
                    label = row["Emotion"]
                elif "sentiment" in self.config.subset_id:
                    label = row["Sentiment"]
                example = {"id": str(index), "text": row["content"], "label": label.title()}
            yield index, example
