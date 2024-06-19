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

_CITATION = """\
@InProceedings{10.1007/978-981-16-8515-6_44,
author="Juan, Sarah Samson
and Saee, Suhaila
and Mohamad, Fitri",
editor="Alfred, Rayner
and Lim, Yuto",
title="Social Versus Physical Distancing: Analysis of Public Health Messages at the Start of COVID-19 Outbreak in Malaysia Using Natural Language Processing",
booktitle="Proceedings of the 8th International Conference on Computational Science and Technology",
year="2022",
publisher="Springer Singapore",
address="Singapore",
pages="577--589",
abstract="The study presents an attempt to analyse how social media netizens in Malaysia responded to the calls for ``Social Distancing'' and ``Physical Distancing'' as the newly recommended social norm was introduced to the world
as a response to the COVID-19 global pandemic. The pandemic drove a sharp increase in social media platforms' use as a public health communication platform since the first wave of the COVID-19 outbreak in Malaysia in April 2020.
We analysed thousands of tweets posted by Malaysians daily between January 2020 and August 2021 to determine public perceptions and interactions patterns. The analysis focused on positive and negative reactions
and the interchanges of uses of the recommended terminologies ``social distancing'' and ``physical distancing''. Using linguistic analysis and natural language processing,
findings dominantly indicate influences from the multilingual and multicultural values held by Malaysian netizens, as they embrace the concept of distancing as a measure of global public health safety.",
isbn="978-981-16-8515-6"
}
"""

_DATASETNAME = "malaysia_tweets"
_DESCRIPTION = """\
This tweet data was extracted from tweets in Malaysia based on keywords
"social distancing" and "physical distancing". We conducted
sentiment analysis to understand public opinions on health messages
during the COVID-19 pandemic. Tweets from January 2020 to July 2021
were extracted using Python module snscrape and sentiments were obtained
automatically using Polyglot and MALAYA NLP tools due to multilingual data.
"""

_HOMEPAGE = "https://github.com/sarahjuan/malaysia-tweets-with-sentiment-labels"

_LANGUAGES = ["zlm,", "eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.UNKNOWN.value  # example: Licenses.MIT.value, Licenses.CC_BY_NC_SA_4_0.value, Licenses.UNLICENSE.value, Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/sarahjuan/malaysia-tweets-with-sentiment-labels/main/data/cleaned_tweets_sentiments.csv",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MalaysiaTweetsDataset(datasets.GeneratorBasedBuilder):
    """This tweet data was extracted from tweets in Malaysia based on keywords
    "social distancing" and "physical distancing" from January 2020 to July 2021."""

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
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"
    SENTIMENT_LABEL_CLASSES = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "Tweet": datasets.Value("string"),
                    "Sentiment": datasets.ClassLabel(names=self.SENTIMENT_LABEL_CLASSES),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(self.SENTIMENT_LABEL_CLASSES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = pd.read_csv(filepath, encoding="utf-8")
        if self.config.schema == "source":
            for idx, row in df.iterrows():
                yield idx, dict(row)

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for idx, row in df.iterrows():
                yield idx, {"id": idx, "text": row["Tweet"], "label": row["Sentiment"]}
