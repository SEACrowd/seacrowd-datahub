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

import json
import os

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (DEFAULT_SEACROWD_VIEW_NAME,
                                      DEFAULT_SOURCE_VIEW_NAME, Licenses,
                                      Tasks)

_CITATION = """\
@software{bact_2019_3457447,
  author       = {Suriyawongkul, Arthit and
                  Chuangsuwanich, Ekapol and
                  Chormai, Pattarawat and
                  Polpanumas, Charin},
  title        = {PyThaiNLP/wisesight-sentiment: First release},
  month        = sep,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.3457447},
  url          = {https://doi.org/10.5281/zenodo.3457447}
}
"""


_DATASETNAME = "wisesight_thai_sentiment"


_DESCRIPTION = """\
Wisesight Sentiment Corpus: Social media messages in Thai language with sentiment category (positive, neutral, negative, question)
* Released to public domain under Creative Commons Zero v1.0 Universal license.
* Category (Labels): {"pos": 0, "neu": 1, "neg": 2, "q": 3}
* Size: 26,737 messages
* Language: Central Thai
* Style: Informal and conversational. With some news headlines and advertisement.
* Time period: Around 2016 to early 2019. With small amount from other period.
* Domains: Mixed. Majority are consumer products and services (restaurants, cosmetics, drinks, car, hotels), with some current affairs.
* Privacy:
    * Only messages that made available to the public on the internet (websites, blogs, social network sites).
    * For Facebook, this means the public comments (everyone can see) that made on a public page.
    * Private/protected messages and messages in groups, chat, and inbox are not included.
* Alternations and modifications:
    * Keep in mind that this corpus does not statistically represent anything in the language register.
    * Large amount of messages are not in their original form. Personal data are removed or masked.
    * Duplicated, leading, and trailing whitespaces are removed. Other punctuations, symbols, and emojis are kept intact.
    (Mis)spellings are kept intact.
    * Messages longer than 2,000 characters are removed.
    * Long non-Thai messages are removed. Duplicated message (exact match) are removed.
* More characteristics of the data can be explore: https://github.com/PyThaiNLP/wisesight-sentiment/blob/master/exploration.ipynb
"""

_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME

_HOMEPAGE = "https://github.com/PyThaiNLP/wisesight-sentiment"

_LANGUAGES = ["tha"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)


_LICENSE = Licenses.CC0_1_0.value


_LOCAL = False


_URLS = {
    _DATASETNAME: "https://github.com/PyThaiNLP/wisesight-sentiment/raw/master/huggingface/data.zip",
}


_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]


_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class WisesightSentimentDataset(datasets.GeneratorBasedBuilder):
    """Wisesight Sentiment Corpus: Social media messages in Thai language with sentiment category (positive, neutral, negative, question)"""

    _DOWNLOAD_URL = _URLS[_DATASETNAME]
    _TRAIN_FILE = "train.jsonl"
    _VAL_FILE = "valid.jsonl"
    _TEST_FILE = "test.jsonl"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="wisesight_thai_sentiment_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Wisesight Sentiment Corpus Source version (positive, neutral, negative, question)",
            schema="source",
            subset_id="wisesight_thai_sentiment",
        ),
        SEACrowdConfig(
            name="wisesight_thai_sentiment_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description="Wisesight Sentiment Corpus Seacrowd version (positive, neutral, negative, question)",
            schema="seacrowd_text",
            subset_id="wisesight_thai_sentiment",
        ),
    ]

    DEFAULT_CONFIG_NAME = "wisesight_thai_sentiment_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "texts": datasets.Value("string"),
                    "category": datasets.features.ClassLabel(names=["pos", "neu", "neg", "q"]),
                }
            )
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(["pos", "neu", "neg", "q"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        arch_path = dl_manager.download_and_extract(self._DOWNLOAD_URL)
        data_dir = os.path.join(arch_path, "data")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, self._TRAIN_FILE)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, self._VAL_FILE)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, self._TEST_FILE)},
            ),
        ]

    def _generate_examples(self, filepath):
        """Generate WisesightSentiment examples."""
        with open(filepath, encoding="utf-8") as f:
            if self.config.schema == "source":
                for id_, row in enumerate(f):
                    data = json.loads(row)
                    texts = data["texts"]
                    category = data["category"]
                    yield id_, {"texts": texts, "category": category}

            elif self.config.schema == "seacrowd_text":
                for id_, row in enumerate(f):
                    data = json.loads(row)
                    texts = data["texts"]
                    category = data["category"]
                    ex = {"id": str(id_), "text": texts, "label": category}
                    yield id_, ex
