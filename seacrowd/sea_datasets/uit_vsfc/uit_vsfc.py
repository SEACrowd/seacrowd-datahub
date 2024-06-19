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
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{van2018uit,
  title={UIT-VSFC: Vietnamese students’ feedback corpus for sentiment analysis},
  author={Van Nguyen, Kiet and Nguyen, Vu Duc and Nguyen, Phu XV and Truong, Tham TH and Nguyen, Ngan Luu-Thuy},
  booktitle={2018 10th international conference on knowledge and systems engineering (KSE)},
  pages={19--24},
  year={2018},
  organization={IEEE}
}
"""


_DATASETNAME = "uit_vsfc"

_DESCRIPTION = """\
This corpus consists of student feedback obtained from end-of-semester surveys at a Vietnamese university.
Feedback is classified into four possible topics: lecturer, curriculum, facility or others.
Feedback is also labeled as one of three sentiment polarities: positive, negative or neutral.
"""

_HOMEPAGE = "https://drive.google.com/drive/folders/1HooABJyrddVGzll7fgkJ6VzkG_XuWfRu"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False


_URLS = {
    "train": {
        "sentences": "https://drive.google.com/uc?id=1nzak5OkrheRV1ltOGCXkT671bmjODLhP&export=download",
        "sentiments": "https://drive.google.com/uc?id=1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv&export=download",
        "topics": "https://drive.google.com/uc?id=14MuDtwMnNOcr4z_8KdpxprjbwaQ7lJ_C&export=download",
    },
    "validation": {
        "sentences": "https://drive.google.com/uc?id=1sMJSR3oRfPc3fe1gK-V3W5F24tov_517&export=download",
        "sentiments": "https://drive.google.com/uc?id=1GiY1AOp41dLXIIkgES4422AuDwmbUseL&export=download",
        "topics": "https://drive.google.com/uc?id=1DwLgDEaFWQe8mOd7EpF-xqMEbDLfdT-W&export=download",
    },
    "test": {
        "sentences": "https://drive.google.com/uc?id=1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n&export=download",
        "sentiments": "https://drive.google.com/uc?id=1vkQS5gI0is4ACU58-AbWusnemw7KZNfO&export=download",
        "topics": "https://drive.google.com/uc?id=1_ArMpDguVsbUGl-xSMkTF_p5KpZrmpSB&export=download",
    },
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS, Tasks.TOPIC_MODELING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class UITVSFCDataset(datasets.GeneratorBasedBuilder):
    """This corpus consists of student feedback obtained from end-of-semester surveys at a Vietnamese university.
    Feedback is classified into four possible topics: lecturer, curriculum, facility or others.
    Feedback is also labeled as one of three sentiment polarities: positive, negative or neutral."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SENTIMENT_LABEL_CLASSES = ["positive", "negative", "neutral"]
    TOPIC_LABEL_CLASSES = ["lecturer", "training_program", "others", "facility"]

    SEACROWD_SCHEMA_NAME = "text"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_sentiment_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_topic_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_sentiment_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_topic_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sentence": datasets.Value("string"),
                    "sentiment": datasets.ClassLabel(names=self.SENTIMENT_LABEL_CLASSES),
                    "topic": datasets.ClassLabel(names=self.TOPIC_LABEL_CLASSES),
                }
            )
        elif self.config.name == f"{_DATASETNAME}_sentiment_seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(self.SENTIMENT_LABEL_CLASSES)
        elif self.config.name == f"{_DATASETNAME}_topic_seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(self.TOPIC_LABEL_CLASSES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data_dir = dl_manager.download(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "sentences_path": data_dir["train"]["sentences"],
                    "sentiments_path": data_dir["train"]["sentiments"],
                    "topics_path": data_dir["train"]["topics"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "sentences_path": data_dir["test"]["sentences"],
                    "sentiments_path": data_dir["test"]["sentiments"],
                    "topics_path": data_dir["test"]["topics"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "sentences_path": data_dir["validation"]["sentences"],
                    "sentiments_path": data_dir["validation"]["sentiments"],
                    "topics_path": data_dir["validation"]["topics"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, sentences_path: Path, sentiments_path: Path, topics_path: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(sentences_path, encoding="utf-8") as sentences, open(sentiments_path, encoding="utf-8") as sentiments, open(topics_path, encoding="utf-8") as topics:
                for key, (sentence, sentiment, topic) in enumerate(zip(sentences, sentiments, topics)):
                    yield key, {
                        "sentence": sentence.strip(),
                        "sentiment": int(sentiment.strip()),
                        "topic": int(topic.strip()),
                    }

        elif self.config.name == f"{_DATASETNAME}_sentiment_seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            with open(sentences_path, encoding="utf-8") as sentences, open(sentiments_path, encoding="utf-8") as sentiments:
                for key, (sentence, sentiment) in enumerate(zip(sentences, sentiments)):
                    yield key, {"id": str(key), "text": sentence.strip(), "label": int(sentiment.strip())}
        elif self.config.name == f"{_DATASETNAME}_topic_seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            with open(sentences_path, encoding="utf-8") as sentences, open(topics_path, encoding="utf-8") as topics:
                for key, (sentence, topic) in enumerate(zip(sentences, topics)):
                    yield key, {
                        "id": str(key),
                        "text": sentence.strip(),
                        "label": int(topic.strip()),
                    }
