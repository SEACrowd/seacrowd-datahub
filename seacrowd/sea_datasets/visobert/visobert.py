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
@inproceedings{nguyen-etal-2023-visobert,
    title = "{V}i{S}o{BERT}: A Pre-Trained Language Model for {V}ietnamese Social Media Text Processing",
    author = "Nguyen, Nam  and
      Phan, Thang  and
      Nguyen, Duc-Vu  and
      Nguyen, Kiet",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.315",
    pages = "5191--5207",
    abstract = "English and Chinese, known as resource-rich languages, have witnessed the strong
    development of transformer-based language models for natural language processing tasks. Although
    Vietnam has approximately 100M people speaking Vietnamese, several pre-trained models, e.g., PhoBERT,
    ViBERT, and vELECTRA, performed well on general Vietnamese NLP tasks, including POS tagging and
    named entity recognition. These pre-trained language models are still limited to Vietnamese social
    media tasks. In this paper, we present the first monolingual pre-trained language model for
    Vietnamese social media texts, ViSoBERT, which is pre-trained on a large-scale corpus of high-quality
    and diverse Vietnamese social media texts using XLM-R architecture. Moreover, we explored our
    pre-trained model on five important natural language downstream tasks on Vietnamese social media
    texts: emotion recognition, hate speech detection, sentiment analysis, spam reviews detection, and
    hate speech spans detection. Our experiments demonstrate that ViSoBERT, with far fewer parameters,
    surpasses the previous state-of-the-art models on multiple Vietnamese social media tasks. Our
    ViSoBERT model is available only for research purposes. Disclaimer: This paper contains actual
    comments on social networks that might be construed as abusive, offensive, or obscene.",
}
"""

_DATASETNAME = "visobert"

_DESCRIPTION = """\
The ViSoBERT corpus is composed of Vietnamese textual data crawled from Facebook, TikTok, and YouTube. The
dataset contains Facebook posts, TikTok comments, and Youtube comments of Vietnamese-verified users, from
Jan 2016 (Jan 2020 for TikTok) to Dec 2022. A post-processing mechanism is applied to handles hashtags,
emojis, misspellings, hyperlinks, and other noncanonical texts.
"""

_HOMEPAGE = "https://huggingface.co/uitnlp/visobert"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_NC_4_0.value

_LOCAL = False

_URLS = "https://drive.usercontent.google.com/download?id=1BoiR9k2DrjBcd2aHy5BOq4haEp5V2_ug&confirm=xxx"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class ViSoBERTDataset(datasets.GeneratorBasedBuilder):
    """
    The ViSoBERT corpus is a Vietnamese pretraining dataset from https://huggingface.co/uitnlp/visobert.
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
            name=f"{_DATASETNAME}_seacrowd_ssp",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source" or self.config.schema == "seacrowd_ssp":
            features = schemas.self_supervised_pretraining.features
        else:
            raise ValueError(f"Invalid schema: '{self.config.schema}'")

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

        path = dl_manager.download(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """

        with open(filepath, "r", encoding="utf-8") as f:
            if self.config.schema == "source" or self.config.schema == "seacrowd_ssp":
                for idx, row in enumerate(f):
                    if row.strip() != "":
                        yield (
                            idx,
                            {
                                "id": str(idx),
                                "text": row.strip(),
                            },
                        )
            else:
                raise ValueError(f"Invalid config: '{self.config.name}'")