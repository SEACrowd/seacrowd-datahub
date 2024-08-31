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

"""\
This dataset contains 10,510 examples of product and service reviews in Taglish
from Google Maps Reviews and Shopee Philippines. Reviews are manually labeled by
three human annotators according to four sentiment classes: Positive, Negative, Neutral, and Mixed.
"""
import csv
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

# no citation found for this dataset
_CITATION = ""
_DATASETNAME = "sentiment_taglish_product_review"
_DESCRIPTION = """\
Sentiment-Annotated Taglish Product and Service Reviews (SentiTaglish: Products and
Services) is a gold standard, sentiment-annotated corpus for the Tagalog-English
language pair. It contains 10,510 product and service reviews which were manually
labeled by three human annotators according to four sentiment classes:
Positive, Negative, Neutral, and Mixed.
"""

_HOMEPAGE = "https://huggingface.co/datasets/ccosme/SentiTaglishProductsAndServices"

_LANGUAGES = ["tgl", "eng"]
_LICENSE = Licenses.CC_BY_NC_SA_4_0.value
_LOCAL = False
_URLS = {
    _DATASETNAME: "https://huggingface.co/datasets/ccosme/SentiTaglishProductsAndServices/resolve/main/SentiTaglish_ProductsAndServices.csv",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class SentimentTaglishProductReviewDataset(datasets.GeneratorBasedBuilder):
    """A sentiment-annotated corpus comprised of product/service reviews
    in Tagalog-English (Taglish)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "text"
    LABEL_CLASSES = [str(i) for i in range(1, 5)]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"review": datasets.Value("string"), "sentiment": datasets.features.ClassLabel(names=self.LABEL_CLASSES)})

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(self.LABEL_CLASSES)

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

    def _generate_examples(self, filepath: Path, split: str) -> Generator[Tuple[int, Dict], None, None]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
            )
            # skip first row
            next(csv_reader)
            for id_, row in enumerate(csv_reader):
                review, sentiment = row
                if self.config.schema == "source":
                    yield id_, {"review": review, "sentiment": sentiment}
                elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                    yield id_, {"id": id_, "text": review, "label": sentiment}
