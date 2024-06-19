import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

# no BibTeX citation
_CITATION = ""

_DATASETNAME = "wongnai_reviews"

_DESCRIPTION = """
Wongnai features over 200,000 restaurants, beauty salons, and spas across Thailand on its platform, with detailed
information about each merchant and user reviews. Its over two million registered users can search for what’s top rated
in Bangkok, follow their friends, upload photos, and do quick write-ups about the places they visit. Each write-up
(review) also comes with a rating score ranging from 1-5 stars. The task here is to create a rating prediction model
using only textual information.
"""

_HOMEPAGE = "https://huggingface.co/datasets/wongnai_reviews"

_LANGUAGES = ["tha"]

_LICENSE = Licenses.LGPL_3_0.value

_LOCAL = False

_URLS = {_DATASETNAME: "https://archive.org/download/wongnai_reviews/wongnai_reviews_withtest.zip"}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_CLASSES = ["1", "2", "3", "4", "5"]


class WongnaiReviewsDataset(datasets.GeneratorBasedBuilder):
    """WongnaiReviews consists reviews for over 200,000 restaurants, beauty salons, and spas across Thailand."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "review_body": datasets.Value("string"),
                    "star_rating": datasets.ClassLabel(names=_CLASSES),
                }
            )

        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(label_names=_CLASSES)

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
                gen_kwargs={"filepath": os.path.join(data_dir, "w_review_train.csv"), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "w_review_test.csv"), "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        if self.config.schema == "source":
            with open(filepath, encoding="utf-8") as f:
                spamreader = csv.reader(f, delimiter=";", quotechar='"')
                for i, row in enumerate(spamreader):
                    yield i, {"review_body": row[0], "star_rating": row[1]}

        elif self.config.schema == "seacrowd_text":
            with open(filepath, encoding="utf-8") as f:
                spamreader = csv.reader(f, delimiter=";", quotechar='"')
                for i, row in enumerate(spamreader):
                    yield i, {"id": str(i), "text": row[0], "label": _CLASSES[int(row[1].strip()) - 1]}
