import os
import csv
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

_SUPPORTED_TASKS = [Tasks.ASPECT_BASED_SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class WongnaiReviews(datasets.GeneratorBasedBuilder):
    """WongnaiReviews consists reviews for over 200,000 restaurants, beauty salons, and spas across Thailand."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="wongnai_reviews_source",
            version=SOURCE_VERSION,
            description="wongnai_reviews source schema",
            schema="source",
            subset_id="wongnai_reviews",
        ),
        SEACrowdConfig(
            name="wongnai_reviews_seacrowd_text_multi",
            version=SEACROWD_VERSION,
            description="wongnai_reviews SEACrowd schema",
            schema="seacrowd_text_multi",
            subset_id="wongnai_reviews",
        ),
    ]

    DEFAULT_CONFIG_NAME = "wongnai_reviews_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "review_body": datasets.Value("string"),
                    "star_rating": datasets.ClassLabel(names=["1", "2", "3", "4", "5"]),
                }
            )

        elif self.config.schema == "seacrowd_text_multi":
            features = schemas.text_multi_features(["1", "2", "3", "4", "5"])

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

        # import ipdb
        # ipdb.set_trace()

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

        elif self.config.schema == "seacrowd_text_multi":
            with open(filepath, encoding="utf-8") as f:
                spamreader = csv.reader(f, delimiter=";", quotechar='"')
                for i, row in enumerate(spamreader):
                    yield i, {"id": str(i), "text": row[0], "labels": [row[1]]}
