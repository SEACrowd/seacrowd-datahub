import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@article{riegoenhancement,
  title={Enhancement to Low-Resource Text Classification via Sequential Transfer Learning},
  author={Riego, Neil Christian R. and Villarba, Danny Bell and Sison, Ariel Antwaun Rolando C. and Pineda, Fernandez C. and Lagunzad, Herminiño C.}
  journal={United International Journal for Research & Technology},
  volume={04},
  issue={08},
  pages={72--82}
}
"""

_LOCAL = False
_LANGUAGES = ["fil", "tgl"]
_DATASETNAME = "shopee_reviews_tagalog"
_DESCRIPTION = """\
The Shopee reviews dataset is constructed by randomly taking 2100 training
samples and 450 samples for testing and validation for each review star from 1
to 5. In total, there are 10500 training samples and 2250 each in validation and
testing samples.
"""

_HOMEPAGE = "https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars"
_LICENSE = Licenses.MPL_2_0.value
_URLS = {
    "train": "https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars/resolve/main/train.csv",
    "validation": "https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars/resolve/main/validation.csv",
    "test": "https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars/resolve/main/test.csv",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ShopeeReviewsTagalogDataset(datasets.GeneratorBasedBuilder):
    """Shopee Reviews Tagalog dataset from https://huggingface.co/datasets/scaredmeow/shopee-reviews-tl-stars"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "text"
    # "N" means N+1 star(s) review, e.g. "2" means 3 stars review
    LABEL_CLASSES = ["0", "1", "2", "3", "4"]

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
        # The SEACrowd schema and the source schema is the same
        features = schemas.text_features(self.LABEL_CLASSES)
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        data_files = {
            "train": Path(dl_manager.download_and_extract(_URLS["train"])),
            "validation": Path(dl_manager.download_and_extract(_URLS["validation"])),
            "test": Path(dl_manager.download_and_extract(_URLS["test"])),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["validation"], "split": "validation"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"], "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yield (idx, example) tuples"""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
            )

            next(csv_reader)
            for idx, row in enumerate(csv_reader):
                text, label = row
                example = {"id": idx, "text": text, "label": label}
                yield idx, example
