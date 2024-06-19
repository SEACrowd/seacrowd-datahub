import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@article{galinato-etal-2023-context,
    title="Context-Based Profanity Detection and Censorship Using Bidirectional Encoder Representations from Transformers",
    author="Galinato, Valfrid and Amores, Lawrence and Magsino, Gino Ben and Sumawang, David Rafael",
    month="jan",
    year="2023"
    url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4341604"
}
"""

_LOCAL = False
_LANGUAGES = ["tgl"]
_DATASETNAME = "tgl_profanity"
_DESCRIPTION = """\
This dataset contains 13.8k Tagalog sentences containing profane words, together
with binary labels denoting whether or not the sentence conveys profanity /
abuse / hate speech. The data was scraped from Twitter using a Python library
called SNScrape and annotated manually by a panel of native Filipino speakers.
"""

_HOMEPAGE = "https://huggingface.co/datasets/mginoben/tagalog-profanity-dataset/"
_LICENSE = Licenses.UNKNOWN.value
_SUPPORTED_TASKS = [Tasks.ABUSIVE_LANGUAGE_PREDICTION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"
_URLS = {
    "train": "https://huggingface.co/datasets/mginoben/tagalog-profanity-dataset/resolve/main/train.csv",
    "val": "https://huggingface.co/datasets/mginoben/tagalog-profanity-dataset/resolve/main/val.csv",
}


class TagalogProfanityDataset(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "text"

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
    CLASS_LABELS = ["1", "0"]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("int64"),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(label_names=self.CLASS_LABELS)
        else:
            raise ValueError(f"Invalid config name: {self.config.schema}")
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["val"]},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""
        with open(filepath, encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")
            next(csv_reader, None)  # skip the headers
            for idx, row in enumerate(csv_reader):
                text, label = row
                if self.config.schema == "source":
                    example = {"text": text, "label": int(label)}
                elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                    example = {"id": idx, "text": text, "label": int(label)}
                yield idx, example
