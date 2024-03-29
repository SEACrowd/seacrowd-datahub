import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
    @misc{rtatman2017hse_thai,
        author = {Rachel Tatman},
        title = {HSE Thai Corpus},
        howpublished = {\\url{https://www.kaggle.com/datasets/rtatman/hse-thai-corpus}},
        note = {Accessed: 2023-11-22}
    }
"""

_DATASETNAME = "hse_thai"

_DESCRIPTION = """\
HSE Thai Corpus is a corpus of modern texts written in Thai language. The texts, containing in whole 50 million tokens, were collected from various Thai websites (mostly news websites). To make it easier for non-Thai-speakers to comprehend and use texts in the corpus the researchers decided to separate words in each sentence with spaces. The data for the corpus was collected by means of Scrapy. To tokenize texts the Pythai module was used. The text in this dataset is encoded in UTF-8. This dataset contains text from two sources: Wikipedia and thaigov.go.th. The former is licensed under a standard Wikipedia license, and the latter under an Open Government License for Thailand.

Before running the dataset, please make sure your CLI can run Kaggle API. Guide for installing: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md
"""

_HOMEPAGE = "https://www.kaggle.com/datasets/rtatman/hse-thai-corpus"

_LANGUAGES = ["tha"]

_LICENSE = Licenses.APACHE_2_0.value

_LOCAL = False

_URLS = "rtatman/hse-thai-corpus"

_SUPPORTED_TASKS = [Tasks.LANGUAGE_IDENTIFICATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class HSEThaiDataset(datasets.GeneratorBasedBuilder):
    """Modern Thai corpus taken from https://www.kaggle.com/datasets/rtatman/hse-thai-corpus"""

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

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "article": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(_LANGUAGES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        kaggle_path = _URLS
        os.system(f"kaggle datasets download {kaggle_path}")

        data_dir = dl_manager.extract(f"{os.getcwd()}/hse-thai-corpus.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        base_path = filepath
        if self.config.schema == "source":
            i = -1
            added_text = set()
            for filepath in os.listdir(base_path):
                with open(f"{base_path}/{filepath}", mode="r", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        i += 1
                        if row["text"] in added_text:
                            continue
                        added_text.add(row["text"])
                        yield i, {"article": row["article"], "text": row["text"]}

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            i = -1
            added_text = set()
            for filepath in os.listdir(base_path):
                with open(f"{base_path}/{filepath}", mode="r", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        i += 1
                        if row["text"] in added_text:
                            continue
                        added_text.add(row["text"])
                        yield i, {
                            "id": str(i),
                            "text": row["text"],
                            "label": "tha",
                        }
