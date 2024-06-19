from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{flores-radev-2022-look,
    title = "Look Ma, Only 400 Samples! Revisiting the Effectiveness of Automatic N-Gram Rule Generation for Spelling Normalization in {F}ilipino",
    author = "Flores, Lorenzo Jaime  and
      Radev, Dragomir",
    booktitle = "Proceedings of The Third Workshop on Simple and Efficient Natural Language Processing (SustaiNLP)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.sustainlp-1.5",
    pages = "29--35",
}
"""

_LOCAL = False
_LANGUAGES = ["fil"]
_DATASETNAME = "filipino_slang_norm"
_DESCRIPTION = """\
This dataset contains 398 abbreviated and/or contracted Filipino words used in
Facebook comments made on weather advisories from a Philippine weather bureau.
volunteers.
"""

_HOMEPAGE = "https://github.com/ljyflores/efficient-spelling-normalization-filipino"
_LICENSE = Licenses.UNKNOWN.value
_URLS = {
    "train": "https://github.com/ljyflores/efficient-spelling-normalization-filipino/raw/main/data/train_words.csv",
    "test": "https://github.com/ljyflores/efficient-spelling-normalization-filipino/raw/main/data/test_words.csv",
}

_SUPPORTED_TASKS = [Tasks.MULTILEXNORM]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class FilipinoSlangNormDataset(datasets.GeneratorBasedBuilder):
    """Filipino Slang Norm dataset by Flores and Radev (2022)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "t2t"

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
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "src_sent": datasets.Value("string"),
                    "norm_sent": datasets.Value("string"),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text2text_features
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_files = {
            "train": Path(dl_manager.download_and_extract(_URLS["train"])),
            "test": Path(dl_manager.download_and_extract(_URLS["test"])),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_files["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""
        with open(filepath, encoding="utf-8") as f:
            for guid, line in enumerate(f):
                src_sent, norm_sent = line.strip("\n").split(",")
                if self.config.schema == "source":
                    example = {
                        "id": str(guid),
                        "src_sent": src_sent,
                        "norm_sent": norm_sent,
                    }
                elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                    example = {
                        "id": str(guid),
                        "text_1": src_sent,
                        "text_2": norm_sent,
                        "text_1_name": "src_sent",
                        "text_2_name": "norm_sent",
                    }
                yield guid, example
