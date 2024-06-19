from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@article{hidayatullah2023corpus,
  title={Corpus creation and language identification for code-mixed Indonesian-Javanese-English Tweets},
  author={Hidayatullah, Ahmad Fathan and Apong, Rosyzie Anna and Lai, Daphne TC and Qazi, Atika},
  journal={PeerJ Computer Science},
  volume={9},
  pages={e1312},
  year={2023},
  publisher={PeerJ Inc.}
}
"""

_LOCAL = False
_LANGUAGES = ["ind", "jav", "eng"]
_DATASETNAME = "ijelid"
_DESCRIPTION = """\
This is a code-mixed Indonesian-Javanese-English dataset for token-level
language identification. We named this dataset as IJELID
(Indonesian-Javanese-English Language Identification). This dataset contains
tweets that have been tokenized with the corresponding token and its language
label. There are seven language labels in the dataset, namely: ID (Indonesian)JV
(Javanese), EN (English), MIX_ID_EN (mixed Indonesian-English), MIX_ID_JV (mixed
Indonesian-Javanese), MIX_JV_EN (mixed Javanese-English), OTH (Other).
"""

_HOMEPAGE = "https://github.com/fathanick/Code-mixed-Indonesian-Javanese-English-Twitter-Data"
_LICENSE = Licenses.CC_BY_NC_SA_4_0.value
_URLS = {
    "train": "https://raw.githubusercontent.com/fathanick/Code-mixed-Indonesian-Javanese-English-Twitter-Data/main/train.tsv",
    "dev": "https://raw.githubusercontent.com/fathanick/Code-mixed-Indonesian-Javanese-English-Twitter-Data/main/val.tsv",
    "test": "https://raw.githubusercontent.com/fathanick/Code-mixed-Indonesian-Javanese-English-Twitter-Data/main/test.tsv",
}

_SUPPORTED_TASKS = [Tasks.TOKEN_LEVEL_LANGUAGE_IDENTIFICATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class IJELIDDataset(datasets.GeneratorBasedBuilder):
    """IJELID dataset from https://github.com/fathanick/Code-mixed-Indonesian-Javanese-English-Twitter-Data"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "seq_label"
    LABEL_CLASSES = ["ID", "JV", "EN", "MIX_ID_EN", "MIX_ID_JV", "MIX_JV_EN", "OTH"]

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
        # No specific schema for the source, so for consistency,
        # I will use the same schema with SEACrowd
        features = schemas.seq_label_features(self.LABEL_CLASSES)

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
            "dev": Path(dl_manager.download_and_extract(_URLS["dev"])),
            "test": Path(dl_manager.download_and_extract(_URLS["test"])),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["dev"], "split": "dev"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"], "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            labels = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "labels": labels,
                        }
                        guid += 1
                        tokens = []
                        labels = []
                else:
                    # IJELID TSV are separated by \t
                    token, label = line.split("\t")
                    tokens.append(token)
                    labels.append(label.rstrip())

            # Last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "labels": labels,
                }
