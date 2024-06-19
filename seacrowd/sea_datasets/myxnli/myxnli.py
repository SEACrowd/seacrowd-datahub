from pathlib import Path

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks


_CITATION = """
@misc{myXNLI2023,
    title = "myXNLI",
    author = "akhtet",
    year = "202",
    url = "https://github.com/akhtet/myXNLI",
}
"""

_DATASETNAME = "myxnli"

_DESCRIPTION = """
The myXNLI corpus is a collection of Myanmar language data designed for the Natural Language Inference (NLI) task, which
originated from the XNLI and MultiNLI English datasets. The 7,500 sentence pairs from the XNLI English development and
test sets are human-translated into Myanmar. The 392,702 data from the NLI English training data is translated using
machine translation. In addition, it also extends its scope by adding Myanmar translations to the XNLI 15-language
parallel corpus, to create a 16-language parallel corpus.
"""

_HOMEPAGE = "https://github.com/akhtet/myXNLI"

_LANGUAGES = ["mya"]

_LICENSE = Licenses.CC_BY_NC_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "train": "https://huggingface.co/datasets/akhtet/myXNLI/resolve/main/data/train-00000-of-00001-2614419e00195781.parquet",
        "dev": "https://huggingface.co/datasets/akhtet/myXNLI/resolve/main/data/validation-00000-of-00001-9c168eb31d1d810b.parquet",
        "test": "https://huggingface.co/datasets/akhtet/myXNLI/resolve/main/data/test-00000-of-00001-0fd9f93baf8c9cdb.parquet",
    },
}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.1.0"

_SEACROWD_VERSION = "2024.06.20"


class MyXNLIDataset(datasets.GeneratorBasedBuilder):
    """The myXNLI corpus is a collection of Myanmar language data designed for the Natural Language Inference task."""

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
            name=f"{_DATASETNAME}_seacrowd_pairs",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_pairs",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "genre": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=["contradiction", "entailment", "neutral"]),
                    "sentence1_en": datasets.Value("string"),
                    "sentence2_en": datasets.Value("string"),
                    "sentence1_my": datasets.Value("string"),
                    "sentence2_my": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_pairs":
            features = schemas.pairs_features(["contradiction", "entailment", "neutral"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_dir, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir, "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_dir, "split": "dev"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> tuple[int, dict]:
        if self.config.schema == "source":
            df = pd.read_parquet(filepath[split])
            for i, row in df.iterrows():
                yield i, {
                    "genre": row["genre"],
                    "label": row["label"],
                    "sentence1_en": row["sentence1_en"],
                    "sentence2_en": row["sentence2_en"],
                    "sentence1_my": row["sentence1_my"],
                    "sentence2_my": row["sentence2_my"],
                }

        elif self.config.schema == "seacrowd_pairs":
            df = pd.read_parquet(filepath[split])
            for i, row in df.iterrows():
                yield i, {
                    "id": str(i),
                    "text_1": row["sentence1_my"],
                    "text_2": row["sentence2_my"],
                    "label": row["label"],
                }
