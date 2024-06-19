from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@incollection{nguyen2021vietnamese,
  title={Vietnamese Complaint Detection on E-Commerce Websites},
  author={Nguyen, Nhung Thi-Hong and Ha, Phuong Phan-Dieu and Nguyen, Luan Thanh and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy},
  booktitle={New Trends in Intelligent Software Methodologies, Tools and Techniques},
  pages={618--629},
  year={2021},
  publisher={IOS Press}
}
"""

_DATASETNAME = "uit_viocd"

_DESCRIPTION = """\
The UIT-ViOCD dataset includes 5,485 reviews e-commerce sites across four categories: fashion, cosmetics, applications,
and phones. Each review is annotated by humans, assigning a label of 1 for complaints and 0 for non-complaints.
The dataset is divided into training, validation, and test sets, distributed approximately in an 80:10:10 ratio.
"""

_HOMEPAGE = "https://huggingface.co/datasets/tarudesu/ViOCD"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "train": "https://huggingface.co/datasets/tarudesu/ViOCD/resolve/main/train.csv?download=true",
    "val": "https://huggingface.co/datasets/tarudesu/ViOCD/resolve/main/val.csv?download=true",
    "test": "https://huggingface.co/datasets/tarudesu/ViOCD/resolve/main/test.csv?download=true",
}

_SUPPORTED_TASKS = [Tasks.COMPLAINT_DETECTION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class UITVIOCDDataset(datasets.GeneratorBasedBuilder):
    """The UIT-ViOCD dataset includes 5,485 reviews e-commerce sites across four categories: fashion, cosmetics, applications, and phones."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    LABEL_CLASSES = [1, 0]

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

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "review": datasets.Value("string"),
                    "review_tokenize": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=self.LABEL_CLASSES),
                    "domain": datasets.Value("string"),
                }
            )

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
        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["val"],
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        df = pd.read_csv(filepath)

        if self.config.schema == "source":
            for key, example in df.iterrows():
                yield key, {
                    "review": example["review"],
                    "review_tokenize": example["review_tokenize"],
                    "label": example["label"],
                    "domain": example["domain"],
                }

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for key, example in df.iterrows():
                yield key, {"id": str(key), "text": str(example["review"]), "label": int(example["label"])}
