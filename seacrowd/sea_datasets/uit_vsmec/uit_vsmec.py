# coding=utf-8
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{ho2020emotion,
  title={Emotion recognition for vietnamese social media text},
  author={Ho, Vong Anh and Nguyen, Duong Huynh-Cong and Nguyen, Danh Hoang and Pham, Linh Thi-Van and Nguyen, Duc-Vu and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy},
  booktitle={Computational Linguistics: 16th International Conference of the Pacific Association for Computational Linguistics, PACLING 2019, Hanoi, Vietnam, October 11--13, 2019, Revised Selected Papers 16},
  pages={319--333},
  year={2020},
  organization={Springer}
}
"""

_DATASETNAME = "uit_vsmec"

_DESCRIPTION = """\
This dataset consists of Vietnamese Facebook comments that were manually annotated for sentiment.
There are seven possible emotion labels: enjoyment, sadness, fear, anger, disgust, surprise or other (for comments with no or neutral emotions).
Two rounds of manual annotations were done to train annotators with tagging and editing guidelines.
Annotation was performed until inter-annotator agreement reached at least 80%.
"""

_HOMEPAGE = "https://drive.google.com/drive/folders/1HooABJyrddVGzll7fgkJ6VzkG_XuWfRu"

_LICENSE = Licenses.UNKNOWN.value

_LANGUAGES = ["vie"]

_LOCAL = False

_URLS = {
    "train": "https://docs.google.com/spreadsheets/export?id=10VYzfK7JLg-vfmqH0UmKX62z_uaXU-Hp&format=csv",
    "valid": "https://docs.google.com/spreadsheets/export?id=1EsSFZ94fj2yTvFKO6EyxM0wBRcG0s1KE&format=csv",
    "test": "https://docs.google.com/spreadsheets/export?id=1D16FCKKgJ0T6t2aSA3biWVwvD9fa4G9a&format=csv",
}

_SUPPORTED_TASKS = [Tasks.EMOTION_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class UITVSMECDataset(datasets.GeneratorBasedBuilder):
    """
    This is the main class of SEACrowd dataloader for UIT-VSMEC, focusing on emotion/sentiment classification task.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}",
        ),
    ]
    LABEL_NAMES = ["Other", "Disgust", "Enjoyment", "Anger", "Surprise", "Sadness", "Fear"]
    DEFAULT_CONFIG_NAME = "uit_vsmec_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"Emotion": datasets.Value("string"), "Sentence": datasets.Value("string")})

        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(self.LABEL_NAMES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        path_dict = dl_manager.download_and_extract(_URLS)
        train_path, valid_path, test_path = path_dict["train"], path_dict["valid"], path_dict["test"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": valid_path,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        df = pd.read_csv(filepath).reset_index()
        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {"Emotion": row.Emotion, "Sentence": row.Sentence}
                yield row.index, ex

        elif self.config.schema == "seacrowd_text":
            for row in df.itertuples():
                ex = {"id": str(row.index), "text": row.Sentence, "label": self.LABEL_NAMES.index(row.Emotion)}
                yield row.index, ex
