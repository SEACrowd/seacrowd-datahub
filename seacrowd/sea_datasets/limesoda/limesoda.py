import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@INPROCEEDINGS{9678187,
  author={Payoungkhamdee, Patomporn and Porkaew, Peerachet and Sinthunyathum, Atthasith and Songphum, Phattharaphon and Kawidam, Witsarut and Loha-Udom, Wichayut and Boonkwan, Prachya and Sutantayawalee, Vipas},
  booktitle={2021 16th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP)},
  title={LimeSoda: Dataset for Fake News Detection in Healthcare Domain},
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/iSAI-NLP54397.2021.9678187}}
"""

_DATASETNAME = "limesoda"

_DESCRIPTION = """\
Thai fake news dataset in the healthcare domain consisting of curate and manually annotated 7,191 documents
(only 4,141 documents contain token labels and are used as a test set of the baseline models).
Each document in the dataset is classified as fact, fake, or undefined.
"""

_HOMEPAGE = "https://github.com/byinth/LimeSoda"

_LICENSE = Licenses.CC_BY_4_0.value

_LANGUAGES = ["tha"]
_LOCAL = False

_URLS = {
    "split": {
        "train": "https://raw.githubusercontent.com/byinth/LimeSoda/main/dataset_train_wo_tokentags_v1/train_v1.jsonl",
        "val": "https://raw.githubusercontent.com/byinth/LimeSoda/main/dataset_train_wo_tokentags_v1/val_v1.jsonl",
        "test": "https://raw.githubusercontent.com/byinth/LimeSoda/main/dataset_train_wo_tokentags_v1/test_v1.jsonl",
    },
    "raw": "https://raw.githubusercontent.com/byinth/LimeSoda/main/LimeSoda/Limesoda.jsonl",
}

_SUPPORTED_TASKS = [Tasks.HOAX_NEWS_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class LimeSodaDataset(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="limesoda source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_split_source",
            version=SOURCE_VERSION,
            description="limesoda source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_split",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text",
            version=SEACROWD_VERSION,
            description="limesoda SEACrowd schema",
            schema="seacrowd_text",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_split_seacrowd_text",
            version=SEACROWD_VERSION,
            description="limesoda: split SEACrowd schema",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}_split",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            if self.config.subset_id == "limesoda":
                features = datasets.Features(
                    {
                        "id": datasets.Value("string"),
                        "title": datasets.Value("string"),
                        "detail": datasets.Sequence(datasets.Value("string")),
                        "title_token_tags": datasets.Value("string"),
                        "detail_token_tags": datasets.Sequence(datasets.Value("string")),
                        "document_tag": datasets.Value("string"),
                    }
                )
            else:
                features = datasets.Features({"id": datasets.Value("string"), "text": datasets.Value("string"), "document_tag": datasets.Value("string")})
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(["Fact News", "Fake News", "Undefined"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        path_dict = dl_manager.download_and_extract(_URLS)
        if self.config.subset_id == "limesoda":
            raw_path = path_dict["raw"]
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": raw_path,
                    },
                ),
            ]
        elif self.config.subset_id == "limesoda_split":
            train_path, val_path, test_path = path_dict["split"]["train"], path_dict["split"]["val"], path_dict["split"]["test"]
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
                        "filepath": val_path,
                    },
                ),
            ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        with open(filepath, "r") as f:
            entries = [json.loads(line) for line in f.readlines()]
        if self.config.schema == "source":
            if self.config.subset_id == "limesoda":
                for i, row in enumerate(entries):
                    ex = {"id": str(i), "title": row["Title"], "detail": row["Detail"], "title_token_tags": row["Title Token Tags"], "detail_token_tags": row["Detail Token Tags"], "document_tag": row["Document Tag"]}
                    yield i, ex
            else:
                for i, row in enumerate(entries):
                    ex = {"id": str(i), "text": row["Text"], "document_tag": row["Document Tag"]}
                    yield i, ex
        elif self.config.schema == "seacrowd_text":
            for i, row in enumerate(entries):
                ex = {
                    "id": str(i),
                    "text": row["Detail"] if self.config.subset_id == "limesoda" else row["Text"],
                    "label": row["Document Tag"],
                }
                yield i, ex
