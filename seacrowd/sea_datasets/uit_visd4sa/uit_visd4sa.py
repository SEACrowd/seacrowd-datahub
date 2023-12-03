# coding=utf-8

import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{thanh-etal-2021-span,
    title = "Span Detection for Aspect-Based Sentiment Analysis in Vietnamese",
    author = "Thanh, Kim Nguyen Thi  and
      Khai, Sieu Huynh  and
      Huynh, Phuc Pham  and
      Luc, Luong Phan  and
      Nguyen, Duc-Vu  and
      Van, Kiet Nguyen",
    booktitle = "Proceedings of the 35th Pacific Asia Conference on Language, Information and Computation",
    year = "2021",
    publisher = "Association for Computational Lingustics",
    url = "https://aclanthology.org/2021.paclic-1.34",
    pages = "318--328",
}
"""

_DATASETNAME = "uit_visd4sa"

_DESCRIPTION = """\
This dataset is designed for span detection for aspect-based sentiment analysis NLP task.
A Vietnamese dataset consisting of 35,396 human-annotated spans on 11,122 feedback
comments for evaluating span detection for aspect-based sentiment analysis for mobile e-commerce
"""

_HOMEPAGE = "https://github.com/kimkim00/UIT-ViSD4SA"

_LICENSE = Licenses.UNKNOWN.value

_URLS = {
    "train": "https://raw.githubusercontent.com/kimkim00/UIT-ViSD4SA/main/data/train.jsonl",
    "dev": "https://raw.githubusercontent.com/kimkim00/UIT-ViSD4SA/main/data/dev.jsonl",
    "test": "https://raw.githubusercontent.com/kimkim00/UIT-ViSD4SA/main/data/test.jsonl",
}

_SUPPORTED_TASKS = [Tasks.SPAN_BASED_ABSA]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class UITViSD4SADataset(datasets.GeneratorBasedBuilder):
    """This dataset is designed for span detection for aspect-based sentiment analysis NLP task.
    A Vietnamese dataset consisting of 35,396 human-annotated spans on 11,122 feedback
    comments for evaluating span detection for aspect-based sentiment analysis for mobile e-commerce"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="uit_visd4sa source schema",
            schema="source",
            subset_id="uit_visd4sa",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_kb",
            version=SEACROWD_VERSION,
            description="uit_visd4sa SEACrowd schema",
            schema="seacrowd_kb",
            subset_id="uit_visd4sa",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Sequence({"start": datasets.Value("int32"), "end": datasets.Value("int32"), "aspect": datasets.Value("string"), "rating": datasets.Value("string")}),
                }
            )

        elif self.config.schema == "seacrowd_kb":
            # e.g. features = schemas.kb_features
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_URLS["train"])
        dev_path = dl_manager.download_and_extract(_URLS["dev"])
        test_path = dl_manager.download_and_extract(_URLS["test"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_path,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dev_path,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, "r") as f:
            df = [json.loads(line) for line in f.readlines()]
        f.close()
        if self.config.schema == "source":
            for _id, row in enumerate(df):
                labels = row["labels"]
                entry_label = []
                for lb in labels:
                    entry_label.append({"start": lb[0], "end": lb[1], "aspect": lb[-1].split("#")[0], "rating": lb[-1].split("#")[-1]})
                entry = {
                    "text": row["text"],
                    "label": entry_label,
                }
                yield _id, entry

        elif self.config.schema == "seacrowd_kb":
            for _id, row in enumerate(df):
                entry = {
                    "id": _id,
                    "passages": [
                        {
                            "id": "text-" + str(_id),
                            "type": "text",
                            "text": [row["text"]],
                            "offsets": [[0, len(row["text"])]],
                        }
                    ],
                    "entities": [
                        {
                            "id": str(_id) + "-aspect-rating-" + str(lbl_id),
                            "type": label[-1],  # (ASPECT NAME # RATING (POSITIVE / NEGATIVE))
                            "text": [row["text"][label[0] : label[1]]],  # PART OF TEXT AFFECTED BY THE TYPE,
                            "offsets": [label[:2]],  # [START, END]
                            "normalized": [],
                        }
                        for lbl_id, label in enumerate(row["labels"])
                    ],
                    "events": [],
                    "coreferences": [{"id": str(_id) + "-0", "entity_ids": [str(_id) + "-aspect-rating-" + str(lbl_id) for lbl_id, _ in enumerate(row["labels"])]}],
                    "relations": [],
                }
                yield _id, entry
