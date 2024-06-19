# coding=utf-8

import json
from pathlib import Path
import re
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

_LANGUAGES = ["vie"]

_URLS = {
    "train": "https://raw.githubusercontent.com/kimkim00/UIT-ViSD4SA/main/data/train.jsonl",
    "dev": "https://raw.githubusercontent.com/kimkim00/UIT-ViSD4SA/main/data/dev.jsonl",
    "test": "https://raw.githubusercontent.com/kimkim00/UIT-ViSD4SA/main/data/test.jsonl",
}

_SUPPORTED_TASKS = [Tasks.SPAN_BASED_ABSA]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_LOCAL = False


def construct_label_classes():
    IOB_tag = ["I", "O", "B"]
    aspects = ["SCREEN", "CAMERA", "FEATURES", "BATTERY", "PERFORMANCE", "STORAGE", "DESIGN", "PRICE", "GENERAL", "SER&ACC"]
    ratings = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    label_classes = []
    for iob in IOB_tag:
        if iob == "O":
            label_classes.append("O")
        else:
            for aspect in aspects:
                for rating in ratings:
                    label_classes.append("{iob}-{aspect}#{rating}".format(iob=iob, aspect=aspect, rating=rating))
    return label_classes


def construct_IOB_sequences(text, labels):
    labels.sort()
    word_start = [0] + [match.start() + 1 for match in re.finditer(" ", text)]
    is_not_O = False
    iob_sequence = []
    word_count = 0
    lb_count = 0

    while word_count < len(word_start):
        if lb_count == len(labels):
            for x in range(word_count, len(word_start)):
                iob_sequence.append("O")
            break
        if not is_not_O:
            if word_start[word_count] >= labels[lb_count][0]:
                is_not_O = True
                iob_sequence.append("B-" + labels[lb_count][-1])
                word_count += 1
            else:
                iob_sequence.append("O")
                word_count += 1
        else:
            if word_start[word_count] > labels[lb_count][1]:
                is_not_O = False
                lb_count += 1
            else:
                iob_sequence.append("I-" + labels[lb_count][-1])
                word_count += 1
    return iob_sequence


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
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="uit_visd4sa SEACrowd schema",
            schema="seacrowd_seq_label",
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

        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(construct_label_classes())

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
        train_path, dev_path, test_path = path_dict["train"], path_dict["dev"], path_dict["test"]

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
                    "filepath": dev_path,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, "r") as f:
            df = [json.loads(line) for line in f.readlines()]
        f.close()
        if self.config.schema == "source":
            for _id, row in enumerate(df):
                labels = row["labels"]
                entry_labels = []
                for lb in labels:
                    entry_labels.append({"start": lb[0], "end": lb[1], "aspect": lb[-1].split("#")[0], "rating": lb[-1].split("#")[-1]})
                entry = {
                    "text": row["text"],
                    "label": entry_labels,
                }
                yield _id, entry
        elif self.config.schema == "seacrowd_seq_label":
            for _id, row in enumerate(df):
                entry = {
                    "id": str(_id),
                    "tokens": row["text"].split(" "),
                    "labels": construct_IOB_sequences(row["text"], row["labels"]),
                }
                yield _id, entry
