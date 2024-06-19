import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import SCHEMA_TO_FEATURES, Licenses, Tasks

_CITATION = """\
@inproceedings{lin2022fewshot,
    author       = {Xi Victoria Lin and
                    Todor Mihaylov and
                    Mikel Artetxe and
                    Tianlu Wang and
                    Shuohui Chen and
                    Daniel Simig and
                    Myle Ott and
                    Naman Goyal and
                    Shruti Bhosale and
                    Jingfei Du and
                    Ramakanth Pasunuru and
                    Sam Shleifer and
                    Punit Singh Koura and
                    Vishrav Chaudhary and
                    Brian O'Horo and
                    Jeff Wang and
                    Luke Zettlemoyer and
                    Zornitsa Kozareva and
                    Mona T. Diab and
                    Veselin Stoyanov and
                    Xian Li},
    editor       = {Yoav Goldberg and
                    Zornitsa Kozareva and
                    Yue Zhang},
    title        = {Few-shot Learning with Multilingual Generative Language Models},
    booktitle    = {Proceedings of the 2022 Conference on Empirical Methods in Natural
                    Language Processing, {EMNLP} 2022, Abu Dhabi, United Arab Emirates,
                    December 7-11, 2022},
    pages        = {9019--9052},
    publisher    = {Association for Computational Linguistics},
    year         = {2022},
    url          = {https://doi.org/10.18653/v1/2022.emnlp-main.616},
    doi          = {10.18653/V1/2022.EMNLP-MAIN.616},
}
"""

_DATASETNAME = "xstorycloze"
_DESCRIPTION = """\
XStoryCloze consists of the professionally translated version of the English StoryCloze
dataset (Spring 2016 version) to 10 non-English languages. This dataset is released by
Meta AI.
"""
_HOMEPAGE = "https://huggingface.co/datasets/juletxara/xstory_cloze"
_LANGUAGES = ["ind", "mya"]
_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False
_BASE_URL = "https://huggingface.co/datasets/juletxara/xstory_cloze/resolve/main/spring2016.val.{lang}.tsv.split_20_80_{split}.tsv"
_SUPPORTED_TASKS = [Tasks.COMMONSENSE_REASONING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class XStoryClozeDataset(datasets.GeneratorBasedBuilder):
    """XStoryCloze subset for Indonesian and Burmese language."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SUBSET = ["id", "my"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} {subset} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in SEACROWD_SUBSET
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_seacrowd_qa",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} {subset} SEACrowd schema",
            schema="seacrowd_qa",
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in SEACROWD_SUBSET
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{SEACROWD_SUBSET[0]}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "story_id": datasets.Value("string"),
                    "input_sentence_1": datasets.Value("string"),
                    "input_sentence_2": datasets.Value("string"),
                    "input_sentence_3": datasets.Value("string"),
                    "input_sentence_4": datasets.Value("string"),
                    "sentence_quiz1": datasets.Value("string"),
                    "sentence_quiz2": datasets.Value("string"),
                    "answer_right_ending": datasets.Value("int32"),
                }
            )
        elif self.config.schema == "seacrowd_qa":
            features = SCHEMA_TO_FEATURES["QA"]

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        lang = self.config.name.split("_")[1]
        filepaths = dl_manager.download_and_extract(
            {
                "train": _BASE_URL.format(lang=lang, split="train"),
                "test": _BASE_URL.format(lang=lang, split="eval"),
            }
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepaths["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": filepaths["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        with open(filepath, encoding="utf-8") as f:
            data = csv.reader(f, quotechar='"', delimiter="\t", quoting=csv.QUOTE_ALL, skipinitialspace=True)
            _ = next(data)  # skip header
            if self.config.schema == "source":
                for id, row in enumerate(data):
                    yield id, {
                        "story_id": row[0],
                        "input_sentence_1": row[1],
                        "input_sentence_2": row[2],
                        "input_sentence_3": row[3],
                        "input_sentence_4": row[4],
                        "sentence_quiz1": row[5],
                        "sentence_quiz2": row[6],
                        "answer_right_ending": int(row[7]),
                    }
            elif self.config.schema == "seacrowd_qa":
                for id, row in enumerate(data):
                    question = " ".join(row[1:5])
                    choices = [row[5], row[6]]
                    yield id, {
                        "id": str(id),
                        "question_id": row[0],
                        "document_id": None,
                        "question": question,
                        "type": "multiple_choice",
                        "choices": choices,
                        "context": None,
                        "answer": [choices[int(row[7]) - 1]],
                        "meta": {},
                    }
