# coding=utf-8
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{thai-etal-2022-uit,
title = "{UIT}-{V}i{C}o{V}19{QA}: A Dataset for {COVID}-19 Community-based Question Answering on {V}ietnamese Language",
author = "Thai, Triet and Thao-Ha, Ngan Chu and Vo, Anh  and Luu, Son",
editor = "Dita, Shirley and Trillanes, Arlene and Lucas, Rochelle Irene",
booktitle = "Proceedings of the 36th Pacific Asia Conference on Language, Information and Computation",
month = oct,
year = "2022",
address = "Manila, Philippines",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/2022.paclic-1.88",
pages = "801--810",
}
"""
_DATASETNAME = "uit_vicov19qa"
_DESCRIPTION = """\
UIT-ViCoV19QA is the first Vietnamese community-based question answering dataset for developing question answering
systems for COVID-19. The dataset comprises 4,500 question-answer pairs collected from trusted medical sources,
with at least one answer and at most four unique paraphrased answers per question. This dataset contains 1800 questions
that have at least two answers, 700 questions have at least three answers and half of them have a maximum of four paraphrased
answers.
"""
_HOMEPAGE = "https://github.com/triet2397/UIT-ViCoV19QA"
_LANGUAGES = ["vie"]
_LICENSE = Licenses.UNKNOWN.value
_PAPER_URL = "https://aclanthology.org/2022.paclic-1.88"
_LOCAL = False
_URLS = {
    "train": {
        "1_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/1_ans/UIT-ViCoV19QA_train.csv",
        "2_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/2_ans/UIT-ViCoV19QA_train.csv",
        "3_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/3_ans/UIT-ViCoV19QA_train.csv",
        "4_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/4_ans/UIT-ViCoV19QA_train.csv",
    },
    "val": {
        "1_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/1_ans/UIT-ViCoV19QA_val.csv",
        "2_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/2_ans/UIT-ViCoV19QA_val.csv",
        "3_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/3_ans/UIT-ViCoV19QA_val.csv",
        "4_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/4_ans/UIT-ViCoV19QA_val.csv",
    },
    "test": {
        "1_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/1_ans/UIT-ViCoV19QA_test.csv",
        "2_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/2_ans/UIT-ViCoV19QA_test.csv",
        "3_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/3_ans/UIT-ViCoV19QA_test.csv",
        "4_ans": "https://raw.githubusercontent.com/triet2397/UIT-ViCoV19QA/main/dataset/4_ans/UIT-ViCoV19QA_test.csv",
    },
}
_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ViHealthQADataset(datasets.GeneratorBasedBuilder):
    """
    This is a SeaCrowed dataloader for dataset uit_vicov19qa, The dataset comprises 4,500 question-answer pairs collected from trusted medical sources,
    with at least one answer and at most four unique paraphrased answers per question.
    """

    subsets = ["1_ans", "2_ans", "3_ans", "4_ans"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source", subset_id=f"{_DATASETNAME}"),

        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_qa",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_qa",
            subset_id=f"{_DATASETNAME}",
        )
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features
        else:
            raise ValueError(f"No schema matched for {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        data_dir = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["val"],
                    "split": "val",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Dict, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        print(f"Generating examples for split {split}")
        sample_id = -1
        for path in filepath.values():
            raw_examples = pd.read_csv(path, na_filter=False, delimiter="|")
            for eid, exam in raw_examples.iterrows():
                sample_id += 1
                exam_id = exam[0]
                exam_quest = exam[1]
                exam_answers = exam[2:].values
                if self.config.schema == "source":
                    yield sample_id, {"id": str(exam_id),
                                      "question": exam_quest,
                                      "answers": exam_answers
                                      }

                elif self.config.schema == "seacrowd_qa":
                    yield sample_id, {"id": str(sample_id),
                                      "question_id": exam_id,
                                      "document_id": str(sample_id),
                                      "question": exam_quest,
                                      "type": None,
                                      "choices": [],
                                      "context": None,
                                      "answer": exam_answers,
                                      "meta": {}
                                      }
