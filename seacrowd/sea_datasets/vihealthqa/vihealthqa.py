# coding=utf-8
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@InProceedings{nguyen2022viheathqa,
  author="Nguyen, Nhung Thi-Hong
  and Ha, Phuong Phan-Dieu
  and Nguyen, Luan Thanh
  and Van Nguyen, Kiet
  and Nguyen, Ngan Luu-Thuy",
  title="SPBERTQA: A Two-Stage Question Answering System Based on Sentence Transformers for Medical Texts",
  booktitle="Knowledge Science, Engineering and Management",
  year="2022",
  publisher="Springer International Publishing",
  address="Cham",
  pages="371--382",
  isbn="978-3-031-10986-7"
}
"""
_DATASETNAME = "vihealthqa"
_DESCRIPTION = """\
Vietnamese Visual Question Answering (ViVQA) consist of 10328 images and 15000 question-answer
pairs in Vietnamese for evaluating Vietnamese VQA models. This dataset is built based on 10328 randomly
selected images from MS COCO dataset. The question-answer pairs were based on the COCO-QA dataset that
was automatically translated from English to Vietnamese.
"""
_HOMEPAGE = "https://huggingface.co/datasets/tarudesu/ViHealthQA"
_LANGUAGES = ["vie"]
_LICENSE = Licenses.UNKNOWN.value
_PAPER_URL = "https://link.springer.com/chapter/10.1007/978-3-031-10986-7_30"
_LOCAL = False
_URLS = {
    "vihealthqa": {
        "train": "https://huggingface.co/datasets/tarudesu/ViHealthQA/raw/main/train.csv",
        "val": "https://huggingface.co/datasets/tarudesu/ViHealthQA/raw/main/val.csv",
        "test": "https://huggingface.co/datasets/tarudesu/ViHealthQA/raw/main/test.csv",
    }
}
_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ViHealthQADataset(datasets.GeneratorBasedBuilder):
    '''
This is a SeaCrowed dataloader for dataset Vietnamese Visual Question Answering (ViVQA), which consists of 10328 images and 15000 question-answer
pairs in Vietnamese for evaluating Vietnamese VQA models.
    '''
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
            name=f"{_DATASETNAME}_seacrowd_qa",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_qa",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "link": datasets.Value("string")
                }
            )
        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features
            features["meta"] = {"link": datasets.Value("string")}
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
        urls = _URLS["vihealthqa"]
        data_dir = dl_manager.download_and_extract(urls)
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

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        raw_examples = pd.read_csv(filepath)

        for eid, exam in raw_examples.iterrows():
            assert len(exam) == 4
            exam_id, exam_quest, exam_answer, exam_link = exam

            if self.config.schema == "source":
                yield eid, {"id": str(exam_id), "question": exam_quest, "answer": exam_answer, "link": exam_link}

            elif self.config.schema == "seacrowd_qa":
                yield eid, {
                    "id": str(eid),
                    "question_id": exam_id,
                    "document_id": str(eid),
                    "question": exam_quest,
                    "type": None,
                    "choices": [],
                    "context": exam_link,
                    "answer": [exam_answer],
                    "meta": {
                        "link": exam_link,
                    },
                }
