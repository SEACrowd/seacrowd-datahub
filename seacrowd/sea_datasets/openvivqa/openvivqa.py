# coding=utf-8
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd
import json
from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{tran2021vivqa,
  title={ViVQA: Vietnamese visual question answering},
  author={Tran, Khanh Quoc and Nguyen, An Trong and Le, An Tran-Hoai and Van Nguyen, Kiet},
  booktitle={Proceedings of the 35th Pacific Asia Conference on Language, Information and Computation},
  pages={683--691},
  year={2021}
}
"""
_DATASETNAME = "openvivqa"
_DESCRIPTION = """\
OpenViVQA (Open-domain Vietnamese Visual Question Answering) is a dataset for VQA (Visual Question Answering) with
open-ended answers in Vietnamese. It consisted of 11199 images associated with 37914 question-answer pairs (QAs).
Images in the OpenViVQA dataset are captured in Vietnam and question-answer pairs are created manually by Vietnamese
crowd workers.
"""
_HOMEPAGE = "https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset"
_LANGUAGES = ["vie"]
_LICENSE = Licenses.MIT.value
_LOCAL = False
_HF_URL = "https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset"
_URLS = {
    "dataset": {
        "train": "https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset/raw/main/vlsp2023_train_data.json",
        "test": "https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset/raw/main/vlsp2023_test_data.json",
        "dev": "https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset/raw/main/vlsp2023_dev_data.json"},
    "images": {
        "train": "https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset/raw/main/train-images.zip",
        "test": "https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset/raw/main/test-images.zip",
        "dev": "https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset/raw/main/dev-images.zip"
    }
}
_SUPPORTED_TASKS = [Tasks.VISUAL_QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class OpenViVQADataset(datasets.GeneratorBasedBuilder):

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
            name=f"{_DATASETNAME}_seacrowd_imqa",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_imqa",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({"img_path": datasets.Value("string"),
                                          "question": datasets.Value("string"),
                                          "answer": datasets.Value("string"),
                                          "id": datasets.Value("string")})
        elif self.config.schema == "seacrowd_imqa":
            features = schemas.imqa_features
            features['meta'] = {"image_path": datasets.Value("string")}
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
        data_dir = dl_manager.download_and_extract(_URLS["dataset"])
        train_image_dir = dl_manager.download_and_extract(_URLS["images"]["train"])
        test_image_dir = dl_manager.download_and_extract(_URLS["images"]["test"])
        dev_image_dir = dl_manager.download_and_extract(_URLS["images"]["dev"])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "imagepath": train_image_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "imagepath": test_image_dir,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["dev"],
                    "imagepath": dev_image_dir,
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, imagepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        raw_examples = json.load(open(filepath, 'r'))
        images = raw_examples["images"]
        data_annotations = raw_examples["annotations"]
        for sample_id, q_key in enumerate(list(data_annotations.keys())):
            # '12746': {'image_id': 6860, 'question': 'vị trí cô gái ngồi như thế nào ?', 'answer': 'khá nguy hiểm'}
            quest_id = q_key
            sample = data_annotations[q_key]
            sample_img_id = sample["image_id"]
            sample_img_name = images[str(sample_img_id)]
            sample_img_path = os.path.join(imagepath, sample_img_name)
            sample_question = sample["question"]
            sample_answer = sample["answer"]
            if self.config.schema == "source":
                yield sample_id, {"img_path": sample_img_path,
                                  "question": sample_question,
                                  "answer": sample_answer,
                                  "id": quest_id
                                  }
            elif self.config.schema == "seacrowd_imqa":
                example = {
                    "id": q_key,
                    "question_id": q_key,
                    "document_id": q_key,
                    "questions": [sample_question],
                    "type": None,
                    "choices": None,
                    "context": sample_img_id,
                    "answer": [sample_answer],
                    "image_paths": [sample_img_path],
                    "meta": {"image_path": sample_img_path}
                }

                yield sample_id, example
