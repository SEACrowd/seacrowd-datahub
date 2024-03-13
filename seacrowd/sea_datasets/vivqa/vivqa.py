# coding=utf-8
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

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
_DATASETNAME = "vivqa"
_DESCRIPTION = """\
Vietnamese Visual Question Answering (ViVQA) consist of 10328 images and 15000 question-answer
pairs in Vietnamese for evaluating Vietnamese VQA models. This dataset is built based on 10328 randomly
selected images from MS COCO dataset. The question-answer pairs were based on the COCO-QA dataset that
was automatically translated from English to Vietnamese.
"""
_HOMEPAGE = "https://github.com/kh4nh12/ViVQA"
_LANGUAGES = ["vie"]
_LICENSE = Licenses.UNKNOWN.value
_LOCAL = False
_URLS = {"viviq": {"train": "https://raw.githubusercontent.com/kh4nh12/ViVQA/main/train.csv",
                   "test": "https://raw.githubusercontent.com/kh4nh12/ViVQA/main/test.csv"},
         "cocodata": {"coco2014_train_val_annots": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                      "coco2014_train_images": "http://images.cocodataset.org/zips/train2014.zip",
                      "coco2014_val_images": "http://images.cocodataset.org/zips/val2014.zip",
                      }
         }
_SUPPORTED_TASKS = [Tasks.VISUAL_QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class VivQADataset(datasets.GeneratorBasedBuilder):
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
            features = datasets.Features({"img_id": datasets.Value("string"), "question": datasets.Value("string"), "answer": datasets.Value("string"), "type": datasets.Value("string")})
        elif self.config.schema == "seacrowd_imqa":
            features = schemas.imqa_features
            features["meta"] = {"coco_img_id": datasets.Value("string"),
                                "type":datasets.Value("int")}
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
        urls = _URLS["viviq"]
        data_dir = dl_manager.download_and_extract(urls)
        cocodata = dl_manager.download_and_extract(_URLS["cocodata"])
        Coco_Dict = self._get_image_detail(cocodata)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
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

    def _get_image_detail(self, coco_dir) -> Dict:
        print(coco_dir)
        exit()
    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        raw_examples = pd.read_csv(filepath)

        for eid, exam in raw_examples.iterrows():
            assert len(exam) == 5
            exam_id, exam_quest, exam_answer, exam_img_id, exam_type = exam

            if self.config.schema == "source":
                yield eid, {"img_id": str(exam_img_id), "question": exam_quest, "answer": exam_answer, "type": exam_type}
            elif self.config.schema == "seacrowd_imqa":
                example = {
                    "id": str(eid),
                    "question_id": str(exam_id),
                    "document_id": str(eid),
                    "questions": [exam_quest],
                    "type": None,
                    "choices": None,
                    "context": None,
                    "answer": [exam_answer],
                    "image_paths": [exam_img_id],
                    "meta": {"coco_img_id": str(exam_img_id),
                             "type": exam_type},
                }

                yield eid, example
