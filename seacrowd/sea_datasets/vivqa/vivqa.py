# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

# TODO: Add BibTeX citation
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

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
Vietnamese Visual Question Answering (ViVQA) consist of 10328 images and 15000 question-answer
pairs in Vietnamese for evaluating Vietnamese VQA models. This dataset is built based on 10328 randomly
selected images from MS COCO dataset. The question-answer pairs were based on the COCO-QA dataset that
was automatically translated from English to Vietnamese.
"""

_HOMEPAGE = "https://github.com/kh4nh12/ViVQA"

_LANGUAGES = ["vie"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.UNKNOWN.value # example: Licenses.MIT.value, Licenses.CC_BY_NC_SA_4_0.value, Licenses.UNLICENSE.value, Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "viviq":
        {
            "train": "https://raw.githubusercontent.com/kh4nh12/ViVQA/main/train.csv",
            "test": "https://raw.githubusercontent.com/kh4nh12/ViVQA/main/test.csv"
        }
}

_SUPPORTED_TASKS = [Tasks.IMAGE_CAPTIONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class vivqaDataset(datasets.GeneratorBasedBuilder):
    """Vietnamese Visual Question Answering (ViVQA) consist of 10328 images and 15000 question-answer
pairs in Vietnamese for evaluating Vietnamese VQA models. This dataset is built based on 10328 randomly
selected images from MS COCO dataset. The question-answer pairs were based on the COCO-QA dataset that
was automatically translated from English to Vietnamese."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="vivqa_source",
            version=SOURCE_VERSION,
            description="vivqa source schema",
            schema="source",
            subset_id="vivqa",
        ),
        SEACrowdConfig(
            name="vivqa_seacrowd_imtext",
            version=SEACROWD_VERSION,
            description="vivqa SEACrowd schema",
            schema="seacrowd_imtext",
            subset_id="vivqa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "vivqa_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "img_id": datasets.Value("string"),  # img_id in coco-dataset
                    "question": datasets.Value("string"),  # question
                    "answer": datasets.Value("string"),  # answer
                    "type": datasets.Value("string")  # type
                }
            )

        # For example seacrowd_kb, seacrowd_t2t
        elif self.config.schema == "seacrowd_imtext":
            features = schemas.image_text_features()

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


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir['train'],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir['test'],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        raw_examples = pd.read_csv(filepath)

        for eid, exam in raw_examples.iterrows():
            assert(len(exam) == 5)
            exam_id, exam_quest, exam_answer, exam_img_id, exam_type = exam

            if self.config.schema == "source":
                yield eid, {
                    "img_id": str(exam_img_id),  # img_id in coco-dataset
                    "question": exam_quest,  # question
                    "answer": exam_answer,  # answer
                    "type": exam_type  # type
                }
            elif self.config.schema == "seacrowd_imtext":
                yield eid, {
                    "id": str(eid),
                    "image_paths": [exam_img_id],
                    "texts": exam_answer,
                    "metadata": {
                        "context": exam_quest,
                        "labels": ["No"]
                    }

                }

            # "id": datasets.Value("string"),
            # "image_paths": datasets.Sequence(datasets.Value("string")),
            # "texts": datasets.Value("string"),
            # "metadata": {
            #     "context": datasets.Value("string"),
            #     "labels": datasets.Sequence(datasets.ClassLabel(names=label_names)),
            # }