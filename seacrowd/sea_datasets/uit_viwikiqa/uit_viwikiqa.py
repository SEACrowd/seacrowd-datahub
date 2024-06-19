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

import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@misc{do2021sentence,
    title={Sentence Extraction-Based Machine Reading Comprehension for Vietnamese},
    author={Phong Nguyen-Thuan Do and Nhat Duy Nguyen and Tin Van Huynh and Kiet Van Nguyen and Anh Gia-Tuan Nguyen and Ngan Luu-Thuy Nguyen},
    year={2021},
    eprint={2105.09043},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DATASETNAME = "uit_viwikiqa"

_DESCRIPTION = """
UIT-ViWikiQA is a Vietnamese sentence extraction-based machine reading comprehension
dataset. It is created from the UIT-ViQuAD dataset. It comprises of 23,074
question-answers based on 5,109 passages of 174 Wikipedia Vietnamese articles.
"""

_HOMEPAGE = "https://sites.google.com/uit.edu.vn/kietnv/datasets#h.bp2c6hj2hb5q"

_LANGUAGES = ["vie"]

_LICENSE = f"""{Licenses.OTHERS.value} |
The user of UIT-ViWikiQA developed by the NLP@UIT research group must respect the
following terms and conditions:
1. The dataset is only used for non-profit research for image captioning.
2. The dataset is not allowed to be used in commercial systems.
3. Do not redistribute the dataset. This dataset may be modified or improved to serve a
   research purpose better, but the edited dataset may not be distributed.
4. Summaries, analyses, and interpretations of the properties of the dataset may be
   derived and published, provided it is not possible to reconstruct the information from
   these summaries.
5. Published research works that use the dataset must cite the following paper: Do,
   P.N.T., Nguyen, N.D., Van Huynh, T., Van Nguyen, K., Nguyen, A.G.T. and Nguyen, N.L.T.,
   2021. Sentence Extraction-Based Machine Reading Comprehension for Vietnamese. arXiv
   preprint arXiv:2105.09043.
"""

_LOCAL = True  # need to signed a user agreement, see _HOMEPAGE

_URLS = {}  # local dataset

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # qa

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class UITViWikiQADataset(datasets.GeneratorBasedBuilder):
    """Vietnamese sentence extraction-based machine reading comprehension dataset from UIT-ViQuAD dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{_SEACROWD_SCHEMA}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=_SEACROWD_SCHEMA,
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]]  # qa_features
            features["meta"] = {
                "title": datasets.Value("string"),
                "answers_start": datasets.Sequence(datasets.Value("int32")),
            }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the `data_dir` kwarg (where the .json is located) to load_dataset.")
        else:
            data_dir = Path(self.config.data_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file_path": data_dir / "train_ViWikiQA.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "file_path": data_dir / "dev_ViWikiQA.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "file_path": data_dir / "test_ViWikiQA.json",
                },
            ),
        ]

    def _generate_examples(self, file_path: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        key = 0
        for example in data["data"]:

            if self.config.schema == "source":
                for paragraph in example["paragraphs"]:
                    for qa in paragraph["qas"]:
                        yield key, {
                            "id": qa["id"],
                            "title": example["title"],
                            "context": paragraph["context"],
                            "question": qa["question"],
                            "answers": qa["answers"],
                        }
                        key += 1

            elif self.config.schema == _SEACROWD_SCHEMA:
                for paragraph in example["paragraphs"]:
                    for qa in paragraph["qas"]:
                        yield key, {
                            "id": str(key),
                            "question_id": qa["id"],
                            "document_id": None,
                            "question": qa["question"],
                            "type": None,
                            "choices": [],  # escape multiple choice qa seacrowd test
                            "context": paragraph["context"],
                            "answer": [answer["text"] for answer in qa["answers"]],
                            "meta": {
                                "title": example["title"],
                                "answers_start": [answer["answer_start"] for answer in qa["answers"]],
                            },
                        }
                        key += 1
