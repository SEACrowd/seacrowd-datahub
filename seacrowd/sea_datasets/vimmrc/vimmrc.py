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

_CITATION = """
@ARTICLE{vimmrc,
    author={Nguyen, Kiet Van and Tran, Khiem Vinh and Luu, Son T. and Nguyen, Anh Gia-Tuan and Nguyen, Ngan Luu-Thuy},
    journal={IEEE Access},
    title={Enhancing Lexical-Based Approach With External Knowledge for Vietnamese Multiple-Choice Machine Reading Comprehension},
    year={2020},
    volume={8},
    pages={201404-201417},
    doi={10.1109/ACCESS.2020.3035701}}
"""

_DATASETNAME = "vimmrc"

_DESCRIPTION = """
ViMMRC, a challenging machine comprehension corpus with multiple-choice questions,
intended for research on the machine comprehension of Vietnamese text. This corpus
includes 2,783 multiple-choice questions and answers based on a set of 417 Vietnamese
texts used for teaching reading comprehension for 1st to 5th graders.
"""

_HOMEPAGE = "https://sites.google.com/uit.edu.vn/kietnv/datasets#h.1qeaynfs79d1"

_LANGUAGES = ["vie"]

_LICENSE = f"{Licenses.UNKNOWN.value} | The corpus is freely available at our website for research purposes."

_LOCAL = False

_URL = "https://drive.google.com/file/d/14Rq-YANUv8qyi4Ze8ReEAEu_uxgcV_Yk/view"  # ~2mb

_SUPPORTED_TASKS = [Tasks.COMMONSENSE_REASONING]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # qa

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class ViMMRCDataset(datasets.GeneratorBasedBuilder):
    """A Vietnamese machine comprehension corpus with multiple-choice questions"""

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
                    "file_path": datasets.Value("string"),
                    "article": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.Sequence(datasets.Value("string")),
                    "answer": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]]  # qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # check if gdown is installed
        try:
            import gdown
        except ImportError as err:
            raise ImportError("Please install `gdown` to enable reliable data download from google drive.") from err

        # download data from gdrive
        output_dir = Path.cwd() / "data" / "vimmrc"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "vimmrc.zip"
        if not output_file.exists():
            gdown.download(_URL, str(output_file), fuzzy=True)
        else:
            print(f"File already downloaded: {str(output_file)}")

        # extract data
        data_dir = Path(dl_manager.extract(output_file)) / "ViMMRC"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir / "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_dir": data_dir / "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir / "test",
                },
            ),
        ]

    def _generate_examples(self, data_dir: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # a data_dir consists of several json files
        json_files = sorted(list(data_dir.glob("*.json")))

        key = 0
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as file:
                # load per json file
                data = json.load(file)
                assert len(data["questions"]) == len(data["options"]) == len(data["answers"]), f"Mismatched data length on {str(json_file)}"

                for idx, question in enumerate(data["questions"]):

                    # get answer based on the answer key
                    if data["answers"][idx] == "A":
                        answer = data["options"][idx][0]
                    elif data["answers"][idx] == "B":
                        answer = data["options"][idx][1]
                    elif data["answers"][idx] == "C":
                        answer = data["options"][idx][2]
                    elif data["answers"][idx] == "D":
                        answer = data["options"][idx][3]

                    if self.config.schema == "source":
                        yield key, {
                            "file_path": str(json_file),
                            "article": data["article"],
                            "question": question,
                            "choices": data["options"][idx],
                            "answer": answer,
                        }
                        key += 1

                    elif self.config.schema == _SEACROWD_SCHEMA:
                        yield key, {
                            "id": key,
                            "question_id": None,
                            "document_id": str(json_file),
                            "question": question,
                            "type": "multiple_choice",
                            "choices": data["options"][idx],
                            "context": data["article"],
                            "answer": [answer],
                            "meta": None,
                        }
                        key += 1
