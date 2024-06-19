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
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{zhang2023m3exam,
      title={M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models},
      author={Wenxuan Zhang and Sharifah Mahani Aljunied and Chang Gao and Yew Ken Chia and Lidong Bing},
      year={2023},
      eprint={2306.05179},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "m3exam"

_DESCRIPTION = """\
M3Exam is a novel benchmark sourced from real and official human exam questions for evaluating LLMs\
in a multilingual, multimodal, and multilevel context. In total, M3Exam contains 12,317 questions in 9\
diverse languages with three educational levels, where about 23% of the questions require processing images\
for successful solving. M3Exam dataset covers 3 languages spoken in Southeast Asia.
"""

_HOMEPAGE = "https://github.com/DAMO-NLP-SG/M3Exam"

_LANGUAGES = ["jav", "tha", "vie"]
_LANG_MAPPER = {"jav": "javanese", "tha": "thai", "vie": "vietnamese"}
_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_LOCAL = False
_PASSWORD = "12317".encode("utf-8")  # password to unzip dataset after downloading
_URLS = {
    _DATASETNAME: "https://drive.usercontent.google.com/download?id=1eREETRklmXJLXrNPTyHxQ3RFdPhq_Nes&authuser=0&confirm=t",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING, Tasks.VISUAL_QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class M3ExamDataset(datasets.GeneratorBasedBuilder):
    """
    M3Exam is a novel benchmark sourced from real and official human exam questions for evaluating LLMs
    in a multilingual, multimodal, and multilevel context. In total, M3Exam contains 12,317 questions in 9
    diverse languages with three educational levels, where about 23% of the questions require processing images
    for successful solving. M3Exam dataset covers 3 languages spoken in Southeast Asia.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = (
        [SEACrowdConfig(name=f"{_DATASETNAME}_{lang}_source", version=datasets.Version(_SOURCE_VERSION), description=f"{_DATASETNAME} source schema", schema="source", subset_id=f"{_DATASETNAME}") for lang in _LANGUAGES]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang}_seacrowd_qa",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} SEACrowd schema",
                schema="seacrowd_qa",
                subset_id=f"{_DATASETNAME}",
            )
            for lang in _LANGUAGES
        ]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang}_seacrowd_imqa",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} SEACrowd schema",
                schema="seacrowd_imqa",
                subset_id=f"{_DATASETNAME}",
            )
            for lang in _LANGUAGES
        ]
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_jav_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "question_text": datasets.Value("string"),
                    "background_description": datasets.Sequence(datasets.Value("string")),
                    "answer_text": datasets.Value("string"),
                    "options": datasets.Sequence(datasets.Value("string")),
                    "language": datasets.Value("string"),
                    "level": datasets.Value("string"),
                    "subject": datasets.Value("string"),
                    "subject_category": datasets.Value("string"),
                    "year": datasets.Value("string"),
                    "need_image": datasets.Value("string"),
                    "image_paths": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features
            features["meta"] = {
                "background_description": datasets.Sequence(datasets.Value("string")),
                "level": datasets.Value("string"),
                "subject": datasets.Value("string"),
                "subject_category": datasets.Value("string"),
                "year": datasets.Value("string"),
            }
        elif self.config.schema == "seacrowd_imqa":
            features = schemas.imqa_features
            features["meta"] = {
                "background_description": datasets.Sequence(datasets.Value("string")),
                "level": datasets.Value("string"),
                "subject": datasets.Value("string"),
                "subject_category": datasets.Value("string"),
                "year": datasets.Value("string"),
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
        urls = _URLS[_DATASETNAME]
        lang = self.config.name.split("_")[1]

        data_dir = dl_manager.download(urls)

        if not os.path.exists(data_dir + "_extracted"):
            if not os.path.exists(data_dir + ".zip"):
                os.rename(data_dir, data_dir + ".zip")
            with zipfile.ZipFile(data_dir + ".zip", "r") as zip_ref:
                zip_ref.extractall(data_dir + "_extracted", pwd=_PASSWORD)  # unzipping with password
        if not os.path.exists(data_dir):
            os.rename(data_dir + ".zip", data_dir)
        image_generator = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir + "_extracted", "data/multimodal-question"),
                    "split": "train",
                },
            ),
        ]

        text_generator = [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir + "_extracted", f"data/text-question/{_LANG_MAPPER[lang]}-questions-test.json"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir + "_extracted", f"data/text-question/{_LANG_MAPPER[lang]}-questions-dev.json"),
                    "split": "dev",
                },
            ),
        ]
        if "imqa" in self.config.name:
            return image_generator
        else:
            if "source" in self.config.name:
                image_generator.extend(text_generator)
                return image_generator
            else:
                return text_generator

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        lang = self.config.name.split("_")[1]
        thai_answer_mapper = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "๑": "1", "๒": "2", "๓": "3", "๔": "4", "๕": "5"}
        if self.config.schema == "source":
            if split == "train":
                filepath_json = os.path.join(filepath, f"{_LANG_MAPPER[lang]}-questions-image.json")
                with open(filepath_json, "r") as file:
                    data = json.load(file)
                idx = 0
                for json_obj in data:
                    image_paths = []
                    for text in [json_obj["question_text"]] + json_obj["options"] + json_obj["background_description"]:
                        matches = re.findall(r"\[image-(\d+)\.(jpg|png)\]", text)
                        if matches:
                            image_path = [os.path.join(filepath, f"images-{_LANG_MAPPER[lang]}/image-{image_number[0]}.{image_number[1]}") for image_number in matches]
                            image_paths.extend(image_path)
                    example = {
                        "question_text": json_obj["question_text"],
                        "background_description": json_obj["background_description"] if "background_description" in json_obj.keys() else None,
                        "answer_text": json_obj["answer_text"],
                        "options": json_obj["options"],
                        "language": json_obj["language"] if "language" in json_obj.keys() else None,
                        "level": json_obj["level"] if "level" in json_obj.keys() else None,
                        "subject": json_obj["subject"] if "subject" in json_obj.keys() else None,
                        "subject_category": json_obj["subject_category"] if "subject_category" in json_obj.keys() else None,
                        "year": json_obj["year"] if "year" in json_obj.keys() else None,
                        "need_image": "yes",
                        "image_paths": image_paths,
                    }
                    yield idx, example
                    idx += 1
            else:
                with open(filepath, "r") as file:
                    data = json.load(file)
                idx = 0
                for json_obj in data:
                    example = {
                        "question_text": json_obj["question_text"],
                        "background_description": json_obj["background_description"] if "background_description" in json_obj.keys() else None,
                        "answer_text": json_obj["answer_text"],
                        "options": json_obj["options"],
                        "language": json_obj["language"] if "language" in json_obj.keys() else None,
                        "level": json_obj["level"] if "level" in json_obj.keys() else None,
                        "subject": json_obj["subject"] if "subject" in json_obj.keys() else None,
                        "subject_category": json_obj["subject_category"] if "subject_category" in json_obj.keys() else None,
                        "year": json_obj["year"] if "year" in json_obj.keys() else None,
                        "need_image": "no",
                        "image_paths": None,
                    }
                    yield idx, example
                    idx += 1

        elif self.config.schema == "seacrowd_qa":
            with open(filepath, "r") as file:
                data = json.load(file)
            idx = 0

            for json_obj in data:
                answer = [".".join(answer.split(".")[1:]).strip() for answer in json_obj["options"] if json_obj["answer_text"] == answer.split(".")[0]]
                if "_tha_" in self.config.name and len(answer) == 0:
                    answer = [".".join(answer.split(".")[1:]).strip() for answer in json_obj["options"] if thai_answer_mapper[json_obj["answer_text"]] == thai_answer_mapper[answer.split(".")[0]]]

                example = {
                    "id": idx,
                    "question_id": idx,
                    "document_id": idx,
                    "question": json_obj["question_text"],
                    "type": "multiple_choice",
                    "choices": [".".join(answer.split(".")[1:]).strip() for answer in json_obj["options"]],
                    "context": "",
                    "answer": answer,
                    "meta": {
                        "background_description": json_obj["background_description"] if "background_description" in json_obj.keys() else None,
                        "level": json_obj["level"] if "level" in json_obj.keys() else None,
                        "subject": json_obj["subject"] if "subject" in json_obj.keys() else None,
                        "subject_category": json_obj["subject_category"] if "subject_category" in json_obj.keys() else None,
                        "year": json_obj["year"] if "year" in json_obj.keys() else None,
                    },
                }
                yield idx, example
                idx += 1

        elif self.config.schema == "seacrowd_imqa":
            filepath_json = os.path.join(filepath, f"{_LANG_MAPPER[lang]}-questions-image.json")
            with open(filepath_json, "r") as file:
                data = json.load(file)
            idx = 0

            for json_obj in data:
                answer = [".".join(answer.split(".")[1:]).strip() for answer in json_obj["options"] if json_obj["answer_text"] == answer.split(".")[0]]
                if "_tha_" in self.config.name and len(answer) == 0:
                    answer = [".".join(answer.split(".")[1:]).strip() for answer in json_obj["options"] if thai_answer_mapper[json_obj["answer_text"]] == thai_answer_mapper[answer.split(".")[0]]]
                image_paths = []
                for text in [json_obj["question_text"]] + json_obj["options"] + json_obj["background_description"]:
                    matches = re.findall(r"\[image-(\d+)\.(jpg|png)\]", text)
                    if matches:
                        image_path = [os.path.join(filepath, f"images-{_LANG_MAPPER[lang]}/image-{image_number[0]}.{image_number[1]}") for image_number in matches]
                        image_paths.extend(image_path)

                example = {
                    "id": idx,
                    "question_id": idx,
                    "document_id": idx,
                    "questions": [json_obj["question_text"]],
                    "type": "multiple_choice",
                    "choices": [".".join(answer.split(".")[1:]).strip() for answer in json_obj["options"]],
                    "context": "",
                    "answer": answer,
                    "image_paths": image_paths,
                    "meta": {
                        "background_description": json_obj["background_description"] if "background_description" in json_obj.keys() else None,
                        "level": json_obj["level"] if "level" in json_obj.keys() else None,
                        "subject": json_obj["subject"] if "subject" in json_obj.keys() else None,
                        "subject_category": json_obj["subject_category"] if "subject_category" in json_obj.keys() else None,
                        "year": json_obj["year"] if "year" in json_obj.keys() else None,
                    },
                }
                yield idx, example
                idx += 1
