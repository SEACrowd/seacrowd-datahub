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

"""
This test is for 15 years old Malaysia student, it is about reading comprehension and general knowledge for malay language.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import re

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses, TASK_TO_SCHEMA

_CITATION = None

_DATASETNAME = "bm_pt3"

_DESCRIPTION = """\
This test is for 15 years old Malaysia student, it is about reading comprehension and general knowledge for malay language.
"""

_HOMEPAGE = "https://github.com/mesolitica/malaysian-dataset/tree/master/llm-benchmark/BM-pt3"

_LANGUAGES = ["zlm"]

_LICENSE = Licenses.UNLICENSE.value

_LOCAL = False

_URLS = {
    "A": "https://raw.githubusercontent.com/mesolitica/malaysian-dataset/master/llm-benchmark/BM-pt3/BM-A-pt3",
    "B": "https://raw.githubusercontent.com/mesolitica/malaysian-dataset/master/llm-benchmark/BM-pt3/BM-B-pt3"
}

_SUPPORTED_TASKS = [Tasks.COMMONSENSE_REASONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class BMPT3Dataset(datasets.GeneratorBasedBuilder):
    """This test is for 15 years old Malaysia student, it is about reading comprehension and general knowledge for malay language."""


    SUBSETS = ["A", "B"]
    SEACROWD_SCHEMA = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema for {subset} subset",
            schema="source",
            subset_id=f"{_DATASETNAME}_{subset}",
        ) 
        for subset in SUBSETS
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_seacrowd_qa",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema for {subset} subset",
            schema=f"seacrowd_qa",
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in SUBSETS
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "num": datasets.Value("string"),
                    "objective": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.Sequence(datasets.Value("string")),
                    "answer": datasets.Value("string"),
                    "source": {
                        "title": datasets.Value("string"),
                        "num": datasets.Value("string"),
                        "url": datasets.Value("string"),
                    }
                }
            )

        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features
            features["meta"] = {
                "source": {
                    "title": datasets.Value("string"),
                    "num": datasets.Value("string"),
                    "url": datasets.Value("string"),
                }
            }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:

        if "A" in self.config.subset_id:
            subset_type = "A"
            data_dir = dl_manager.download_and_extract(_URLS["A"])
        elif "B" in self.config.subset_id:
            subset_type = "B"
            data_dir = dl_manager.download_and_extract(_URLS["B"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "subset_type": subset_type
                },
            ),
        ]


    def _generate_examples(self, filepath: Path, subset_type: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = self._extract_data(f.read(), subset_type)
        
        if self.config.schema == "source":
            for i, entry in enumerate(data):
                yield i, entry

        elif self.config.schema == "seacrowd_qa":
            for i, entry in enumerate(data):
                yield i, {
                    "id": str(i),
                    "question_id": entry["num"],
                    "document_id": None,
                    "question": entry["question"],
                    "type": "multiple_choice" if entry["choices"] else "open_ended",
                    "choices": entry["choices"],
                    "context": entry["objective"],
                    "answer": [entry["answer"]] if entry["answer"] else [],
                    "meta": {
                        "source": entry["source"]
                    }
                }
    
    def _extract_data(self, doc: str, subset_type: str) -> List[Dict]:
        """Extracts data from the source schema"""

        # RegEx pattern
        pattern_num = re.compile(r"(no:\s*\d+)")
        pattern_objective = re.compile(r"objektif:\s*(.*)")
        pattern_question = re.compile(r'soalan:\s*(.*?)(?=\njawapan:|asal soalan:)', re.DOTALL)
        pattern_choices = re.compile(r'([A-D]\.\s+.+?)(?=\n[A-D]\.|\Z)', re.DOTALL)
        if subset_type == "A":
            pattern_answer = re.compile(r'jawapan:\s*([A-D])[,\s]', re.DOTALL)
        elif subset_type == "B":
            pattern_answer = re.compile(r'jawapan:\s*(.*?)\s*asal soalan:', re.DOTALL)
        pattern_asal_soalan = re.compile(r'asal soalan:\s*(.*?),\s*no\s*(\d+),\s*(.*?)\n', re.DOTALL)

        res = []
        doc_split = re.sub(pattern_num, "<NUMBER>", doc).split("<NUMBER>")[1:]

        for i, entry in enumerate(doc_split):
            # Objektif
            objective = re.findall(pattern_objective, entry)
            objective = objective[0] if objective else None
            
            # Soalan
            _question = re.findall(pattern_question, entry)
            question = re.sub(pattern_choices, '', _question[0]).strip("\n") if _question else None
            
            # Choices Soalan
            choices = {}
            if _question and subset_type == "A":
                _choices = re.findall(pattern_choices, _question[0])
                for _c in _choices:
                    alpha, txt = _c.split(". ")[0], ' '.join(_c.split(". ")[1:])
                    choices[alpha] = txt
            
            # Answer
            if subset_type == "A":
                _answer = re.findall(pattern_answer, entry)
                answer = choices[_answer[0]] if (_answer and choices) else None
            elif subset_type == "B":
                answer = re.findall(pattern_answer, entry)
                answer = answer[0] if answer else None
            
            # Asal soalan
            source = re.findall(pattern_asal_soalan, entry)
            source = source[0] if source else [None,None,None]

            res.append({
                "num": str(i+1),
                "objective": objective,
                "question": question,
                "choices": list(choices.values()) if choices else [],
                "answer": answer,
                "source": {
                    "title": source[0],
                    "num": source[1],
                    "url": source[2]
                }
            })

        return res
