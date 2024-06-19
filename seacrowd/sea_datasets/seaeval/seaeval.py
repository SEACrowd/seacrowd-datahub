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

from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{SeaEval2023,
  title={SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning},
  author={Wang, Bin and Liu, Zhengyuan and Huang, Xin and Jiao, Fangkai and Ding, Yang and Aw, Ai Ti and Chen, Nancy F.},
  journal={arXiv preprint arXiv:2309.04766},
  year={2023},
  url={https://github.com/SeaEval/SeaEval}
}
"""

_DATASETNAME = "seaeval"

_DESCRIPTION = """\
SeaEval is a benchmark toolkit for evaluating multilingual LLMs. The benchmark contains 28 datasets,
covering 7 languages. It contains 2 datasets for cross-lingual consistency, each containing parallel
questions for the 7 represented languages. It alsoc ontains 4 datasets for cultural reasoning
(multiple choice Q&A) that are in English but focused on regions including Singapore and Philipines.

This dataloader provides examples for Indonesia, Vietnamese, Malay, and Filipino.
"""

_HOMEPAGE = "https://github.com/SeaEval/SeaEval"

_LANGUAGES = {"ind": "Indonesian", "vie": "Vietnamese", "zlm": "Malay", "fil": "Filipino"}

_LICENSE = Licenses.CC_BY_NC_4_0.value

_LOCAL = False

_URLS = {
    "cross_mmlu": "https://huggingface.co/datasets/SeaEval/SeaEval_datasets/raw/main/cross_mmlu.json",
    "cross_logiqa": "https://huggingface.co/datasets/SeaEval/SeaEval_datasets/raw/main/cross_logiqa.json",
    "sg_eval": "https://huggingface.co/datasets/SeaEval/SeaEval_datasets/raw/main/sg_eval.json",
    "ph_eval": "https://huggingface.co/datasets/SeaEval/SeaEval_datasets/raw/main/ph_eval.json",
}

_SUPPORTED_TASKS = [Tasks.COMMONSENSE_REASONING, Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class SeaEvalDataset(datasets.GeneratorBasedBuilder):
    """
    SeaEval is a benchmark for evaluating multilingual LLMs from https://github.com/SeaEval/SeaEval.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    LANGUAGES_EXCHANGED = dict((v, k) for k, v in _LANGUAGES.items())
    SUBSETS_CROSS_MMLU = ["cross_mmlu_" + lang for lang in _LANGUAGES.keys()]
    SUBSETS_CROSS_LOGIQA = ["cross_logiqa_" + lang for lang in _LANGUAGES.keys()]
    SUBSETS = SUBSETS_CROSS_MMLU + SUBSETS_CROSS_LOGIQA + ["sg_eval_eng", "ph_eval_eng"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME}_{subset} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in SUBSETS
    ]

    BUILDER_CONFIGS += [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_seacrowd_qa",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME}_{subset} SEACrowd schema",
            schema="seacrowd_qa",
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in SUBSETS
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source" and self.config.subset_id not in ["cross_logiqa", "ph_eval"]:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.Sequence(datasets.Value("string")),
                    "answer": datasets.Value("string"),
                }
            )
        elif self.config.schema == "source" and self.config.subset_id == "cross_logiqa":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "choices": datasets.Sequence(datasets.Value("string")),
                    "answer": datasets.Value("string"),
                }
            )
        elif self.config.schema == "source" and self.config.subset_id == "ph_eval":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.Sequence(datasets.Value("string")),
                    "answer": datasets.Value("string"),
                    "category": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features
        else:
            raise ValueError(f"Unexpected schema received! {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """
        Returns SplitGenerators.
        """

        data = {key: dl_manager.download_and_extract(value) for key, value in _URLS.items()}

        paths = {}
        file = self.config.subset_id.split("_")
        file = "_".join(file[1:3])
        paths[self.config.subset_id] = data[file]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "paths": paths,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, paths: Path, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """

        language = self.config.subset_id.split("_")[3]
        examples = None

        for key, path in paths.items():
            if "cross" in key:
                data = pd.read_json(path).rename(columns=self.LANGUAGES_EXCHANGED)
                data = pd.melt(data, id_vars=["id"], value_vars=_LANGUAGES.keys(), var_name="language")
                data_flattened = pd.json_normalize(data["value"])
                data_merged = pd.merge(data, data_flattened, left_index=True, right_index=True)
                data_filtered = data_merged[data_merged["language"] == language].drop(columns=["value", "language"])
                examples = data_filtered.to_records()
            elif "eval" in key:
                data = pd.read_json(path)
                examples = data.to_records()

        idx = 0
        if self.config.schema == "source" and self.config.subset_id not in ["cross_logiqa", "ph_eval"]:
            for row in examples:
                x = {
                    "id": row["id"],
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],
                }
                yield idx, x
                idx += 1
        elif self.config.schema == "source" and self.config.subset_id == "cross_logiqa":
            for row in examples:
                x = {
                    "id": row["id"],
                    "question": row["question"],
                    "context": row["context"] if "context" in row else None,
                    "choices": row["choices"],
                    "answer": row["answer"],
                }
                yield idx, x
                idx += 1
        elif self.config.schema == "source" and self.config.subset_id == "ph_eval":
            for row in examples:
                x = {
                    "id": row["id"],
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],
                    "category": row["category"] if "category" in row else None,
                }
                yield idx, x
                idx += 1
        elif self.config.schema == "seacrowd_qa":
            for row in examples:
                x = {
                    "id": idx,
                    "question_id": row["id"],
                    "document_id": row["id"],
                    "question": row["question"],
                    "type": "multiple_choice",
                    "choices": row["choices"],
                    "context": row["context"] if "context" in row else None,
                    "answer": [row["answer"]],
                    "meta": {},
                }
                yield idx, x
                idx += 1
        else:
            raise ValueError(f"Invalid schema: {self.config.schema}")
