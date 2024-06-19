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
This is an automatically-produced question answering dataset \
    generated from Indonesian Wikipedia articles. Each entry \
    in the dataset consists of a context paragraph, the \
    question and answer, and the question's equivalent SPARQL \
    query. Questions are separated into two subsets: simple \
    (question consists of a single SPARQL triple pattern) and \
    complex (question consists of two triples plus an optional \
    typing triple).
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{afa5bf8149d6406786539c1ea827087d,
    title = "AC-IQuAD: Automatically Constructed Indonesian Question Answering Dataset by Leveraging Wikidata",
    abstract = "Constructing a question-answering dataset can be prohibitively expensive, making it difficult for researchers
    to make one for an under-resourced language, such as Indonesian. We create a novel Indonesian Question Answering dataset
    that is produced automatically end-to-end. The process uses Context Free Grammar, the Wikipedia Indonesian Corpus, and
    the concept of the proxy model. The dataset consists of 134 thousand simple questions and 60 thousand complex questions.
    It achieved competitive grammatical and model accuracy compared to the translated dataset but suffers from some issues
    due to resource constraints.",
    keywords = "Automatic dataset construction, Question answering dataset, Under-resourced Language",
    author = "Kerenza Doxolodeo and Krisnadhi, {Adila Alfa}",
    note = "Publisher Copyright: {\textcopyright} 2024, The Author(s).",
    year = "2024",
    doi = "10.1007/s10579-023-09702-y",
    language = "English",
    journal = "Language Resources and Evaluation",
    issn = "1574-020X",
    publisher = "Springer Netherlands",
}
"""

_DATASETNAME = "ac_iquad"

_DESCRIPTION = """
This is an automatically-produced question answering dataset \
    generated from Indonesian Wikipedia articles. Each entry \
    in the dataset consists of a context paragraph, the \
    question and answer, and the question's equivalent SPARQL \
    query. Questions are separated into two subsets: simple \
    (question consists of a single SPARQL triple pattern) and \
    complex (question consists of two triples plus an optional \
    typing triple).
"""

_HOMEPAGE = "https://www.kaggle.com/datasets/realdeo/indonesian-qa-generated-by-kg"

_LANGUAGES = ["ind"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://github.com/muhammadravi251001/ac-iquad/raw/main/data/ac_iquad.zip",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class ACIQuADDataset(datasets.GeneratorBasedBuilder):
    """
    This is an automatically-produced question answering dataset \
    generated from Indonesian Wikipedia articles. Each entry \
    in the dataset consists of a context paragraph, the \
    question and answer, and the question's equivalent SPARQL \
    query. Questions are separated into two subsets: simple \
    (question consists of a single SPARQL triple pattern) and \
    complex (question consists of two triples plus an optional \
    typing triple).
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "qa"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_simple_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_simple",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_simple_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_simple",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_complex_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_complex",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_complex_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_complex",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_simple_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features_dict = {
                "question": datasets.Value("string"),
                "sparql": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "context": datasets.Value("string"),
                "answerline": datasets.Value("string"),
            }

            if self.config.subset_id.split("_")[2] == "complex":
                features_dict["type"] = datasets.Value("string")

            features = datasets.Features(features_dict)

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.qa_features

            if self.config.subset_id.split("_")[2] == "complex":
                features["meta"] = {"sparql": datasets.Value("string"), "answer_meta": datasets.Value("string"), "type": datasets.Value("string")}

            else:
                features["meta"] = {"sparql": datasets.Value("string"), "answer_meta": datasets.Value("string")}

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        subset = self.config.name.split("_")[2]
        data_dir = dl_manager.download_and_extract(_URLS[_DATASETNAME])

        if subset == "simple":
            subset = "single"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"{subset}_train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"{subset}_test.json"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r", encoding="utf-8") as file:
            data_json = json.load(file)

        df = pd.json_normalize(data_json)

        for index, row in df.iterrows():

            if self.config.schema == "source":
                example = row.to_dict()

                if self.config.subset_id.split("_")[2] == "complex":
                    example["type"] = example.pop("tipe", None)

            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":

                subset = self.config.name.split("_")[2]
                if subset == "simple":
                    row["answerline"] = f"[{row['answerline']}]"

                example = {
                    "id": str(index),
                    "question_id": "question_id",
                    "document_id": "document_id",
                    "question": row["question"],
                    "type": "extractive",
                    "choices": [],
                    "context": row["context"],
                    "answer": eval(row["answerline"]),
                    "meta": {"sparql": row["sparql"], "answer_meta": row["answer"]},
                }

                if self.config.subset_id.split("_")[2] == "complex":
                    example["meta"]["type"] = row["tipe"]

            yield index, example
