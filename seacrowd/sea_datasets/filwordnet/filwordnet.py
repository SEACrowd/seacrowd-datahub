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

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses

_CITATION = """\
@article{article,
author = {Borra, Allan and Pease, Adam and Edita, Rachel and Roxas, and Dita, Shirley},
year = {2010},
month = {01},
pages = {},
title = {Introducing Filipino WordNet}
}
"""

_DATASETNAME = "filwordnet"

_DESCRIPTION = """\
Filipino WordNet (FilWordNet) is a lexical database of Filipino language.
It was derived from the Princeton WordNet and translated by humans to Filipino.
It documents 13,539 unique words and 9,519 synsets. Each synset includes the definition,
part-of-speech, word senses, and Suggested Upper Merged Ontology terms (SUMO terms).
"""

_HOMEPAGE = "https://github.com/danjohnvelasco/Filipino-WordNet"

_LANGUAGES = ["fil"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/danjohnvelasco/Filipino-WordNet/main/filwordnet.csv",
}

_SUPPORTED_TASKS = []

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class FilWordNetDataset(datasets.GeneratorBasedBuilder):
    """The Filipino WordNet (FilWordNet) is a lexical database of Filipino language containing 13,539 unique words and 9,519 synsets."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        )
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "word_id": datasets.Value("int32"),
                    "lemma": datasets.Value("string"),
                    "synset_id": datasets.Value("int32"),
                    "sense_id": datasets.Value("int32"),
                    "pos": datasets.Value("string"),
                    "lexdomain_id": datasets.Value("int32"),
                    "definition": datasets.Value("string"),
                    "last_modifier": datasets.Value("int32"),
                    "sumo": datasets.Value("string"),
                }
            )

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
        file = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        rows = []
        is_first_row = True
        with open(filepath, "r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            for row in csv_reader:
                if is_first_row:  # skip first row, they are column names
                    is_first_row = False
                    continue

                rows.append(row)

        if self.config.schema == "source":
            for key, row in enumerate(rows):
                example = {
                    "word_id": row[0],
                    "lemma": row[1],
                    "synset_id": row[2],
                    "sense_id": row[3],
                    "pos": row[4],
                    "lexdomain_id": row[5],
                    "definition": row[6],
                    "last_modifier": row[7],
                    "sumo": row[8],
                }
                yield key, example
