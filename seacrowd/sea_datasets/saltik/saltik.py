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
Saltik is a dataset for benchmarking non-word error correction method accuracy in evaluating Indonesian words.
It consists of 58,532 non-word errors generated from 3,000 of the most popular Indonesian words.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import jsonlines
import json

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {Audah, Hanif Arkan and Yuliawati, Arlisa and Alfina, Ika},
  title     = {A Comparison Between SymSpell and a Combination of Damerau-Levenshtein Distance With the Trie Data Structure},
  journal   = {2023 10th International Conference on Advanced Informatics: Concept, Theory and Application (ICAICTA)},
  volume    = {},
  year      = {2023},
  url       = {https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10390399&casa_token=HtJUCIGGlWYAAAAA:q8ll1RWmpHtSAq2Qp5uQAE1NJETx7tUYFZIvTO1IWoaYy4eqFETSsm9p6C7tJwLZBGq5y8zc3A&tag=1},
  doi       = {},
  biburl    = {https://scholar.googleusercontent.com/scholar.bib?q=info:5C2ZSuuJCDcJ:scholar.google.com/&output=citation&scisdr=ClF7-LnEEO-epgRychQ:AFWwaeYAAAAAZa50ahRiC4_sEsA-4WL6HB3SpIo&scisig=AFWwaeYAAAAAZa50aiWPCHL1NAShFbo-QupEL2c&scisf=4&ct=citation&cd=-1&hl=en},
  bibsource = {google}
}
"""

_DATASETNAME = "saltik"

_DESCRIPTION = """\
Saltik is a dataset for benchmarking non-word error correction method accuracy in evaluating Indonesian words.
It consists of 58,532 non-word errors generated from 3,000 of the most popular Indonesian words.
"""

_HOMEPAGE = "https://github.com/ir-nlp-csui/saltik"

_LANGUAGES = ["ind"]


_LICENSE = "GNU Affero General Public License v3.0 (agpl-3.0)"

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/ir-nlp-csui/saltik/main/saltik.json",
}

_SUPPORTED_TASKS = [Tasks.NON_WORD_ERROR_SPELLING_CORRECTION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class NewDataset(datasets.GeneratorBasedBuilder):
    """It consists of 58,532 non-word errors generated from 3,000 of the most popular Indonesian words."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)


    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="saltik_source",
            version=SOURCE_VERSION,
            description="saltik source schema",
            schema="source",
            subset_id="saltik",
        ),
    ]

    DEFAULT_CONFIG_NAME = "saltik_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            # EX: Arbitrary NER type dataset
            features = datasets.Features(
               {
                   "id": datasets.Value("string"),
                   "word": datasets.Value("string"),
                   "errors": [
                       {
                           "typo": datasets.Value("string"),
                           "error_type": datasets.Value("string"),
                       }
                   ],
               }
            )
        else:
            raise NotImplementedError()

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
        file_path = dl_manager.download(urls)
        data = self._read_jsonl(file_path)
        all_words = list(data.keys())
        processed_data = []
        id = 0
        for word in all_words:
            processed_data.append({'id': id, 'word': word, 'errors': data[word]})
            id += 1
        self._write_jsonl(file_path+'.jsonl', processed_data)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": file_path+'.jsonl',
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            i = 0
            with jsonlines.open(filepath) as f:
                for each_data in f.iter():
                    ex = {
                        "id": each_data["id"],
                        "word": each_data["word"],
                        "errors": each_data["errors"],
                    }

                    yield i, ex
                    i += 1

    def _read_jsonl(self, filepath: Path):
        with open(filepath) as user_file:
            parsed_json = json.load(user_file)
            return parsed_json

    def _write_jsonl(self, filepath, values):
        with jsonlines.open(filepath, "w") as writer:
            for line in values:
                writer.write(line)