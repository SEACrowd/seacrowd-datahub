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

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{buechel-etal-2020-learning-evaluating,
    title = "Learning and Evaluating Emotion Lexicons for 91 Languages",
    author = {Buechel, Sven  and
      R{\"u}cker, Susanna  and
      Hahn, Udo},
    editor = "Jurafsky, Dan  and
      Chai, Joyce  and
      Schluter, Natalie  and
      Tetreault, Joel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.112",
    doi = "10.18653/v1/2020.acl-main.112",
    pages = "1202--1217",
}
"""

_DATASETNAME = "memolon"

_DESCRIPTION = """\
MEmoLon is an emotion lexicons for 91 languages, each one covers eight emotional variables and comprises over 100k word entries. There are several versions of the lexicons, the difference being the choice of the expansion model.
"""

_HOMEPAGE = "https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1"

_LICENSE = Licenses.MIT.value

_URLS = {
    _DATASETNAME: "https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1",
}

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_LANGUAGES = ["ceb", "tgl", "ind", "sun", "jav", "zsm", "vie", "tha", "mya"]

_LANGUAGE_MAP = {"ceb": "Cebuano", "tgl": "Tagalog", "ind": "Indonesian", "sun": "Sundanese", "jav": "Javanese", "zsm": "Malay", "vie": "Vietnamese", "tha": "Thai", "mya": "Burmese"}

_SUPPORTED_TASKS = [Tasks.EMOTION_CLASSIFICATION]

_LOCAL = False


def seacrowd_config_constructor(lang: str, schema: str, version: str) -> SEACrowdConfig:
    if lang not in _LANGUAGE_MAP:
        raise ValueError(f"Invalid lang {lang}")

    if schema != "source" and schema != "seacrowd_text_multi":
        raise ValueError(f"Invalid schema: {schema}")

    return SEACrowdConfig(
        name="memolon_{lang}_{schema}".format(lang=lang, schema=schema),
        version=datasets.Version(version),
        description="MEmoLon {schema} schema for {lang} language".format(lang=_LANGUAGE_MAP[lang], schema=schema),
        schema=schema,
        subset_id="memolon",
    )


class Memolon(datasets.GeneratorBasedBuilder):
    """MEmoLon is an emotion lexicons for 91 languages, each one covers eight emotional variables and comprises over 100k word entries."""

    BUILDER_CONFIGS = [SEACrowdConfig(name=f"{_DATASETNAME}_{lang}_source", version=datasets.Version(_SOURCE_VERSION), description=f"MEmoLon source schema for {lang} language", schema="source", subset_id="memolon") for lang in _LANGUAGE_MAP]

    DEFAULT_CONFIG_NAME = None

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "word": datasets.Value("string"),
                    "valence": datasets.Value("float32"),
                    "arousal": datasets.Value("float32"),
                    "dominance": datasets.Value("float32"),
                    "joy": datasets.Value("float32"),
                    "anger": datasets.Value("float32"),
                    "sadness": datasets.Value("float32"),
                    "fear": datasets.Value("float32"),
                    "disgust": datasets.Value("float32"),
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
        base_path = Path(dl_manager.download_and_extract(urls))
        lang = self.config.name.split("_")[1]
        train_data_path = base_path / f"{lang}.tsv"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_data_path,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        rows = []
        with open(filepath, encoding='utf-8') as file:
            for line in file:
                rows.append(line.split("\t"))

        if self.config.schema == "source":
            for key, row in enumerate(rows[1:]):
                example = {"word": row[0], "valence": row[1], "arousal": row[2], "dominance": row[3], "joy": row[4], "anger": row[5], "sadness": row[6], "fear": row[7], "disgust": row[8]}
                yield key, example
