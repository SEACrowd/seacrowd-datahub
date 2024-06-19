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
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@inproceedings{mayhew-etal-2020-simultaneous,
    title = "Simultaneous Translation and Paraphrase for Language Education",
    author = "Mayhew, Stephen  and
        Bicknell, Klinton  and
        Brust, Chris  and
        McDowell, Bill  and
        Monroe, Will  and
        Settles, Burr",
    editor = "Birch, Alexandra  and
        Finch, Andrew  and
        Hayashi, Hiroaki  and
        Heafield, Kenneth  and
        Junczys-Dowmunt, Marcin  and
        Konstas, Ioannis  and
        Li, Xian  and
        Neubig, Graham  and
        Oda, Yusuke",
    booktitle = "Proceedings of the Fourth Workshop on Neural Generation and Translation",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.ngt-1.28",
    doi = "10.18653/v1/2020.ngt-1.28",
    pages = "232--243",
}
"""

_DATASETNAME = "duolingo_staple_2020"

_DESCRIPTION = """\
This dataset is provided by Duolingo for their Simultaneous Translation and
Paraphrase for Language Education (STAPLE) shared task in 2020. It contains
English prompts and corresponding sets of plausible translations in five other
languages, including Vietnamese. Each prompt is provided with a baseline
automatic reference translation from Amazon, as well as some accepted
translations with corresponding user response rates used for task scoring.
"""

_HOMEPAGE = "https://sharedtask.duolingo.com/#data"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_NC_4_0.value

_LOCAL = True  # needs to fill a form to download the dataset (dynamic link)

_URLS = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/38OJR6&version=6.0"

# `aws_baseline` refers to reference translations from Amazon Automated MT model,
# while `gold` refers to translations accepted by Duolingo learners
_SUBSETS = ["aws_baseline", "gold"]

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # t2t

_SOURCE_VERSION = "6.0.0"

_SEACROWD_VERSION = "2024.06.20"


class DuolingoStaple2020Dataset(datasets.GeneratorBasedBuilder):
    """Dataset for the Duolingo STAPLE 2020 shared task."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in _SUBSETS:
        BUILDER_CONFIGS += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} {subset} source schema",
                schema="source",
                subset_id=subset,
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset}_{_SEACROWD_SCHEMA}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {subset} SEACrowd schema",
                schema=_SEACROWD_SCHEMA,
                subset_id=subset,
            ),
        ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_gold_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            if self.config.subset_id == "aws_baseline":
                features = datasets.Features(
                    {
                        "prompt_id": datasets.Value("string"),
                        "source_text": datasets.Value("string"),
                        "translation": datasets.Value("string"),
                    }
                )
            elif self.config.subset_id == "gold":
                features = datasets.Features(
                    {
                        "prompt_id": datasets.Value("string"),
                        "source_text": datasets.Value("string"),
                        "translations": [
                            {
                                "text": datasets.Value("string"),
                                "weight": datasets.Value("float64"),
                            }
                        ],
                    }
                )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]]  # text2text_features

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
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg (staple-2020 dir) to load_dataset.")
        else:
            data_dir = Path(self.config.data_dir) / "en_vi"

        if self.config.subset_id == "aws_baseline":
            filename = "aws_baseline.pred"
        elif self.config.subset_id == "gold":
            filename = "2020-02-20.gold"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir / f"train.en_vi.{'2020-01-13.gold' if self.config.subset_id == 'gold' else filename}.txt",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir / f"test.en_vi.{filename}.txt",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir / f"dev.en_vi.{filename}.txt",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # aws_baseline subset
        if self.config.subset_id == "aws_baseline":
            with open(filepath, "r", encoding="utf-8") as f:
                entries = f.read().strip().split("\n\n")

            for key, entry in enumerate(entries):
                parts = entry.split("|")
                prompt_id = parts[0].strip()
                source_text, translation = list(map(str.strip, parts[1].split("\n")))

                if self.config.schema == "source":
                    yield key, {
                        "prompt_id": prompt_id,
                        "source_text": source_text,
                        "translation": translation,
                    }
                elif self.config.schema == _SEACROWD_SCHEMA:
                    yield key, {
                        "id": str(key),
                        "text_1": source_text,
                        "text_2": translation,
                        "text_1_name": "english",
                        "text_2_name": "translation",
                    }

        # gold subset
        elif self.config.subset_id == "gold":
            with open(filepath, "r", encoding="utf-8") as f:
                entries = f.read().strip().split("\n\n")

            key = 0
            for entry in entries:
                parts = entry.split("\n")
                prompt_id, source_text = list(map(str.strip, parts[0].split("|")))

                if self.config.schema == "source":
                    translations = []
                    for answer in parts[1:]:
                        translation, weight = list(map(str.strip, answer.split("|")))
                        translations.append({"text": translation, "weight": float(weight)})
                    yield key, {
                        "prompt_id": prompt_id,
                        "source_text": source_text,
                        "translations": translations,
                    }
                    key += 1

                elif self.config.schema == _SEACROWD_SCHEMA:
                    for answer in parts[1:]:
                        translation, _ = list(map(str.strip, answer.split("|")))
                        yield key, {
                            "id": str(key),
                            "text_1": source_text,
                            "text_2": translation,
                            "text_1_name": "english",
                            "text_2_name": "translation",
                        }
                        key += 1
