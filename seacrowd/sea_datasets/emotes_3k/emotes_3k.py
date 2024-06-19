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
English-Tagalog Parallel Dataset intended for two tasks:
1. Moral Text Classification
2. Instruction Tuning
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{Catapang:2023,
  author    = {Catapang, Jasper Kyle and Visperas, Moses},
  title     = {Emotion-based Morality in Tagalog and English Scenarios (EMoTES-3K): A Parallel Corpus for Explaining (Im)morality of Actions},
  booktitle = {Proceedings of the Joint 3rd NLP4DH and 8th IWCLUL},
  pages     = {1--6},
  month     = {December 1-3},
  year      = {2023},
  organization = {Association for Computational Linguistics},
}
"""

_DATASETNAME = "emotes_3k"

_DESCRIPTION = """\
This dataset is used on the paper "Emotion-based Morality in Tagalog and English Scenarios (EMoTES-3K): A Parallel Corpus for Explaining (Im)morality of Actions"
This dataset is designed for for two tasks:
1. Moral Text Classification
2. Instruction Tuning
"""

_HOMEPAGE = "https://huggingface.co/datasets/NLPinas/EMoTES-3K"

_LANGUAGES = ["tgl"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://huggingface.co/datasets/NLPinas/EMoTES-3K/resolve/main/EMoTES-3K.jsonl?download=true",
}

_SUPPORTED_TASKS = [Tasks.MORALITY_CLASSIFICATION, Tasks.INSTRUCTION_TUNING]  # Roberta moral or immoral classification  # FLAN-T5 Training

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class Emotes3KDatasets(datasets.GeneratorBasedBuilder):
    """
    Emotes3K consists of one human annotated dataset for the purpose of morality classification and instruction tuning.
    """

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
            name=f"{_DATASETNAME}_seacrowd_text",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_eng_seacrowd_text",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_tgl_seacrowd_text",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_eng_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_tgl_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "entry_id": datasets.Value("string"),
                    "Filipino": datasets.Value("string"),
                    "English": datasets.Value("string"),
                    "Annotation": datasets.ClassLabel(names=["Moral", "Immoral"]),
                    "Explanation": datasets.Value("string"),
                    "Personality Traits": datasets.Value("string"),
                    "Topic": datasets.Value("string"),
                    "Topic Name": datasets.Value("string"),
                }
            )
        # For example seacrowd_kb, seacrowd_t2t
        elif self.config.schema == "seacrowd_text":
            features = schemas.text.features(["Moral", "Immoral"])
        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text_to_text.features
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
        path = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": path,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                # Use json.loads to parse each line as a JSON object
                data = json.loads(line.strip())

                if self.config.schema == "source":
                    yield (
                        data["entry_id"],
                        {
                            "entry_id": data["entry_id"],
                            "Filipino": data["Filipino"],
                            "English": data["English"],
                            "Annotation": data["Annotation"],
                            "Explanation": data["Explanation"],
                            "Personality Traits": data["Personality Traits"],
                            "Topic": data["Topic"],
                            "Topic Name": data["Topic Name"],
                        },
                    )
                elif self.config.schema == "seacrowd_text":
                    if "eng" in self.config.name or self.config.name == "emotes_3k_seacrowd_text":
                        yield (
                            data["entry_id"],
                            {
                                "id": data["entry_id"],
                                "text": data["English"],
                                "label": data["Annotation"],
                            },
                        )
                    elif "tgl" in self.config.name:
                        yield (
                            data["entry_id"],
                            {
                                "id": data["entry_id"],
                                "text": data["Filipino"],
                                "label": data["Annotation"],
                            },
                        )
                elif self.config.schema == "seacrowd_t2t":
                    if "eng" in self.config.name or self.config.name == "emotes_3k_seacrowd_t2t":
                        yield (
                            data["entry_id"],
                            {
                                "id": data["entry_id"],
                                "text_1": "Explain the morality of this scenario\n" + data["English"],
                                "text_2": data["Explanation"],
                                "text_1_name": "prompt",
                                "text_2_name": "system",
                            },
                        )
                    elif "tgl" in self.config.name:
                        yield (
                            data["entry_id"],
                            {
                                "id": data["entry_id"],
                                "text_1": "Explain the morality of this scenario\n" + data["Filipino"],
                                "text_2": data["Explanation"],
                                "text_1_name": "prompt",
                                "text_2_name": "system",
                            },
                        )
