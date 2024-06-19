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
A high-quality Vietnamese-English parallel dataset constructed specifically for the medical domain, comprising approximately 360K sentence pairs
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{medev,
    title     = {{Improving Vietnamese-English Medical Machine Translation}},
    author    = {Nhu Vo and Dat Quoc Nguyen and Dung D. Le and Massimo Piccardi and Wray Buntine},
    booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING)},
    year      = {2024}
}
"""

_DATASETNAME = "medev"

_DESCRIPTION = """\
A high-quality Vietnamese-English parallel dataset constructed specifically for the medical domain, comprising approximately 360K sentence pairs
"""

_HOMEPAGE = "https://huggingface.co/datasets/nhuvo/MedEV"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "train_en": "https://huggingface.co/datasets/nhuvo/MedEV/resolve/main/train.en.txt?download=true", 
    "train_vie": "https://huggingface.co/datasets/nhuvo/MedEV/resolve/main/train.vi.txt?download=true",
    "val_en": "https://huggingface.co/datasets/nhuvo/MedEV/resolve/main/val.en.new.txt?download=true", 
    "val_vie": "https://huggingface.co/datasets/nhuvo/MedEV/resolve/main/val.vi.new.txt?download=true",
    "test_en": "https://huggingface.co/datasets/nhuvo/MedEV/resolve/main/test.en.new.txt?download=true",
    "test_vie": "https://huggingface.co/datasets/nhuvo/MedEV/resolve/main/test.vi.new.txt?download=true",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MedEVDataset(datasets.GeneratorBasedBuilder):
    """A high-quality Vietnamese-English parallel dataset constructed specifically for the medical domain, comprising approximately 360K sentence pairs"""

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
            name=f"{_DATASETNAME}_seacrowd_t2t",
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
                    "id": datasets.Value("string"),
                    "vie_text": datasets.Value("string"),
                    "eng_text": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath_en": data_dir["train_en"],
                    "filepath_vie": data_dir["train_vie"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath_en": data_dir["test_en"],
                    "filepath_vie": data_dir["test_vie"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath_en": data_dir["val_en"],
                    "filepath_vie": data_dir["val_vie"],
                },
            ),
        ]

    def _generate_examples(self, filepath_en: Path, filepath_vie: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath_en, "r", encoding="utf-8") as f:
            en_lines = f.readlines()
        with open(filepath_vie, "r", encoding="utf-8") as f:
            vie_lines = f.readlines()

        if self.config.schema == "source":
            for i in range(len(vie_lines)):
                yield i, {
                    "id": str(i),
                    "vie_text": vie_lines[i],
                    "eng_text": en_lines[i],
                }

        elif self.config.schema == "seacrowd_t2t":
            for i, (en_line, vie_line) in enumerate(list(zip(en_lines, vie_lines))):
                yield i, {
                    "id": str(i),
                    "text_1": en_line,
                    "text_2": vie_line,
                    "text_1_name": "eng",
                    "text_2_name": "vie",
                }
