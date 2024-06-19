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

import conllu
import datasets

from seacrowd.sea_datasets.vndt.utils import parse_token_and_impute_metadata
from seacrowd.utils import schemas
from seacrowd.utils.common_parser import (load_ud_data,
                                          load_ud_data_as_seacrowd_kb)
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@InProceedings{Nguyen2014NLDB,
  author = {Nguyen, Dat Quoc  and  Nguyen, Dai Quoc  and  Pham, Son Bao and Nguyen, Phuong-Thai and Nguyen, Minh Le},
  title = {{From Treebank Conversion to Automatic Dependency Parsing for Vietnamese}},
  booktitle = {{Proceedings of 19th International Conference on Application of Natural Language to Information Systems}},
  year = {2014},
  pages = {196-207},
  url = {https://github.com/datquocnguyen/VnDT},
}
"""

_DATASETNAME = "vndt"

_DESCRIPTION = """\
VnDT is a Vietnamese dependency treebank, consisting of 10K+ sentences (219k words). The VnDT Treebank is automatically
converted from the input Vietnamese Treebank.
"""

_HOMEPAGE = "https://github.com/datquocnguyen/VnDT"

_LANGUAGES = {"vie": "vi"}

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "gold-dev": "https://raw.githubusercontent.com/datquocnguyen/VnDT/master/VnDTv1.1-gold-POS-tags-dev.conll",
    "gold-test": "https://raw.githubusercontent.com/datquocnguyen/VnDT/master/VnDTv1.1-gold-POS-tags-test.conll",
    "gold-train": "https://raw.githubusercontent.com/datquocnguyen/VnDT/master/VnDTv1.1-gold-POS-tags-train.conll",
    "predicted-dev": "https://raw.githubusercontent.com/datquocnguyen/VnDT/master/VnDTv1.1-predicted-POS-tags-dev.conll",
    "predicted-test": "https://raw.githubusercontent.com/datquocnguyen/VnDT/master/VnDTv1.1-predicted-POS-tags-test.conll",
    "predicted-train": "https://raw.githubusercontent.com/datquocnguyen/VnDT/master/VnDTv1.1-predicted-POS-tags-train.conll",
}

_SUPPORTED_TASKS = [Tasks.DEPENDENCY_PARSING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

class VnDTDataset(datasets.GeneratorBasedBuilder):
    """
    VnDT is a Vietnamese dependency treebank from https://github.com/datquocnguyen/VnDT.
    """

    # Override conllu.parse_token_and_metadata via monkey patching
    conllu.parse_token_and_metadata = parse_token_and_impute_metadata

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_gold_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} gold standard source schema",
            schema="source",
            subset_id="gold",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_gold_seacrowd_kb",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} gold standard SEACrowd schema",
            schema="seacrowd_kb",
            subset_id="gold",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_predicted_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} predicted source schema",
            schema="source",
            subset_id="predicted",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_predicted_seacrowd_kb",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} predicted SEACrowd schema",
            schema="seacrowd_kb",
            subset_id="predicted",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Sequence(datasets.Value("int8")),
                    "form": datasets.Sequence(datasets.Value("string")),
                    "lemma": datasets.Sequence(datasets.Value("string")),
                    "upos": datasets.Sequence(datasets.Value("string")),
                    "xpos": datasets.Sequence(datasets.Value("string")),
                    "feats": datasets.Sequence(datasets.Value("string")),
                    "head": datasets.Sequence(datasets.Value("int8")),
                    "deprel": datasets.Sequence(datasets.Value("string")),
                    "deps": datasets.Sequence(datasets.Value("string")),
                    "misc": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == "seacrowd_kb":
            features = schemas.kb_features
        else:
            raise ValueError(f"Invalid schema: '{self.config.schema}'")

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

        paths = {key: dl_manager.download_and_extract(value) for key, value in _URLS.items()}

        if self.config.subset_id == "gold":
            filtered_paths = {key: value for key, value in paths.items() if "gold" in key}
        elif self.config.subset_id == "predicted":
            filtered_paths = {key: value for key, value in paths.items() if "predicted" in key}
        else:
            raise NotImplementedError(f"Invalid subset: '{self.config.subset_id}'.")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": [value for key, value in filtered_paths.items() if "dev" in key],
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": [value for key, value in filtered_paths.items() if "test" in key],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": [value for key, value in filtered_paths.items() if "train" in key],
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepaths: Path, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """

        dataset = None
        for file in filepaths:
            if self.config.schema == "source":
                dataset = list(load_ud_data(file))
            elif self.config.schema == "seacrowd_kb":
                dataset = list(load_ud_data_as_seacrowd_kb(file, dataset))
            else:
                raise ValueError(f"Invalid config: '{self.config.name}'")

        for idx, example in enumerate(dataset):
            if self.config.schema == "source":
                example.pop('sent_id', None)
                example.pop('text', None)
            yield idx, example