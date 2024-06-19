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
The TyDIQA_ID-NLI dataset is derived from the TyDIQA_ID \
    question answering dataset, utilizing named \
    entity recognition (NER), chunking tags, \
    Regex, and embedding similarity techniques \
    to determine its contradiction sets. \
    Collected through this process, \
    the dataset comprises various columns beyond \
    premise, hypothesis, and label, including \
    properties aligned with NER and chunking tags. \
    This dataset is designed to facilitate Natural\
    Language Inference (NLI) tasks and contains \
    information extracted from diverse sources \
    to provide comprehensive coverage. Each data \
    instance encapsulates premise, hypothesis, label, \
    and additional properties pertinent to NLI evaluation.
"""
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

# The workshop submission at 18 April. I will change this _CITATION on that day.
_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "tydiqa_id_nli"

_DESCRIPTION = """
The TyDIQA_ID-NLI dataset is derived from the TyDIQA_ID \
    question answering dataset, utilizing named \
    entity recognition (NER), chunking tags, \
    Regex, and embedding similarity techniques \
    to determine its contradiction sets. \
    Collected through this process, \
    the dataset comprises various columns beyond \
    premise, hypothesis, and label, including \
    properties aligned with NER and chunking tags. \
    This dataset is designed to facilitate Natural\
    Language Inference (NLI) tasks and contains \
    information extracted from diverse sources \
    to provide comprehensive coverage. Each data \
    instance encapsulates premise, hypothesis, label, \
    and additional properties pertinent to NLI evaluation.
"""

_HOMEPAGE = "https://huggingface.co/datasets/muhammadravi251001/tydiqaid-nli"

_LANGUAGES = ["ind"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "train": "https://huggingface.co/datasets/muhammadravi251001/tydiqaid-nli/resolve/main/tydi-qa-id_nli_train_df.csv?download=true",
    "val": "https://huggingface.co/datasets/muhammadravi251001/tydiqaid-nli/raw/main/tydi-qa-id_nli_val_df.csv",
    "test": "https://huggingface.co/datasets/muhammadravi251001/tydiqaid-nli/raw/main/tydi-qa-id_nli_test_df.csv",
}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class TyDIQAIDNLIDataset(datasets.GeneratorBasedBuilder):
    """
    The TyDIQA_ID-NLI dataset is derived from the TyDIQA_ID \
    question answering dataset, utilizing named \
    entity recognition (NER), chunking tags, \
    Regex, and embedding similarity techniques \
    to determine its contradiction sets. \
    Collected through this process, \
    the dataset comprises various columns beyond \
    premise, hypothesis, and label, including \
    properties aligned with NER and chunking tags. \
    This dataset is designed to facilitate Natural\
    Language Inference (NLI) tasks and contains \
    information extracted from diverse sources \
    to provide comprehensive coverage. Each data \
    instance encapsulates premise, hypothesis, label, \
    and additional properties pertinent to NLI evaluation.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_pairs",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_pairs",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"
    labels = ["entailment", "neutral", "contradiction"]

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=self.labels),
                }
            )

        elif self.config.schema == "seacrowd_pairs":
            features = schemas.pairs_features(self.labels)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        train_path = dl_manager.download_and_extract(_URLS["train"])
        val_path = dl_manager.download_and_extract(_URLS["val"])
        test_path = dl_manager.download_and_extract(_URLS["test"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_path,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": val_path,
                    "split": "val",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath, encoding="utf-8") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for id, row in enumerate(csv_reader):
                    yield id, {"premise": row["premise"], "hypothesis": row["hypothesis"], "label": row["label"]}

        elif self.config.schema == "seacrowd_pairs":
            with open(filepath, encoding="utf-8") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for id, row in enumerate(csv_reader):
                    yield id, {"id": str(id), "text_1": row["premise"], "text_2": row["hypothesis"], "label": row["label"]}


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
