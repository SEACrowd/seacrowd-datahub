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
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@misc{mtet,
  doi = {10.48550/ARXIV.2210.05610},
  url = {https://arxiv.org/abs/2210.05610},
  author = {Ngo, Chinh and Trinh, Trieu H. and Phan, Long and Tran, Hieu and Dang, Tai and Nguyen, Hieu and Nguyen, Minh and Luong, Minh-Thang},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {MTet: Multi-domain Translation for English and Vietnamese},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
"""

_DATASETNAME = "vi_pubmed"

_DESCRIPTION = """\
20M Vietnamese PubMed biomedical abstracts translated by the state-of-the-art English-Vietnamese Translation project. The data has been used as unlabeled dataset for pretraining a Vietnamese Biomedical-domain Transformer model.
"""

_HOMEPAGE = "https://huggingface.co/datasets/VietAI/vi_pubmed"

_LANGUAGES = ["eng", "vie"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.OTHERS.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "pubmed22": [
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/0.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/1.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/2.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/3.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/4.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/5.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/6.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/7.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/8.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/9.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/10.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/11.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/12.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/13.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/14.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/15.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/16.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/17.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/18.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/19.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/20.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/21.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/22.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/23.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/24.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/25.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/26.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/27.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/28.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/29.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/30.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/31.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/32.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/33.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/34.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/35.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/36.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/37.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/38.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/39.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/40.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/41.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/42.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/43.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/44.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/45.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/46.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/47.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/48.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/49.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/50.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/51.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/52.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/53.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/54.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/55.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/56.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/57.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/58.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/59.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/60.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/61.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/62.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/63.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/64.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/65.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/66.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/67.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/68.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/69.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/70.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/71.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/72.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/73.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/74.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/75.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/76.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/77.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/78.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/79.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/80.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/81.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/82.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/83.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/84.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/85.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/86.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/87.parquet",
            "https://huggingface.co/api/datasets/VietAI/vi_pubmed/parquet/default/pubmed22/88.parquet",
        ]
    },
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class ViPubmed(datasets.GeneratorBasedBuilder):
    """20M Vietnamese PubMed biomedical abstracts translated by the state-of-the-art English-Vietnamese Translation project. The data has been used as unlabeled dataset for pretraining a Vietnamese Biomedical-domain Transformer model."""

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
    ]

    seacrowd_schema_config: list[SEACrowdConfig] = []

    for seacrowd_schema in _SUPPORTED_SCHEMA_STRINGS:

        seacrowd_schema_config.append(
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{seacrowd_schema}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {seacrowd_schema} schema",
                schema=f"{seacrowd_schema}",
                subset_id=f"{_DATASETNAME}",
            )
        )

    BUILDER_CONFIGS.extend(seacrowd_schema_config)

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "en": datasets.Value("string"),
                    "vi": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.MACHINE_TRANSLATION]).lower()}":
            features = schemas.text2text_features

        else:
            raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        split_name = "pubmed22"
        paths = dl_manager.download_and_extract(_URLS[_DATASETNAME][split_name])

        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={
                    "paths": paths,
                    "split": split_name,
                },
            ),
        ]

    def _generate_examples(self, paths: list[Path], split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        idx = 0

        if self.config.schema == "source":

            for path in paths:
                df = pd.read_parquet(path)

                for _, row in df.iterrows():
                    yield idx, row.to_dict()
                    idx += 1

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.MACHINE_TRANSLATION]).lower()}":
            for path in paths:
                df = pd.read_parquet(path)

                df["id"] = df.index + idx
                df.rename(columns={"en": "text_1"}, inplace=True)
                df.rename(columns={"vi": "text_2"}, inplace=True)
                df = df.assign(text_1_name="en").astype({"text_1_name": "str"})
                df = df.assign(text_2_name="vi").astype({"text_2_name": "str"})

                for _, row in df.iterrows():
                    yield idx, row.to_dict()
                    idx += 1

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
