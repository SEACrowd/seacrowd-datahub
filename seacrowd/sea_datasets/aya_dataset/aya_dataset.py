# coding=utf-8
# Copyright 2024 The HuggingFace Datasets Authors and the current dataset script contributor.
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
The Aya Dataset is a multilingual instruction fine-tuning dataset curated by an open-science community via Aya Annotation Platform from Cohere For AI. The dataset contains a total of 204k human-annotated prompt-completion pairs along with the demographics data of the annotators. This dataset can be used to train, finetune, and evaluate multilingual LLMs.
"""

from pathlib import Path
from typing import List

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{singh2024aya,
      title={Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning},
      author={Shivalika Singh and Freddie Vargus and Daniel Dsouza and Börje F. Karlsson and Abinaya Mahendiran and Wei-Yin Ko and Herumb Shandilya and Jay Patel and Deividas Mataciunas and Laura OMahony and Mike Zhang and Ramith Hettiarachchi and Joseph Wilson and Marina Machado and Luisa Souza Moura and Dominik Krzemiński and Hakimeh Fadaei and Irem Ergün and Ifeoma Okoh and Aisha Alaagib and Oshan Mudannayake and Zaid Alyafeai and Vu Minh Chien and Sebastian Ruder and Surya Guthikonda and Emad A. Alghamdi and Sebastian Gehrmann and Niklas Muennighoff and Max Bartolo and Julia Kreutzer and Ahmet Üstün and Marzieh Fadaee and Sara Hooker},
      year={2024},
      eprint={2402.06619},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "aya_dataset"

_DESCRIPTION = """\
The Aya Dataset is a multilingual instruction fine-tuning dataset curated by an open-science community via Aya Annotation Platform from Cohere For AI. The dataset contains a total of 204k human-annotated prompt-completion pairs along with the demographics data of the annotators. This dataset can be used to train, finetune, and evaluate multilingual LLMs.
"""

_HOMEPAGE = "https://huggingface.co/datasets/CohereForAI/aya_dataset"

_LANGUAGES = ["ceb", "ind", "jav", "mya", "tam", "fil", "sun", "tha", "vie", "zsm"]

_LICENSE = Licenses.APACHE_2_0.value

_LOCAL = False

# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "train": "https://huggingface.co/datasets/CohereForAI/aya_dataset/resolve/main/data/train-00000-of-00001.parquet",  # test split does not contain SEA languages
}

_SUPPORTED_TASKS = [Tasks.INSTRUCTION_TUNING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_SEACROWD_SCHEMA = "seacrowd_t2t"


def _aya_config_constructor(lang: str, schema: str, version: str) -> SEACrowdConfig:
    return SEACrowdConfig(
        name=f"{_DATASETNAME}_{lang}_{schema}",
        version=version,
        description=f"Aya Dataset {schema} schema",
        schema=schema,
        subset_id=f"Aya {lang}",
    )


class AyaDataset(datasets.GeneratorBasedBuilder):
    """
    The Aya Dataset is a multilingual instruction fine-tuning dataset curated by an open-science community via Aya Annotation Platform from Cohere For AI. The dataset contains a total of 204k human-annotated prompt-completion pairs along with the demographics data of the annotators. This dataset can be used to train, finetune, and evaluate multilingual LLMs.

    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    def _populate_configs():
        configs = [_aya_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGUAGES] + [_aya_config_constructor(lang, _SEACROWD_SCHEMA, _SEACROWD_VERSION) for lang in _LANGUAGES]

        all_lang_source_config = SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=_SOURCE_VERSION,
            description="Aya Dataset source schema",
            schema="source",
            subset_id="Aya",
        )

        all_lang_t2t_config = SEACrowdConfig(
            name=f"{_DATASETNAME}_{_SEACROWD_SCHEMA}",
            version=_SEACROWD_VERSION,
            description=f"Aya Dataset {_SEACROWD_SCHEMA} schema",
            schema=_SEACROWD_SCHEMA,
            subset_id="Aya",
        )

        configs.append(all_lang_source_config)
        configs.append(all_lang_t2t_config)
        return configs

    BUILDER_CONFIGS = _populate_configs()

    DEFAULT_CONFIG_NAME = "aya_dataset_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "inputs": datasets.Value("string"),
                    "targets": datasets.Value("string"),
                    "language": datasets.Value("string"),
                    "language_code": datasets.Value("string"),
                    "annotation_type": datasets.Value("string"),
                    "user_id": datasets.Value("string"),
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

    def get_lang_filter(self, config_name: str):
        # aya_dataset_{lang}_{schema}
        tokens = config_name.split("_")
        if len(tokens) == 0 or len(tokens[2]) != 3:
            return None
        return tokens[2]

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        url = _URLS["train"]
        data_dir = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_path": Path(data_dir),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, data_path: Path, split: str):
        """Yields examples as (key, example) tuples."""

        df = pd.read_parquet(data_path)

        lang_filter = self.get_lang_filter(self.config.name)
        if lang_filter is not None:
            df = df[df["language_code"] == lang_filter]
        else:
            df = df[df["language_code"].isin(_LANGUAGES)]

        if self.config.schema == "source":
            for idx, row in df.iterrows():
                data = row.to_dict()
                yield idx, data

        elif self.config.schema == "seacrowd_t2t":
            for idx, row in df.iterrows():
                sample = {
                    "id": str(idx),
                    "text_1": row["inputs"],
                    "text_2": row["targets"],
                    "text_1_name": "inputs",
                    "text_2_name": "targets",
                }
                yield idx, sample
