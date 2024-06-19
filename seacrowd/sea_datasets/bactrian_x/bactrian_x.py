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

import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses, TASK_TO_SCHEMA, SCHEMA_TO_FEATURES

_CITATION = """\
@misc{li2023bactrianx,
      title={Bactrian-X : A Multilingual Replicable Instruction-Following Model with Low-Rank Adaptation}, 
      author={Haonan Li and Fajri Koto and Minghao Wu and Alham Fikri Aji and Timothy Baldwin},
      year={2023},
      eprint={2305.15011},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "bactrian_x"

_DESCRIPTION = """\
The Bactrain-X dataset is a collection of 3.4M instruction-response pairs in 52
languages, that are obtained by translating 67K English instructions (alpaca-52k
+ dolly-15k) into 51 languages using Google Translate API. The translated
instructions are then fed to ChatGPT (gpt-3.5-turbo) to obtain its natural
responses, resulting in 3.4M instruction-response pairs in 52 languages (52
languages x 67k instances = 3.4M instances). Human evaluations were conducted to
evaluate response quality for several languages, with those of interest to
SEACrowd being Burmese and Tagalog.
"""

_HOMEPAGE = "https://github.com/mbzuai-nlp/Bactrian-X"

_LANGUAGES = ["mya", "tgl", "ind", "khm", "tha", "vie"]

_LICENSE = Licenses.CC_BY_NC_4_0.value

_LOCAL = False

_BASE_URL = "https://huggingface.co/datasets/MBZUAI/Bactrian-X/resolve/main/data/{subset}.json.gz?download=true"
_SUBSETS = ["my", "tl", "id", "km", "th", "vi"]

_SUPPORTED_TASKS = [Tasks.INSTRUCTION_TUNING]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # t2t

_SOURCE_VERSION = "1.0.1"

_SEACROWD_VERSION = "2024.06.20"


class BactrianXDataset(datasets.GeneratorBasedBuilder):
    """A collection of translated instruction-response pairs, evaluated with ChatGPT and human."""

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

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_id_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "output": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[
                TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]
            ]  # text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_url = _BASE_URL.format(subset=self.config.name.split("_")[2])
        data_path = Path(dl_manager.download_and_extract(data_url))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_path": data_path,
                },
            )
        ]

    def _generate_examples(self, data_path: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if self.config.schema == "source":
            for idx, example in enumerate(data):
                yield idx, {
                    "instruction": example["instruction"],
                    "input": example["input"],
                    "id": example["id"],
                    "output": example["output"],
                }
        elif self.config.schema == _SEACROWD_SCHEMA:
            for idx, example in enumerate(data):
                yield idx, {
                    "id": example["id"],
                    "text_1": f"Instruction: {example['instruction']}\nInput: {example['input']}",
                    "text_2": example["output"],
                    "text_1_name": "instruction + input",
                    "text_2_name": "output",
                }
