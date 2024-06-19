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
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@misc{zhu2023extrapolating,
    title={Extrapolating Large Language Models to Non-English by Aligning Languages},
    author={Wenhao Zhu and Yunzhe Lv and Qingxiu Dong and Fei Yuan and Jingjing Xu and Shujian Huang and Lingpeng Kong and Jiajun Chen and Lei Li},
    year={2023},
    eprint={2308.04948},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DATASETNAME = "multilingual_alpaca"

_DESCRIPTION = """\
For multilingual general task instruction data, we incorporate ALPACA dataset
(Taori et al., 2023), which consists of 52k English questions and corresponding
response, and we obtain its foreign version with in-house translation engine.
The six languages are Arabic (Ar), Greek (El), Hindi (Hi), Turkish (Tr),
Vietnamese (Vi), Chinese (Zh).
"""

_HOMEPAGE = "https://github.com/NJUNLP/x-LLM"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {_DATASETNAME: "https://drive.google.com/file/d/1bkejieKDJFDJ45UmQYiY4eeqpGBwj-r-/view"}  # ~660mb

_SUPPORTED_TASKS = [Tasks.INSTRUCTION_TUNING]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # t2t

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MultilingualAlpacaDataset(datasets.GeneratorBasedBuilder):
    """Translated Alpaca Dataset for Vietnamese language."""

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
            name=f"{_DATASETNAME}_{_SEACROWD_SCHEMA}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=_SEACROWD_SCHEMA,
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
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
        # check if gdown is installed
        try:
            import gdown
        except ImportError:
            raise ImportError("Please install `gdown` to enable downloading data from google drive.")

        # download data from gdrive
        output_dir = Path.cwd() / "data" / "multilingual_alpaca"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "multilingual_alpaca.zip"
        if not output_file.exists():
            gdown.download(_URLS[_DATASETNAME], str(output_file), fuzzy=True)
        else:
            print(f"File already downloaded: {str(output_file)}")

        # extract data
        data_dir = Path(dl_manager.extract(output_file))
        data_path = data_dir / "alpaca" / "alpaca_vi.json"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_path": data_path,
                },
            ),
        ]

    def _generate_examples(self, data_path: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # load data
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # generate examples
        for idx, example in enumerate(data):
            if self.config.schema == "source":
                yield idx, {
                    "instruction": example["instruction"],
                    "input": example["input"],
                    "output": example["output"],
                }
            elif self.config.schema == _SEACROWD_SCHEMA:
                input_ = example["input"]
                if input_:
                    text_1 = f"Instruction: {example['instruction']}\nInput: {input_}"
                else:
                    text_1 = f"Instruction: {example['instruction']}"

                yield idx, {
                    "id": str(idx),
                    "text_1": text_1,
                    "text_2": example["output"],
                    "text_1_name": "instruction_and_input",
                    "text_2_name": "output",
                }
