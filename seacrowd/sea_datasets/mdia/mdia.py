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

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@misc{zhang2022mdia,
    title={MDIA: A Benchmark for Multilingual Dialogue Generation in 46 Languages},
    author={Qingyu Zhang and Xiaoyu Shen and Ernie Chang and Jidong Ge and Pengke Chen},
    year={2022},
    eprint={2208.13078},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DATASETNAME = "mdia"

_DESCRIPTION = """\
This is a multilingual benchmark for dialogue generation containing real-life
Reddit conversations (parent and response comment pairs) in 46 languages,
including Indonesian, Tagalog and Vietnamese. English translations are also
provided for comments.
"""

_HOMEPAGE = "https://github.com/DoctorDream/mDIA"

_LANGUAGES = ["ind", "tgl", "vie"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    "raw": "https://github.com/DoctorDream/mDIA/raw/master/datasets/raw.zip",
    "translated": "https://github.com/DoctorDream/mDIA/raw/master/datasets/translated.zip",
}

_SUPPORTED_TASKS = [Tasks.DIALOGUE_SYSTEM, Tasks.MACHINE_TRANSLATION]  # DS, MT
_SEACROWD_SCHEMA = {task.value: f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS}  # t2t
_SUBSETS = [
    "ind_dialogue",
    "ind_eng",
    "tgl_dialogue",
    "tgl_eng",
    "vie_dialogue",
    "vie_eng",
]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MdiaDataset(datasets.GeneratorBasedBuilder):
    """Multilingual benchmark for dialogue generation containing real-life Reddit conversations"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in _SUBSETS:
        if "dialogue" in subset:
            BUILDER_CONFIGS += [
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{subset}_source",
                    version=SOURCE_VERSION,
                    description=f"{_DATASETNAME} {subset} source schema",
                    schema="source",
                    subset_id=subset,
                ),
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{subset}_{_SEACROWD_SCHEMA['DS']}",
                    version=SEACROWD_VERSION,
                    description=f"{_DATASETNAME} {subset} SEACrowd schema",
                    schema=_SEACROWD_SCHEMA["DS"],
                    subset_id=subset,
                ),
            ]
        else:
            BUILDER_CONFIGS += [
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{subset}_source",
                    version=SOURCE_VERSION,
                    description=f"{_DATASETNAME} {subset} source schema",
                    schema="source",
                    subset_id=subset,
                ),
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{subset}_{_SEACROWD_SCHEMA['MT']}",
                    version=SEACROWD_VERSION,
                    description=f"{_DATASETNAME} {subset} SEACrowd schema",
                    schema=_SEACROWD_SCHEMA["MT"],
                    subset_id=subset,
                ),
            ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{_SUBSETS[0]}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "lang": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "source_body": datasets.Value("string"),
                    "target_body": datasets.Value("string"),
                    "link_id": datasets.Value("string"),
                    "source_id": datasets.Value("string"),
                    "target_id": datasets.Value("string"),
                    "translated_source_body": datasets.Value("string"),
                    "translated_target_body": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA["DS"]:  # same schema with _SEACROWD_SCHEMA["MT"]
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
        lang_map = {"ind": "id", "tgl": "tl", "vie": "vi"}
        lang = lang_map[self.config.subset_id.split("_")[0]]

        data_url = _URLS["translated"]
        data_dir = Path(dl_manager.download_and_extract(data_url)) / "translated"
        data_path = "{split}_data/{lang}2en_{split}.csv"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_path": data_dir / data_path.format(split="train", lang=lang),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_path": data_dir / data_path.format(split="test", lang=lang),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_path": data_dir / data_path.format(split="eval", lang=lang),
                },
            ),
        ]

    def _generate_examples(self, data_path: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(data_path)

        # source schema
        if self.config.schema == "source":
            for i, row in df.iterrows():
                yield i, {
                    "lang": row["lang"],
                    "title": row["title"],
                    "source_body": row["source_body"],
                    "target_body": row["target_body"],
                    "link_id": row["link_id"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "translated_source_body": row["translated_source_body"],
                    "translated_target_body": row["translated_target_body"],
                }

        # t2t schema for dialogue
        elif "dialogue" in self.config.subset_id:
            for i, row in df.iterrows():
                yield i, {
                    "id": str(i),
                    "text_1": row["source_body"],
                    "text_2": row["target_body"],
                    "text_1_name": "source_body",
                    "text_2_name": "target_body",
                }

        # t2t schema for machine translation
        elif "eng" in self.config.subset_id:
            for i, row in df.iterrows():
                for j in range(2):
                    idx = i * 2 + j
                    if j == 0:
                        yield idx, {
                            "id": str(idx),
                            "text_1": row["source_body"],
                            "text_2": row["translated_source_body"],
                            "text_1_name": "source_body",
                            "text_2_name": "translated_source_body",
                        }
                    else:
                        yield idx, {
                            "id": str(idx),
                            "text_1": row["target_body"],
                            "text_2": row["translated_target_body"],
                            "text_1_name": "target_body",
                            "text_2_name": "translated_target_body",
                        }
