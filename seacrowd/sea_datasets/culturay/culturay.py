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

import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import zstandard as zstd
from huggingface_hub import HfFileSystem

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@misc{nguyen2024culturay,
      title={CulturaY: A Large Cleaned Multilingual Dataset of 75 Languages}, 
      author={Thuat Nguyen, Huu Nguyen and Thien Nguyen},
      year={2024},
}
"""

_DATASETNAME = "culturay"

_DESCRIPTION = """\
CulturaY: A Large Cleaned Multilingual Dataset of 75 Languages From the team
that brought you CulutraX, we present CulturaY, another substantial multilingual
dataset of 15TB (uncompressed)/3TB (zstd-compressed) that applies the same
dataset cleaning methodology to the HPLT v1.1 dataset. Please note that HPLT
v1.2 has also been released and is an alternative verison with different
cleaning methodolgies. This data was used in part to train our SOTA Vietnamese
model: Vistral-7B-Chat.

Before using this dataloader, please accept the acknowledgement at
https://huggingface.co/datasets/ontocord/CulturaY and use huggingface-cli login
for authentication.
"""

_HOMEPAGE = "https://huggingface.co/datasets/ontocord/CulturaY"

_LANGUAGES = ["mya", "fil", "zlm", "vie", "ind", "tha"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_BASE_URL = "https://huggingface.co/datasets/ontocord/CulturaY/resolve/main/{lang}/"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # ssp

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class CulturaYDataset(datasets.GeneratorBasedBuilder):
    """Substantial multilingual dataset by cleaning HPLT v1.1 (Internet Archive) data."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in _LANGUAGES:
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

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_my_source"  # smallest wrt n_doc

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "document_lang": datasets.Value("string"),
                    "scores": datasets.Sequence(datasets.Value("float64")),
                    "langs": datasets.Sequence(datasets.Value("string")),
                    "text": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "collection": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[
                TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]
            ]  # ssp_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators. Data is not yet extracted for efficient generation."""
        lang_dict = {"mya": "my", "fil": "tl", "zlm": "ms", "vie": "vi", "ind": "id", "tha": "th"}
        subset = lang_dict[self.config.subset_id]
        base_path = _BASE_URL.format(lang=subset)

        fs = HfFileSystem(token=dl_manager.download_config.token)
        file_list = fs.ls(f"datasets/ontocord/CulturaY/{subset}", detail=False)

        data_urls = [
            f"{base_path}{filename.split('/')[-1]}"
            for filename in file_list
            if filename.endswith(".jsonl.zst")
        ]

        data_paths = list(map(Path, dl_manager.download(data_urls)))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_paths": data_paths,
                },
            ),
        ]

    def _generate_examples(self, data_paths: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        key = 0
        for data_path in data_paths:
            with open(data_path, "rb") as f:
                # Zstandard decompression
                dctx = zstd.ZstdDecompressor()
                reader = dctx.stream_reader(f)
                text_io = io.TextIOWrapper(reader, encoding="utf-8")

                # read jsonl file by line and yield
                for line in text_io:
                    data = json.loads(line)
                    if self.config.schema == "source":
                        yield key, {
                            "id": data["id"],
                            "document_lang": data["document_lang"],
                            "scores": data["scores"],
                            "langs": data["langs"],
                            "text": data["text"],
                            "url": data["url"],
                            "collection": data["collection"],
                        }
                    elif self.config.schema == _SEACROWD_SCHEMA:
                        yield key, {
                            "id": str(data["id"]),
                            "text": data["text"],
                        }
                    key += 1
