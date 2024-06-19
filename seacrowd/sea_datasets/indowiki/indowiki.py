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

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses

_CITATION = """\
@INPROCEEDINGS{ramli2022indokepler,
    author={Ramli, Inigo and Krisnadhi, Adila Alfa and Prasojo, Radityo Eko},
    booktitle={2022 7th International Workshop on Big Data and Information Security (IWBIS)},
    title={IndoKEPLER, IndoWiki, and IndoLAMA: A Knowledge-enhanced Language Model, Dataset, and Benchmark for the Indonesian Language},
    year={2022},
    volume={},
    number={},
    pages={19-26},
    doi={10.1109/IWBIS56557.2022.9924844}}
"""

_DATASETNAME = "indowiki"
_DESCRIPTION = """\
IndoWiki is a knowledge-graph dataset taken from WikiData and aligned with Wikipedia Bahasa Indonesia as it's corpus.
"""
_HOMEPAGE = "https://github.com/IgoRamli/IndoWiki"
_LANGUAGES = ["ind"]
_LICENSE = Licenses.MIT.value
_LOCAL = False

_URLS = {
    "inductive": {
        "train": "https://drive.google.com/uc?export=download&id=1S3vNx9By5CWKGkObjtXaI6Jr4xri2Tz3",
        "valid": "https://drive.google.com/uc?export=download&id=1cP-zDIxp9a-Bw9uYd40K9IN-4wg4dOgy",
        "test": "https://drive.google.com/uc?export=download&id=1pLcoJgYmgQiN4Gv9tRcI26zM7-OgHcuZ",
    },
    "transductive": {
        "train": "https://drive.google.com/uc?export=download&id=1KXDVwboo1h2yk_kAqv7IPYnHXCK6g-6X",
        "valid": "https://drive.google.com/uc?export=download&id=1eRwpuRPYOnA-7FZ-YNZjRJ2DHuJsfUIE",
        "test": "https://drive.google.com/uc?export=download&id=1cy9FwDMB_U-js8P8u4IWolvNeIFkQVDh",
    },
    "text": "https://drive.usercontent.google.com/download?id=1YC4P_IPSo1AsEwm5Z_4GBjDdwCbvokxX&export=download&authuser=0&confirm=t&uuid=36aa95f5-e1b6-43c1-a34f-754d14d8b473&at=APZUnTWD7fwarBs4ZVRy_QdKbDXi%3A1709478240158",
}

# none of the tasks in schema
# dataset is used to learn knowledge embedding
_SUPPORTED_TASKS = []

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class IndoWiki(datasets.GeneratorBasedBuilder):
    """IndoWiki knowledge base dataset from https://github.com/IgoRamli/IndoWiki"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        # inductive setting
        SEACrowdConfig(
            name=f"{_DATASETNAME}_inductive_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        # transductive setting
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "ent1": datasets.Value("string"),
                    "ent2": datasets.Value("string"),
                    "ent1_text": datasets.Value("string"),
                    "ent2_text": datasets.Value("string"),
                    "relation": datasets.Value("string"),
                }
            )

        else:
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        if "inductive" in self.config.name:
            setting = "inductive"
            data_paths = {
                "inductive": {
                    "train": Path(dl_manager.download_and_extract(_URLS["inductive"]["train"])),
                    "valid": Path(dl_manager.download_and_extract(_URLS["inductive"]["valid"])),
                    "test": Path(dl_manager.download_and_extract(_URLS["inductive"]["test"])),
                },
                "text": Path(dl_manager.download_and_extract(_URLS["text"])),
            }
        else:
            setting = "transductive"
            data_paths = {
                "transductive": {
                    "train": Path(dl_manager.download_and_extract(_URLS["transductive"]["train"])),
                    "valid": Path(dl_manager.download_and_extract(_URLS["transductive"]["valid"])),
                    "test": Path(dl_manager.download_and_extract(_URLS["transductive"]["test"])),
                },
                "text": Path(dl_manager.download_and_extract(_URLS["text"])),
            }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "triplets_filepath": data_paths[setting]["train"],
                    "text_filepath": data_paths["text"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "triplets_filepath": data_paths[setting]["test"],
                    "text_filepath": data_paths["text"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "triplets_filepath": data_paths[setting]["valid"],
                    "text_filepath": data_paths["text"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, triplets_filepath: Path, text_filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        # read triplets file
        with open(triplets_filepath, "r", encoding="utf-8") as triplets_file:
            triplets_data = triplets_file.readlines()
        triplets_data = [s.strip("\n").split("\t") for s in triplets_data]

        # read text description file
        with open(text_filepath, "r", encoding="utf-8") as text_file:
            text_data = text_file.readlines()
        # dictionary of entity: text description of entity
        text_dict = {s.split("\t")[0]: s.split("\t")[1].strip("\n") for s in text_data}

        num_sample = len(triplets_data)

        for i in range(num_sample):
            if self.config.schema == "source":
                example = {
                    "id": str(i),
                    "ent1": triplets_data[i][0],
                    "ent2": triplets_data[i][2],
                    "ent1_text": text_dict[triplets_data[i][0]],
                    "ent2_text": text_dict[triplets_data[i][2]],
                    "relation": triplets_data[i][1],
                }

            yield i, example
