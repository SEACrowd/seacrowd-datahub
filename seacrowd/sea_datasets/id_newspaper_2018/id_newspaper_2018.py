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
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{feryandi2018,
  author={Nurdiantoro, Feryandi}
  title={Dataset-Artikel},
  year = {2018},
  url = {https://github.com/feryandi/Dataset-Artikel},
}
"""

_DATASETNAME = "id_newspaper_2018"

_DESCRIPTION = """\
The ID Newspapers 2018 dataset provides 500K articles from various Indonesian news sources. Articles were taken from
7 primary sources (Detik, Kompas, Tempo, CNN Indonesia, Sindo, Republika, Poskota). The compressed files can be
retrieved from datahttps://huggingface.co/datasets/indonesian-nlp/id_newspapers_2018.
"""

_HOMEPAGE = "https://github.com/feryandi/Dataset-Artikel"

_LANGUAGES = ["ind"]

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

_URLS = "https://huggingface.co/datasets/indonesian-nlp/id_newspapers_2018/resolve/main/newspapers-json.tgz"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class IDNewspapers2018Dataset(datasets.GeneratorBasedBuilder):
    """
    ID Newspapers 2018 is a pretraining dataset from https://huggingface.co/datasets/indonesian-nlp/id_newspapers_2018.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_ssp",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"url": datasets.Value("string"), "date": datasets.Value("string"), "title": datasets.Value("string"), "content": datasets.Value("string")})
        elif self.config.schema == "seacrowd_ssp":
            features = schemas.ssp_features
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

        path = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": path,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, path: Path, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """
        file_paths = []
        for path, subdirs, files in os.walk(path):
            for name in files:
                if name[-5:] == ".json":
                    file_paths.append(os.path.join(path, name))

        for idx, file_path in enumerate(file_paths):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                if self.config.schema == "source":
                    x = {
                        "url": data["url"],
                        "date": data["date"],
                        "title": data["title"],
                        "content": data["content"],
                    }
                    yield idx, x

                elif self.config.schema == "seacrowd_ssp":
                    x = {
                        "id": str(idx),
                        "text": data["content"],
                    }
                    yield idx, x

                else:
                    raise ValueError(f"Invalid schema: '{self.config.schema}'")
