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

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{bunagtransformer,
  author={Bunag, Kenrick Lance T and Esquivel, Rosanna A}
  title={Transformer-Based Conditional Language Models to Generate Filipino News Articles},
  year = {2023},
  publisher = {IEOM Society International},
  url = {https://ieomsociety.org/proceedings/2023manila/595.pdf},
  booktitle = {Proceedings of the International Conference on Industrial Engineering and Operations Management},
  pages = {2231–2237},
  numpages = {7},
  location = {Manila, Philippines},
}
"""

_DATASETNAME = "balita_nlp"

_DESCRIPTION = """\
BalitaNLP is a dataset for image-conditional language generation and text-conditional image generation. It consists of 300k Filipino news
articles and images gathered from Filipino news outlets. News articles are categorized into five possible classes: News, Sports, Entertainment,
Crime, and Other. Some articles were removed from the SEACrowd `imtext` schema, as their corresponding image files do not exist:
- `train` split (262480 total articles): from the original 281403 articles, 18923 (~6.72%) had missing images
- `test` split (32821 total articles): from the original 35177 articles, 2356 (~6.70%) had missing images
- `validation` split (32806 total articles): from the original 35175 articles, 2369 (~6.73%) had missing images
"""

_HOMEPAGE = "https://github.com/KenrickLance/BalitaNLP-Dataset"

_LANGUAGES = ["fil"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "text": "https://storage.googleapis.com/public-kenricklancebunag/BalitaNLP/2022/BalitaNLP-Dataset.zip",
    "images": {
        "part1": "https://storage.googleapis.com/public-kenricklancebunag/BalitaNLP/2022/BalitaNLP-images_1.zip",
        "part2": "https://storage.googleapis.com/public-kenricklancebunag/BalitaNLP/2022/BalitaNLP-images_2.zip",
        "part3": "https://storage.googleapis.com/public-kenricklancebunag/BalitaNLP/2022/BalitaNLP-images_3.zip",
        "part4": "https://storage.googleapis.com/public-kenricklancebunag/BalitaNLP/2022/BalitaNLP-images_4.zip",
    },
}

_SUPPORTED_TASKS = [Tasks.IMAGE_CAPTIONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class BalitaNLPDataset(datasets.GeneratorBasedBuilder):
    """
    BalitaNLP is an image-text dataset from https://github.com/KenrickLance/BalitaNLP-Dataset.
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
            name=f"{_DATASETNAME}_seacrowd_imtext",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_imtext",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "body": datasets.Sequence(datasets.Value("string")),
                    "title": datasets.Value("string"),
                    "website": datasets.Value("string"),
                    "category": datasets.Value("string"),
                    "date": datasets.Value("string"),
                    "author": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "img_url": datasets.Value("string"),
                    "img_path": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_imtext":
            features = schemas.image_text_features()
            features["metadata"] = {
                "context": datasets.Value("string"),
                "author": datasets.Value("string"),
                "category": datasets.Value("string"),
                "date": datasets.Value("string"),
                "img_url": datasets.Value("string"),
                "url": datasets.Value("string"),
                "website": datasets.Value("string"),
            }
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

        text_path = dl_manager.download_and_extract(_URLS["text"])
        img_paths = dl_manager.download_and_extract([v for k, v in _URLS["images"].items()])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "text_path": os.path.join(text_path, "train.json"),
                    "img_paths": img_paths,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "text_path": os.path.join(text_path, "test.json"),
                    "img_paths": img_paths,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "text_path": os.path.join(text_path, "validation.json"),
                    "img_paths": img_paths,
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, text_path: Path, img_paths: Path, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """
        text_data = pd.read_json(text_path)
        data = text_data.to_records()

        for idx, row in enumerate(data):

            # Search for path of image file
            img_path = ""
            for idx_subpath, img_subpath in enumerate(img_paths):
                candidate_filepath = os.path.join(img_subpath, "part" + str(idx_subpath + 1), row["img_path"])
                if os.path.isfile(candidate_filepath):
                    img_path = candidate_filepath

            if self.config.schema == "source":
                x = {
                    "body": row["body"],
                    "title": row["title"],
                    "website": row["website"],
                    "category": row["category"],
                    "date": row["date"],
                    "author": row["author"],
                    "url": row["url"],
                    "img_url": row["img_url"],
                    "img_path": img_path,
                }
                yield idx, x

            elif self.config.schema == "seacrowd_imtext":

                # Remove examples with no existing image path
                if img_path == "":
                    continue

                x = {
                    "id": idx,
                    "image_paths": [img_path],
                    "texts": row["title"],
                    "metadata": {
                        "context": row["body"],
                        "author": row["author"],
                        "category": row["category"],
                        "date": row["date"],
                        "img_url": row["img_url"],
                        "url": row["url"],
                        "website": row["website"],
                    },
                }
                yield idx, x

            else:
                raise ValueError(f"Invalid schema: '{self.config.schema}'")
