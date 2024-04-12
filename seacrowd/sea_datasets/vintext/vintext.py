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
Vintext is a challenging scene text dataset for Vietnamese, where some characters are equivocal in the visual form due to accent symbols. This dataset contains 2000 fully annotated images with 56,084 text instances. Each text instance is delineated by a quadrilateral bounding box and associated with the ground truth sequence of characters. The dataset is randomly split into three subsets for training (1,200 images), validation (300 images), and testing (500 images).
"""
import os
import requests
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@inproceedings{m_Nguyen-etal-CVPR21,
    author = {Nguyen Nguyen and Thu Nguyen and Vinh Tran and Triet Tran and Thanh Ngo and Thien Nguyen and Minh Hoai},
    title = {Dictionary-guided Scene Text Recognition},
    year = {2021},
    booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
}
"""

_DATASETNAME = "vintext"

_DESCRIPTION = """\
Vintext is a challenging scene text dataset for Vietnamese, where some characters are equivocal in the visual form due to accent symbols. This dataset contains 2000 fully annotated images with 56,084 text instances. Each text instance is delineated by a quadrilateral bounding box and associated with the ground truth sequence of characters. The dataset is randomly split into three subsets for training (1,200 images), validation (300 images), and testing (500 images).
"""

_HOMEPAGE = "https://github.com/VinAIResearch/dict-guided"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.AGPL_3_0.value

_LOCAL = False

_GDRIVE_ID = "1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml"
_URLS = "https://drive.google.com/uc?export=download&confirm=yTib&id=1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml"

_SUPPORTED_TASKS = [Tasks.OPTICAL_CHARACTER_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

class VintextDataset(datasets.GeneratorBasedBuilder):
    """Vintext is a challenging scene text dataset for Vietnamese, where some characters are equivocal in the visual form due to accent symbols. This dataset contains 2000 fully annotated images with 56,084 text instances. Each text instance is delineated by a quadrilateral bounding box and associated with the ground truth sequence of characters. The dataset is randomly split into three subsets for training (1,200 images), validation (300 images), and testing (500 images)."""

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
            name=f"{_DATASETNAME}_seacrowd_imtext",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_imtext",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({
                "id": datasets.Value("string"),
                "image_path": datasets.Value("string"),
                "annotations": datasets.Sequence({
                    "x1": datasets.Value("int32"),
                    "y1": datasets.Value("int32"),
                    "x2": datasets.Value("int32"),
                    "y2": datasets.Value("int32"),
                    "x3": datasets.Value("int32"),
                    "y3": datasets.Value("int32"),
                    "x4": datasets.Value("int32"),
                    "y4": datasets.Value("int32"),
                    "transcript": datasets.Value("string"),
                })
            })

        # For example seacrowd_kb, seacrowd_t2t
        elif self.config.schema == "seacrowd_imtext":
            features = schemas.image_text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _gdrive_large_file_get_url(self, base_url: str, base_id: str):
        """Get the final download URL for large files on Google Drive."""
        # Based on https://stackoverflow.com/a/39225039
        session = requests.Session()
        response = session.get(base_url, stream=True)
        token = None

        print(response.cookies.items())
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
        print("GENERATED TOKEN: ", token)
        if token:
            params = {"id": base_id, "confirm": token}
            response = session.get(base_url, params=params, stream=True)
        return response.url

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        direct_url = self._gdrive_large_file_get_url(_URLS, _GDRIVE_ID)

        print(f"Downloading and extracting {direct_url}")
        data_dir = dl_manager.download_and_extract(direct_url)

        print(data_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images": os.path.join(data_dir, "train_images"),
                    "labels": os.path.join(data_dir, "labels"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "images": os.path.join(data_dir, "test_images"),
                    "labels": os.path.join(data_dir, "labels"),
                },
            )
        ]

    def _generate_examples(self, images: Path, labels: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        
        df_list = []
        for image_file in images.iterdir():
            image_id = image_file.stem
            label_file = labels / f"gt_{image_id}.txt"
            with open(label_file, "r") as f:
                label = f.read().strip()
            df_list.append({"id": image_id, "image_path": str(image_file), "label": label})

        if self.config.schema == "source":
            for i, row in enumerate(df_list):

                yield i, {
                    "id": row["id"],
                    "image_path": row["image_path"],
                    "annotations": [
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "x3": x3,
                            "y3": y3,
                            "x4": x4,
                            "y4": y4,
                            "transcript": transcript,
                        }
                        for x1, y1, x2, y2, x3, y3, x4, y4, transcript in row["label"].split("\n")
                    ]
                }

        elif self.config.schema == "seacrowd_imtext":
            for i, row in enumerate(df_list):

                yield i, {
                    "id": row["id"],
                    "image_path": row["image_path"],
                    "annotations": [
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "x3": x3,
                            "y3": y3,
                            "x4": x4,
                            "y4": y4,
                            "transcript": transcript,
                        }
                        for x1, y1, x2, y2, x3, y3, x4, y4, transcript in row["label"].split("\n")
                    ]
                }


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
