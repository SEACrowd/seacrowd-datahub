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
Vintext is a challenging scene text dataset for Vietnamese, where some characters are equivocal in the visual form due to accent symbols.
This dataset contains 1500 fully annotated images from the original format. Each text instance is delineated by a quadrilateral bounding box and associated with the ground truth sequence of characters.
The dataset is randomly split into 2 subsets for training (1,200 images) and testing (300 images).
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@INPROCEEDINGS{vintext,
    author={Nguyen, Nguyen and Nguyen, Thu and Tran, Vinh and Tran, Minh-Triet and Ngo, Thanh Duc and Huu Nguyen, Thien and Hoai, Minh},
    booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    title={Dictionary-guided Scene Text Recognition},
    year={2021},
    pages={7379-7388},
    keywords={Training;Visualization;Computer vision;Casting;Dictionaries;Codes;Text recognition},
    doi={10.1109/CVPR46437.2021.00730}
}
"""

_DATASETNAME = "vintext"

_DESCRIPTION = """\
Vintext is a challenging scene text dataset for Vietnamese, where some characters are equivocal in the visual form due to accent symbols.
This dataset contains 2000 fully annotated images with 56,084 text instances. Each text instance is delineated by a quadrilateral bounding box and associated with the ground truth sequence of characters.
The dataset is randomly split into three subsets for training (1,200 images), validation (300 images), and testing (500 images).
"""

_HOMEPAGE = "https://github.com/VinAIResearch/dict-guided"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.AGPL_3_0.value

_LOCAL = False

_GDRIVE_ID = "1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml"

_SUPPORTED_TASKS = [Tasks.OPTICAL_CHARACTER_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class VintextDataset(datasets.GeneratorBasedBuilder):
    """
    Vintext is a challenging scene text dataset for Vietnamese, where some characters are equivocal in the visual form due to accent symbols.
    This dataset contains 1500 fully annotated images from the original format. Each text instance is delineated by a quadrilateral bounding box and associated with the ground truth sequence of characters.
    The dataset is randomly split into 2 subsets for training (1,200 images) and testing (300 images).
    """

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
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "image_path": datasets.Value("string"),
                    "annotations": datasets.Sequence(
                        {
                            "x1": datasets.Value("int32"),
                            "y1": datasets.Value("int32"),
                            "x2": datasets.Value("int32"),
                            "y2": datasets.Value("int32"),
                            "x3": datasets.Value("int32"),
                            "y3": datasets.Value("int32"),
                            "x4": datasets.Value("int32"),
                            "y4": datasets.Value("int32"),
                            "transcript": datasets.Value("string"),
                        }
                    ),
                }
            )

        elif self.config.schema == "seacrowd_imtext":
            features = schemas.image_text_features()
            features["metadata"]["annotations"] = datasets.Sequence(
                {
                    "x1": datasets.Value("int32"),
                    "y1": datasets.Value("int32"),
                    "x2": datasets.Value("int32"),
                    "y2": datasets.Value("int32"),
                    "x3": datasets.Value("int32"),
                    "y3": datasets.Value("int32"),
                    "x4": datasets.Value("int32"),
                    "y4": datasets.Value("int32"),
                    "transcript": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        try:
            import gdown
        except ImportError as err:
            raise ImportError("You need to install gdown (`pip install gdown`) to downloads a public file/folder from Google Drive.") from err

        zip_filepath = os.path.join(os.path.dirname(__file__), "vietnamese_original.zip")
        if not os.path.exists(zip_filepath):
            gdown.download(id=_GDRIVE_ID, output=zip_filepath)

        data_dir = dl_manager.extract(zip_filepath)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "imagepath": Path(data_dir) / "vietnamese/train_images",
                    "labelpath": Path(data_dir) / "vietnamese/labels",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "imagepath": Path(data_dir) / "vietnamese/test_image",
                    "labelpath": Path(data_dir) / "vietnamese/labels",
                },
            ),
        ]

    def _generate_examples(self, imagepath: Path, labelpath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df_list = []

        for image in os.listdir(imagepath):
            image_id = int(image.split(".")[0][2:])
            label_file = os.path.join(labelpath, f"gt_{image_id}.txt")
            with open(label_file, "r") as f:
                label = f.read().strip()
            df_list.append({"id": image_id, "image_path": os.path.join(imagepath, image), "label": label})

        if self.config.schema == "source":
            for i, row in enumerate(df_list):
                labels = [label.split(",") for label in row["label"].split("\n")]

                yield i, {
                    "id": row["id"],
                    "image_path": row["image_path"],
                    "annotations": [
                        {
                            "x1": label[0],
                            "y1": label[1],
                            "x2": label[2],
                            "y2": label[3],
                            "x3": label[4],
                            "y3": label[5],
                            "x4": label[6],
                            "y4": label[7],
                            "transcript": label[8],
                        }
                        for label in labels
                    ],
                }

        elif self.config.schema == "seacrowd_imtext":
            for i, row in enumerate(df_list):
                labels = [label.split(",") for label in row["label"].split("\n")]

                yield i, {
                    "id": row["id"],
                    "image_paths": [row["image_path"]],
                    "texts": None,
                    "metadata": {
                        "context": None,
                        "labels": None,
                        "annotations": [
                            {
                                "x1": label[0],
                                "y1": label[1],
                                "x2": label[2],
                                "y2": label[3],
                                "x3": label[4],
                                "y3": label[5],
                                "x4": label[6],
                                "y4": label[7],
                                "transcript": label[8],
                            }
                            for label in labels
                        ],
                    },
                }
