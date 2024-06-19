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

import struct
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import numpy as np

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@inproceedings{10.1145/3151509.3151510,
    author = {Valy, Dona and Verleysen, Michel and Chhun, Sophea and Burie, Jean-Christophe},
    title = {A New Khmer Palm Leaf Manuscript Dataset for Document Analysis and Recognition: SleukRith Set},
    year = {2017},
    isbn = {9781450353908},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3151509.3151510},
    doi = {10.1145/3151509.3151510},
    booktitle = {Proceedings of the 4th International Workshop on Historical Document Imaging and Processing},
    pages = {1-6},
    numpages = {6},
    location = {Kyoto, Japan},
    series = {HIP '17}
}
"""

_DATASETNAME = "sleukrith_ocr"

_DESCRIPTION = """\
SleukRith Set is the first dataset specifically created for Khmer palm leaf
manuscripts. The dataset consists of annotated data from 657 pages of digitized
palm leaf manuscripts which are selected arbitrarily from a large collection of
existing and also recently digitized images. The dataset contains three types of
data: isolated characters, words, and lines. Each type of data is annotated with
the ground truth information which is very useful for evaluating and serving as
a training set for common document analysis tasks such as character/text
recognition, word/line segmentation, and word spotting.

The character mapping (per label) is not explained anywhere in the dataset homepage,
thus the labels are simply numbered from 0 to 110, each corresponds to a distinct character.
"""

_HOMEPAGE = "https://github.com/donavaly/SleukRith-Set"

_LANGUAGES = ["khm"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    # this URL corresponds to the raw unprocessed data (whole images); unused in this dataloader
    "sleukrith-set": {
        "images": "https://drive.google.com/uc?export=download&id=19JIxAjjXWuJ7mEyUl5-xRr2B8uOb-GKk",  # 1GB
        "annotated-data": "https://drive.google.com/uc?export=download&id=1Xi5ucRUb1e9TUU-nv2rCUYv2ANVsXYDk",  # 11.7MB
    },
    # this URL corresponds to the processed data (per characters); used in this dataloader
    "isolated-characters": {
        "images_train": "https://drive.google.com/uc?export=download&id=1KXf5937l-Xu_sXsGPuQOgFt4zRaXlSJ5",  # 249MB
        "images_test": "https://drive.google.com/uc?export=download&id=1KSt5AiRIilRryh9GBcxyUUhnbiScdQ-9",  # 199MB
        "labels_train": "https://drive.google.com/uc?export=download&id=1IbmLg-4l-3BtRhprDWWvZjCp7lqap0Z-",  # 442KB
        "labels_test": "https://drive.google.com/uc?export=download&id=1GYcaUInkxtuuQps-qA38u-4zxK7HgrAB",  # 354KB
    },
}

_SUPPORTED_TASKS = [Tasks.OPTICAL_CHARACTER_RECOGNITION]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # imtext

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class SleukRithSet(datasets.GeneratorBasedBuilder):
    """Annotated OCR dataset from 657 pages of digitized Khmer palm leaf manuscripts."""

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
                    "image_path": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=[i for i in range(111)]),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = schemas.image_text_features(label_names=[i for i in range(111)])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def module_exists(self, module_name):
        try:
            __import__(module_name)
        except ImportError:
            return False
        else:
            return True

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # check if gdown is installed
        if self.module_exists("gdown"):
            import gdown
        else:
            raise ImportError("Please install `gdown` to enable downloading data from google drive.")

        # create custom data directory
        data_dir = Path.cwd() / "data" / "sleukrith_ocr"
        data_dir.mkdir(parents=True, exist_ok=True)

        # reliable google drive downloader
        data_paths = {}
        for key, value in _URLS["isolated-characters"].items():
            idx = value.rsplit("=", maxsplit=1)[-1]
            output = f"{data_dir}/{key}"
            data_paths[key] = Path(output)

            if not Path(output).exists():
                gdown.download(id=idx, output=output)
            else:
                print(f"File {output} already exists, skipping download.")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_data": data_paths["images_train"],
                    "label_data": data_paths["labels_train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_data": data_paths["images_test"],
                    "label_data": data_paths["labels_test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, image_data: Path, label_data: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # check if PIL is installed
        if self.module_exists("PIL"):
            from PIL import Image
        else:
            raise ImportError("Please install `pillow` to process images.")

        # load images
        with open(image_data, "rb") as file:
            # read and unpack the first 12 bytes for metadata
            width, height, nb_samples = struct.unpack(">iii", file.read(12))

            images = []
            for _ in range(nb_samples):
                # read and convert binary data to np array and reshape
                image_data = file.read(width * height)
                image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width))
                images.append(image)

        # save images and store path
        image_paths = []
        for i, image in enumerate(images):
            image_dir = Path.cwd() / "data" / "sleukrith_ocr" / split
            image_dir.mkdir(exist_ok=True)
            image_path = f"{image_dir}/image_{i}.png"

            if not Path(image_path).exists():
                Image.fromarray(image).save(image_path)

            assert Path(image_path).exists(), f"Image {image_path} not found."
            image_paths.append(image_path)

        # load labels
        with open(label_data, "rb") as file:
            # read and unpack the first 8 bytes for nb_classes and nb_samples
            nb_classes, nb_samples = struct.unpack(">ii", file.read(8))
            assert nb_samples == len(image_paths), "Number of labels do not match number of images."

            labels = []
            for _ in range(nb_samples):
                (label,) = struct.unpack(">i", file.read(4))
                assert 0 <= label < nb_classes, f"Label {label} out of bounds."
                labels.append(label)

        if self.config.schema == "source":
            for idx, example in enumerate(zip(image_paths, labels)):
                yield idx, {
                    "image_path": example[0],
                    "label": example[1],
                }

        elif self.config.schema == _SEACROWD_SCHEMA:
            for idx, example in enumerate(zip(image_paths, labels)):
                yield idx, {
                    "id": str(idx),
                    "image_paths": [example[0]],
                    "texts": None,
                    "metadata": {
                        "context": None,
                        "labels": [example[1]],
                    },
                }
