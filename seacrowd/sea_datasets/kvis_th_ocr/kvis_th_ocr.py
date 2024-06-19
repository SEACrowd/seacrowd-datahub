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

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@INPROCEEDINGS{8584876,
  author={Joseph, Ferdin Joe John and Anantaprayoon, Panatchakorn},
  booktitle={2018 International Conference on Information Technology (InCIT)},
  title={Offline Handwritten Thai Character Recognition Using Single Tier Classifier and Local Features},
  year={2018},
  volume={},
  number={},
  pages={1-4},
  abstract={Handwritten character recognition is a conversion process of handwriting into machine-encoded text. Currently,
  several techniques and methods are proposed to enhance accuracy of handwritten character recognition for many languages
  spoken across the globe. In this project, a local feature-based approach is proposed to enhance the accuracy of handwritten
  offline character recognition for Thai alphabets. In the experiment, through MATLAB, 100 images for each class of Thai
  alphabets are collected and k-fold cross validation is applied to manage datasets to train and test. A gradient invariant
  feature set consisting of LBP and shape features is extracted. The classification is operated by using query matching based
  on Euclidean distance. The accuracy would be the percentage of correct classification for each class. For the result, the
  highest accuracy is 68.96% which has 144-bit shape features and uniform pattern LBP for the features.},
  keywords={Character recognition;Feature extraction;Shape;Genetic algorithms;Matlab;Gray-scale;Optical character recognition
  software;Offline Character Recognition;Local Binary Pattern;Thai Handwriting},
  doi={10.23919/INCIT.2018.8584876},
  ISSN={},
  month={Oct},
  url={https://ieeexplore.ieee.org/document/8584876}}

"""

_DATASETNAME = "kvis_th_ocr"

_DESCRIPTION = """\
The KVIS Thai OCR Dataset contains scanned handwritten version of all 44 Thai characters obtained from 27 individuals. It
consisted of 1079 images from 44 classes (letters). This dataset consists of all Thai consonants with different writing
styles of various people from ages between 16 and 75. Vowels and intonation are not taken into consideration for the dataset
collected.
"""

_HOMEPAGE = "https://data.mendeley.com/datasets/8nr3pbdk5c/1"

_LANGUAGES = ["tha"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/8nr3pbdk5c-1.zip"

_SUPPORTED_TASKS = [Tasks.OPTICAL_CHARACTER_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class KVISThaiOCRDataset(datasets.GeneratorBasedBuilder):
    """
    KVIS Thai OCR is a dataset for optical character recognition for Thai characters from https://data.mendeley.com/datasets/8nr3pbdk5c/1.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    labels = ["ก", "ข", "ฃ", "ค", "ฅ", "ฆ", "ง", "จ", "ฉ", "ช", "ซ", "ฌ", "ญ", "ฎ", "ฏ", "ฐ", "ฑ", "ฒ", "ณ", "ด", "ต", "ถ", "ท", "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ", "ฟ", "ภ", "ม", "ย", "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"]

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
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_imtext",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "file_path": datasets.Value("string"),
                    "character": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_imtext":
            features = schemas.image_text_features(label_names=self.labels)
        else:
            raise ValueError(f"Invalid schema: {self.config.schema}")

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

        dir = dl_manager.download_and_extract(_URLS)
        path = dl_manager.extract(os.path.join(dir, "KVIS TOCR Dataset.zip"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, path: Path, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """

        idx = 0
        path = list(os.walk(path))
        for directory in path[1:]:
            label = directory[0][-3:-2]
            for file in directory[2]:
                file_extension = str(file[-3:])
                if file_extension == "jpg":
                    file_id = str(file[:-4])
                    file_path = os.path.join(directory[0], file)
                    if self.config.schema == "source":
                        data = {
                            "id": file_id,
                            "file_path": file_path,
                            "character": label,
                        }
                        yield idx, data
                        idx += 1
                    elif self.config.schema == "seacrowd_imtext":
                        data = {
                            "id": file_id,
                            "image_paths": [file_path],
                            "texts": "",
                            "metadata": {
                                "context": "",
                                "labels": [label],
                            },
                        }
                        yield idx, data
                        idx += 1
                    else:
                        raise ValueError(f"Invalid schema: {self.config.schema}")
