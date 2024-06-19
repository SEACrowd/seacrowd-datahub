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
from scipy.io import loadmat

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@article{Pino2021,
    title = {Optical character recognition system for Baybayin scripts using support vector machine},
    volume = {7},
    ISSN = {2376-5992},
    url = {http://dx.doi.org/10.7717/peerj-cs.360},
    DOI = {10.7717/peerj-cs.360},
    journal = {PeerJ Computer Science},
    publisher = {PeerJ},
    author = {Pino,  Rodney and Mendoza,  Renier and Sambayan,  Rachelle},
    year = {2021},
    month = feb,
    pages = {e360}
}
"""

_DATASETNAME = "baybayin"

_DESCRIPTION = """\
The Baybayin dataset contains binary images of Baybayin characters, Latin
characters, and 4 character symbols of Baybayin diacritics in MATLAB format. It
consisted of 17000 images for Baybayin (1000 per character), 18200 images for
Latin (700 per character), and 2000 images for Baybayin diacritics (500 per
symbol). Each character image is strictly center-fitted with a size 56x56
pixels. This dataset was initially used to discriminate Latin script from
Baybayin script in character recognition.

This is local dataset, please download the dataset from the `_HOMEPAGE` URL.
"""

_HOMEPAGE = "https://www.kaggle.com/datasets/rodneypino/baybayin-and-latin-binary-images-in-mat-format"

_LANGUAGES = ["tgl"]
_SUBSETS = ["baybayin", "latin", "diacritic"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = True  # kaggle dataset need to register to download

_URLS = {}

_SUPPORTED_TASKS = [Tasks.OPTICAL_CHARACTER_RECOGNITION]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # imtext

_SOURCE_VERSION = "4.0.0"

_SEACROWD_VERSION = "2024.06.20"


class BaybayinDataset(datasets.GeneratorBasedBuilder):
    """Binary images of Baybayin and Latin characters, and 4 character symbols of Baybayin diacritics"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in _SUBSETS:
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

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{_SUBSETS[0]}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "image": datasets.Array2D(shape=(56, 56), dtype="uint8"),
                    "character": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]]  # image_text_features()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the `data_dir` kwarg (where the .pdf is located) to load_dataset.")
        else:
            data_dir = Path(self.config.data_dir)

        subset_path = {
            "baybayin": "Baybayin/Baybayin.mat",
            "latin": "Latin/Latin.mat",
            "diacritic": "Baybayin Diacritics/Baybayin_Diacritics.mat",
        }

        mat_file = data_dir / subset_path[self.config.subset_id]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "mat_file": mat_file,
                },
            )
        ]

    def _generate_examples(self, mat_file: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        try:
            from PIL import Image
        except ImportError as err:
            raise ImportError("You need to install PIL (`pip install pillow`) to store the image from MATLAB structs to .png files.") from err

        raw_data = loadmat(str(mat_file))
        contained_data = raw_data[str(mat_file.stem)][0, 0]

        characters = list(contained_data.dtype.fields.keys())
        data = {char: contained_data[char] for char in characters}

        if self.config.schema == "source":
            key = 0
            for char, char_data in data.items():
                for i in range(char_data.shape[0]):
                    image = char_data[i].reshape((56, 56))
                    yield key, {
                        "image": image,
                        "character": char,
                    }
                    key += 1

        elif self.config.schema == _SEACROWD_SCHEMA:
            key = 0
            for char, char_data in data.items():
                # prepare path for saving images
                image_dir = mat_file.parent / char
                image_dir.mkdir(exist_ok=True)

                image_paths = []
                for i in range(char_data.shape[0]):
                    image = (char_data[i].reshape((56, 56)) * 255).astype("uint8")
                    image_path = str(image_dir / f"{char}_{i}.png")

                    # save image
                    Image.fromarray(image).save(image_path)
                    image_paths.append(image_path)

                yield key, {"id": str(key), "image_paths": image_paths, "texts": char, "metadata": None}
                key += 1
