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

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@article{SURINTA2015405,
    title = "Recognition of handwritten characters using local gradient feature descriptors",
    journal = "Engineering Applications of Artificial Intelligence",
    volume = "45",
    number = "Supplement C",
    pages = "405 - 414",
    year = "2015",
    issn = "0952-1976",
    doi = "https://doi.org/10.1016/j.engappai.2015.07.017",
    url = "http://www.sciencedirect.com/science/article/pii/S0952197615001724",
    author = "Olarik Surinta and Mahir F. Karaaba and Lambert R.B. Schomaker and Marco A. Wiering",
    keywords = "Handwritten character recognition, Feature extraction, Local gradient feature descriptor,
    Support vector machine, k-nearest neighbors"
}
"""

_DATASETNAME = "alice_thi"

_DESCRIPTION = """\
ALICE-THI is a Thai handwritten script dataset that contains 24045 character
images, which is split into Thai handwritten character dataset (THI-C68) for
14490 images and Thai handwritten digit dataset (THI-D10) for 9555 images. The
data was collected from 150 native writers aged from 20 to 23 years old. The
participants were allowed to write only the isolated Thai script on the form and
at least 100 samples per character. The character images obtained from this
dataset generally have no background noise.
"""

_HOMEPAGE = "https://www.ai.rug.nl/~mrolarik/ALICE-THI/"

_LANGUAGES = ["tha"]
_SUBSETS = {
    "THI-D10": {
        "data_dir": "Thai_digit_sqr",
        "label_dict": {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
        },
    },
    "THI-C68": {
        "data_dir": "Thai_char_sqr",
        "label_dict": {
            0: "ko kai",
            1: "kho khai",
            2: "kho khuat",
            3: "kho khwai",
            4: "kho khon",
            5: "kho rakhang",
            6: "ngo ngu",
            7: "cho chan",
            8: "cho ching",
            9: "cho chang",
            10: "so so",
            11: "cho choe",
            12: "yo ying",
            13: "do chada",
            14: "to patak",
            15: "tho than",
            16: "tho nangmontho",
            17: "tho phuthao",
            18: "no nen",
            19: "do dek",
            20: "to tao",
            21: "tho thung",
            22: "tho thahan",
            23: "tho thong",
            24: "no nu",
            25: "bo baimai",
            26: "po pla",
            27: "pho phung",
            28: "fo fa",
            29: "pho phan",
            30: "fo fan",
            31: "pho samphao",
            32: "mo ma",
            33: "yo yak",
            34: "ro rua",
            35: "ru",
            36: "lo ling",
            37: "lu",
            38: "wo waen",
            39: "so rusi",
            40: "so sala",
            41: "so sua",
            42: "ho hip",
            43: "lo chula",
            44: "o ang",
            45: "ho nokhuk",
            46: "paiyannoi",
            47: "sara a",
            48: "mai han",
            49: "sara aa",
            50: "sara i",
            51: "sara ii",
            52: "sara ue",
            53: "sara uee",
            54: "sara u",
            55: "sara uu",
            56: "sara e",
            57: "sara o",
            58: "sara ai maimuan",
            59: "sara ai maimalai",
            60: "maiyamok",
            61: "maitaikhu",
            62: "mai ek",
            63: "mai tho",
            64: "mai tri",
            65: "mai chattawa",
            66: "thanthakhat",
            67: "nikhahit",
        },
    },
}

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://www.ai.rug.nl/~mrolarik/ALICE-THI/ALICE-THI-Dataset.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.OPTICAL_CHARACTER_RECOGNITION]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # imtext

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class AliceTHIDataset(datasets.GeneratorBasedBuilder):
    """Thai handwritten script dataset for character and digit recognition."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in list(_SUBSETS.keys()):
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

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_THI-C68_source"

    def _info(self) -> datasets.DatasetInfo:
        label_names = [val for _, val in sorted(_SUBSETS[self.config.subset_id]["label_dict"].items())]
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "label": datasets.ClassLabel(names=label_names),
                    "text": datasets.Value("string"),
                    "image_path": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = schemas.image_text_features(label_names=label_names)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_name = "ALICE-THI Dataset"
        data_path = Path(dl_manager.download_and_extract(_URLS[_DATASETNAME]))
        data_path = Path(dl_manager.extract(data_path / data_name / f"{data_name}.tar.gz"))
        data_path = data_path / _SUBSETS[self.config.subset_id]["data_dir"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_path": data_path,
                },
            ),
        ]

    def _generate_examples(self, data_path: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # iterate over files and directories
        for subfolder in data_path.iterdir():
            if subfolder.is_dir():

                # source schema yield one image per label
                if self.config.schema == "source":
                    _get_label = True  # efficiency placeholder
                    for image_file in subfolder.glob("*.png"):
                        if _get_label:  # get label from filename
                            label = int(image_file.name.split("-")[0].lower())
                            _get_label = False

                        yield image_file.stem, {
                            "label": label,
                            "text": _SUBSETS[self.config.subset_id]["label_dict"][label],
                            "image_path": str(image_file),
                        }

                # seacrowd schema yield multiple images per label
                elif self.config.schema == _SEACROWD_SCHEMA:
                    image_files = list(subfolder.glob("*.png"))
                    label = int(image_files[0].name.split("-")[0].lower())

                    yield subfolder.name, {
                        "id": subfolder.name,
                        "image_paths": [str(file) for file in image_files],
                        "texts": _SUBSETS[self.config.subset_id]["label_dict"][label],
                        "metadata": {
                            "context": "",
                            "labels": [label] * len(image_files),
                        },
                    }
