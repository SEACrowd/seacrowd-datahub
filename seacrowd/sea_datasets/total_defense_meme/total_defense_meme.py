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
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import gdown

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@inproceedings{10.1145/3587819.3592545,
    author = {Prakash, Nirmalendu and Hee, Ming Shan and Lee, Roy Ka-Wei},
    title = {TotalDefMeme: A Multi-Attribute Meme dataset on Total Defence in Singapore},
    year = {2023},
    isbn = {9798400701481},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3587819.3592545},
    doi = {10.1145/3587819.3592545},
    booktitle = {Proceedings of the 14th Conference on ACM Multimedia Systems},
    pages = {369–375},
    numpages = {7},
    keywords = {multimodal, meme, dataset, topic clustering, stance classification},
    location = {Vancouver, BC, Canada},
    series = {MMSys '23}
}
"""

_DATASETNAME = "total_defense_meme"

_DESCRIPTION = """\
This is a large-scale multimodal and multi-attribute dataset containing memes
about Singapore's Total Defence policy from different social media platforms.
The type (Singaporean or generic), pillars (military, civil, economic, social,
psychological, digital, others), topics and stances (against, neutral,
supportive) of each meme are manually identified by annotators.
"""

_HOMEPAGE = "https://gitlab.com/bottle_shop/meme/TotalDefMemes"

_LANGUAGES = ["eng"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "image": "https://drive.google.com/file/d/1oJIh4QQS3Idff2g6bZORstS5uBROjUUz/view?usp=share_link",
    "annotations": "https://gitlab.com/bottle_shop/meme/TotalDefMemes/-/raw/main/report/annotation.json?ref_type=heads",
}

_SUPPORTED_TASKS = [Tasks.OPTICAL_CHARACTER_RECOGNITION, Tasks.IMAGE_CLASSIFICATION_MULTILABEL]
_SEACROWD_SCHEMA = {
    task.value: f"seacrowd_{TASK_TO_SCHEMA[task].lower()}" for task in _SUPPORTED_TASKS
}  # ocr: imtext, imc_multi: image_multi

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class TotalDefenseMemeDataset(datasets.GeneratorBasedBuilder):
    """Multimodal dataset containing memes about Singapore's Total Defence policy"""

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
            name=f"{_DATASETNAME}_{_SEACROWD_SCHEMA['OCR']}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=_SEACROWD_SCHEMA["OCR"],
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{_SEACROWD_SCHEMA['IMC_MULTI']}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=_SEACROWD_SCHEMA["IMC_MULTI"],
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        # define labelling
        meme_type = ["Non_Memes", "Non_SG_Memes", "SG_Memes"]
        pillar_type = [
            "Social",
            "Economic",
            "Psychological",
            "Military",
            "Civil",
            "Digital",
            "Others",
        ]
        stance_type = ["Against", "Neutral", "Supportive"]

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "image_path": datasets.Value("string"),
                    "meme_type": datasets.Sequence(datasets.ClassLabel(names=meme_type)),
                    "text": datasets.Value("string"),
                    "tags": datasets.Sequence(datasets.Value("string")),
                    "pillar_stances": datasets.Sequence(
                        {
                            "category": datasets.ClassLabel(names=pillar_type),
                            "stance": datasets.Sequence(datasets.ClassLabel(names=stance_type)),
                        }
                    ),
                }
            )

        elif self.config.schema == _SEACROWD_SCHEMA["OCR"]:  # all images
            features = schemas.image_text_features(label_names=meme_type)
            features["metadata"] = {
                "tags": datasets.Sequence(datasets.Value("string")),
                "pillar_stances": datasets.Sequence(
                    {
                        "category": datasets.ClassLabel(names=pillar_type),
                        "stance": datasets.Sequence(datasets.ClassLabel(names=stance_type)),
                    }
                ),
            }
        elif self.config.schema == _SEACROWD_SCHEMA["IMC_MULTI"]:  # sg meme images only
            features = schemas.image_multi_features(label_names=pillar_type)
            features["metadata"] = {
                "tags": datasets.Sequence(datasets.Value("string")),
                "stances": datasets.Sequence(datasets.Sequence(datasets.ClassLabel(names=stance_type))),
            }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # download image from gdrive
        output_dir = Path.cwd() / "data" / _DATASETNAME
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{_DATASETNAME}.zip"
        if not output_file.exists():
            gdown.download(_URLS["image"], str(output_file), fuzzy=True)
        else:
            print(f"File already downloaded: {str(output_file)}")
        # extract image data
        image_dir = Path(dl_manager.extract(output_file)) / "TD_Memes"

        # download annotations
        annotation_path = Path(dl_manager.download(_URLS["annotations"]))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_dir": image_dir,
                    "annotation_file": annotation_path,
                },
            ),
        ]

    def _generate_examples(self, image_dir: Path, annotation_file: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # load annotation
        with open(annotation_file, "r", encoding="utf-8") as file:
            annotation = json.load(file)

        # get unique image names
        image_names = sorted(
            list(
                set(annotation["Non_Memes"])
                | set(annotation["Non_SG_Memes"])
                | set(annotation["SG_Memes"])
            )
        )

        # annotation data is a list of dict, instead of dict of image names
        def get_value(image_name, list_of_dicts):
            for dictionary in list_of_dicts:
                if image_name in dictionary:
                    return dictionary[image_name]
            return None

        key = 0
        for image_name in image_names:
            # assert image exist in directory
            assert (image_dir / image_name).exists(), f"Image {image_name} not found"
            image_path = str(image_dir / image_name)

            # get categories, can be multiple
            categories = []
            if image_name in annotation["Non_Memes"]:
                categories.append("Non_Memes")
            if image_name in annotation["Non_SG_Memes"]:
                categories.append("Non_SG_Memes")
            if image_name in annotation["SG_Memes"]:
                categories.append("SG_Memes")

            # get attributes
            text = get_value(image_name, annotation["Text"])
            tags = get_value(image_name, annotation["Tags"])
            raw_pillar_stances = get_value(image_name, annotation["Pillar_Stances"])

            # process pillar stances
            pillar_stances = []
            if raw_pillar_stances:
                for pillar, stances in raw_pillar_stances:
                    category = pillar.split(" ")[0]
                    pillar_stances.append({"category": category, "stance": stances})

            # source schema
            if self.config.schema == "source":
                yield key, {
                    "image_path": image_path,
                    "meme_type": categories,
                    "text": text,
                    "tags": tags,
                    "pillar_stances": pillar_stances,
                }
                key += 1

            # ocr seacrowd schema
            elif self.config.schema == _SEACROWD_SCHEMA["OCR"]:
                yield key, {
                    "id": str(key),
                    "image_paths": [image_path],
                    "texts": text,
                    "metadata": {
                        "tags": tags,
                        "pillar_stances": pillar_stances,
                    },
                }
                key += 1

            # pillar/topic classification seacrowd schema
            elif self.config.schema == _SEACROWD_SCHEMA["IMC_MULTI"]:
                if pillar_stances:  # only those with pillar stances
                    yield key, {
                        "id": str(key),
                        "labels": [pillar["category"] for pillar in pillar_stances],
                        "image_path": image_path,
                        "metadata": {
                            "tags": tags,
                            "stances": [pillar["stance"] for pillar in pillar_stances],
                        },
                    }
                    key += 1
