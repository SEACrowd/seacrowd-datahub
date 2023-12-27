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
This template serves as a starting point for contributing a dataset to the SEACrowd Datahub repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_LOCAL = False
_LANGUAGES = [
    "eng",
    "mya",
    "ceb",
    "gor",
    "hil",
    "ilo",
    "ind",
    "jav",
    "khm",
    "lao",
    "zlm",
    "nia",
    "tgl",
    "tha",
    "vie"
]
_DATASETNAME = "mozilla_pontoon"
_DESCRIPTION = """
This dataset contains translations from Mozilla's Pontoon localization platform
for more than 200 languages. Source sentences are in English.
"""

_HOMEPAGE = "https://huggingface.co/datasets/ayymen/Pontoon-Translations"
_LICENSE = Licenses.BSD_3_CLAUSE
_URLS = {
    _DATASETNAME: "url or list of urls or ... ",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class MozillaPontoonDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "t2t"

    # Config to load individual datasets per language
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema for {lang} language",
            schema="source",
            subset_id=lang,
        )
        for lang in _LANGUAGES
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{lang}_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema for {lang} language",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=lang,
        )
        for lang in _LANGUAGES
    ]

    # Config to load all datasets
    BUILDER_CONFIGS.extend(
        [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} source schema for all languages",
                schema="source",
                subset_id=_DATASETNAME,
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} SEACrowd schema for all languages",
                schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
                subset_id=_DATASETNAME,
            )
        ]
    )


    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "source_sentence": datasets.Value("string"),
                    "target_sentence": datasets.Value("string"),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access the "source" or "seacrowd" config choice, that will be in self.config.name

        # LOCAL DATASETS: You do not need the dl_manager; you can ignore this argument. Make sure `gen_kwargs` in the return gets passed the right filepath

        # PUBLIC DATASETS: Assign your data-dir based on the dl_manager.

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and will give back the same structure with the url replaced with the path to local files.

        # TODO: KEEP if your dataset is PUBLIC; remove if not
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        # TODO: KEEP if your dataset is LOCAL; remove if NOT
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        for index, row in enumerate(data):
            if self.config.schema == "source":
                example = row

            elif self.config.schema == f"seacrowd_{SEACROWD_SCHEMA_NAME}":
                example = {
                    "id": str(index),
                    "text_1": row["source_sentence"],
                    "text_2": row["target_sentence"],
                    "text_1_name": "eng",
                    "text_2_name": lang,
                }
            


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)