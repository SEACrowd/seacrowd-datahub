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
import pandas as pd
import numpy as np

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses, TASK_TO_SCHEMA


_CITATION = """\
{@inproceedings{Maxwell-Smith_Foley_2023_Automated,
  title     = {{Automated speech recognition of Indonesian-English language lessons on YouTube using transfer learning}},
  author    = {Maxwell-Smith, Zara and Foley, Ben},
  booktitle = {Proceedings of the {Second Workshop on NLP Applications to Field Linguistics (EACL)}},
  pages     = {},
  year      = {forthcoming}
}
"""

_DATASETNAME = "oil"

_DESCRIPTION = """\
The Online Indonesian Learning (OIL) dataset or corpus currently contains lessons from three Indonesian teachers who have posted content on YouTube.
"""

_HOMEPAGE = "https://huggingface.co/datasets/ZMaxwell-Smith/OIL"

_LANGUAGES = ['eng', 'ind']  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_NC_ND_4_0.value

_LOCAL = False

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and seacrowd config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: {
        "train" : "https://huggingface.co/api/datasets/ZMaxwell-Smith/OIL/parquet/default/train/0.parquet"
    },
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{TASK_TO_SCHEMA[task]}" for task in _SUPPORTED_TASKS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

class OIL(datasets.GeneratorBasedBuilder):
    """The Online Indonesian Learning (OIL) dataset or corpus currently contains lessons from three Indonesian teachers who have posted content on YouTube."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    # You will be able to load the "source" or "seacrowd" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_seacrowd = datasets.load_dataset('my_dataset', name='seacrowd')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_seacrowd = datasets.load_dataset('my_dataset', name='seacrowd', data_dir="/path/to/data/files")

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    seacrowd_schema_config: list[SEACrowdConfig] = []

    for seacrowd_schema in _SUPPORTED_SCHEMA_STRINGS:

        seacrowd_schema_config.append(
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_{seacrowd_schema}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} SEACrowd {seacrowd_schema} schema",
                schema=f"seacrowd_{seacrowd_schema}",
                subset_id=f"{_DATASETNAME}",
            )
        )

    BUILDER_CONFIGS.extend(seacrowd_schema_config)

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
               {
                   "label": datasets.Value("string"),
                   "audio": [
                       {
                           "bytes": datasets.Value("bytes"),
                           "path": datasets.Value("string"),
                       }
                   ],
               }
            )

        elif self.config.schema == f"seacrowd_{TASK_TO_SCHEMA[Tasks.SPEECH_RECOGNITION]}":
            features = schemas.speech_text_features

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
            
        urls = _URLS[_DATASETNAME]
        train_path = dl_manager.download_and_extract(urls['train'])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files

        is_schema_found = False

        if self.config.schema == "source":
            # TODO: yield (key, example) tuples in the original dataset schema

            is_schema_found = True

            df = pd.read_parquet(filepath)

            for index, row in df.iterrows():
                yield index, row

        else:
            for seacrowd_schema in _SUPPORTED_SCHEMA_STRINGS:
                if self.config.schema == seacrowd_schema:
                    is_schema_found = True

                    # TODO: yield (key, example) tuples in the seacrowd schema
                    df = pd.read_parquet(filepath)

                    base_folder = os.path.dirname(filepath)
                    base_folder = os.path.join(base_folder, split)

                    if (not os.path.exists(base_folder)):
                        os.makedirs(base_folder)
                    
                    audio_paths = []

                    for _, row in df.iterrows():
                        audio_dict = row["audio"]
                        file_name = audio_dict["path"]

                        path = os.path.join(base_folder, file_name)

                        audio_dict["path"] = path

                        with open(path, "wb") as f:
                            f.write(audio_dict["bytes"])
                        
                        audio_paths.append(path)

                    df.rename(columns={"label": "id"}, inplace=True)

                    df["path"] = audio_paths
                    df = df.assign(text="").astype({'text': 'str'})
                    df = df.assign(speaker_id="").astype({'speaker_id': 'str'})
                    df = df.assign(metadata=[{'speaker_age': np.nan, 'speaker_gender': ""}] * len(df)).astype({'metadata': 'object'})

                    for index, row in df.iterrows():
                        yield index, row

        if not is_schema_found:        
            raise ValueError(f"Invalid config: {self.config.name}")

# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__, name="source")
