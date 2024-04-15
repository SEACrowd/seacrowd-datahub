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
https://huggingface.co/docs/hub/datasets-adding

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

# TODO: create a module level variable with your dataset name (should match script name)
#  E.g. Hallmarks of Cancer: [dataset_name] --> hallmarks_of_cancer
_DATASETNAME = "[dataset_name]"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This dataset is designed for XXX NLP task.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = ""

# TODO: Add languages related to this dataset
_LANGUAGES = []  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

# TODO: Add the licence for the dataset here (see constant choices in https://github.com/SEACrowd/seacrowd-datahub/blob/master/seacrowd/utils/constants.py)
# Note that this doesn't have to be a common open source license.
# In the case of the dataset intentionally is built without license, please use `Licenses.UNLICENSE.value`
# In the case that it's not clear whether the dataset has a license or not, please use `Licenses.UNKNOWN.value`
# Some datasets may also have custom licenses. In this case, simply put f'{Licenses.OTHERS.value} | {FULL_LICENSE_TERM}' into `_LICENSE`
_LICENSE = "" # example: Licenses.MIT.value, Licenses.CC_BY_NC_SA_4_0.value, Licenses.UNLICENSE.value, Licenses.UNKNOWN.value

# TODO: Add a _LOCAL flag to indicate whether the data cannot be sourced from a public link
#  E.g. the dataset requires signing a specific term of use, the dataset is sent through email, etc.
_LOCAL = False

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and seacrowd config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: "url or list of urls or ... ",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks --> # TODO: add supported task by dataset. One dataset may support multiple tasks (see constant choices in https://github.com/SEACrowd/seacrowd-datahub/blob/master/seacrowd/utils/constants.py)
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
#  This version doesn't have to be consistent with semantic versioning. Anything that is
#  provided by the original dataset as a version goes.
_SOURCE_VERSION = ""

_SEACROWD_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using PascalCase instead of snake_case.
# optional: class name can append "Dataset" as suffix to provide better clarity (e.g. OSCAR 2201 --> Oscar2201Dataset/Oscar2201)
class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    # You will be able to load the "source" or "seacrowd" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_seacrowd = datasets.load_dataset('my_dataset', name='seacrowd')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_seacrowd = datasets.load_dataset('my_dataset', name='seacrowd', data_dir="/path/to/data/files")

    # TODO: For each dataset, implement Config for Source and SEACrowd;
    #  If dataset contains more than one subset (see seacrowd/sea_datasets/smsa.py) implement for EACH of them.
    #  Each of them should contain:
    #   - name: should be unique for each dataset config eg. smsa_(source|seacrowd)_[seacrowd_schema_name]
    #   - version: option = (SOURCE_VERSION|SEACROWD_VERSION)
    #   - description: one line description for the dataset
    #   - schema: options = (source|seacrowd_[seacrowd_schema_name])
    #   - subset_id: subset id is the canonical name for the dataset (eg. smsa)
    #  where [seacrowd_schema_name] can be checked in seacrowd/utils/constants.py
    #    under variable `TASK_TO_SCHEMA`, in accordance to values from `_SUPPORTED_TASKS`
    #    for all config(s) defined

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="[dataset_name]_source",
            version=SOURCE_VERSION,
            description="[dataset_name] source schema",
            schema="source",
            subset_id="[dataset_name]",
        ),
        SEACrowdConfig(
            name="[dataset_name]_seacrowd_[seacrowd_schema_name]",
            version=SEACROWD_VERSION,
            description="[dataset_name] SEACrowd schema",
            schema="seacrowd_[seacrowd_schema_name]",
            subset_id="[dataset_name]",
        ),
    ]

    DEFAULT_CONFIG_NAME = "[dataset_name]_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            # TODO: Create your source schema here
            raise NotImplementedError()

            # EX: Arbitrary NER type dataset
            # features = datasets.Features(
            #    {
            #        "doc_id": datasets.Value("string"),
            #        "text": datasets.Value("string"),
            #        "entities": [
            #            {
            #                "offsets": [datasets.Value("int64")],
            #                "text": datasets.Value("string"),
            #                "type": datasets.Value("string"),
            #                "entity_id": datasets.Value("string"),
            #            }
            #        ],
            #    }
            # )

        # Choose the appropriate seacrowd schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple seacrowd configs with a seacrowd_[seacrowd_schema_name] format.

        # For example seacrowd_kb, seacrowd_t2t
        elif self.config.schema == "seacrowd_[seacrowdschema_name]":
            # e.g. features = schemas.kb_features
            # TODO: Choose your seacrowd schema here
            raise NotImplementedError()

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

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/v1.1.1/_modules/datasets/utils/download_manager.html

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
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files

        if self.config.schema == "source":
            # TODO: yield (key, example) tuples in the original dataset schema
            for key, example in thing:
                yield key, example

        elif self.config.schema == "seacrowd_[seacrowd_schema_name]":
            # TODO: yield (key, example) tuples in the seacrowd schema
            for key, example in thing:
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
