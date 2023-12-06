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

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@article{kautsar2023indotod,
    author={Kautsar, Muhammad Dehan Al and Nurdini, Rahmah Khoirussyifa' and Cahyawijaya, Samuel and Winata, Genta Indra and Purwarianti, Ayu},
    title={IndoToD: A Multi-Domain Indonesian Benchmark For End-to-End Task-Oriented Dialogue Systems}, 
    journal={arXiv preprint arXiv:2311.00958},
    year={2023},
}
"""

_LANGUAGES = ["ind"]
_LOCAL = False

_DATASETNAME = "indocamrest"

_DESCRIPTION = """\
IndoCamRest is a synthetic task-oriented dialogue system dataset that translated from Cambridge Restaurant 676 (CamRest) dataset (Wen et al., 2016) into the new Indonesian parallel dataset using the translation pipeline method including the delexicalization, translation, and delexicalization.
The dataset consists of 676 dialogues in the restaurant reservation domain, with a user and an agent talking to each other to search the restaurant near the user.
It also consists of slots and dialogue acts from the user and the agent.
"""

_HOMEPAGE = "https://github.com/dehanalkautsar/IndoToD/tree/main/IndoCamRest"

_LICENSE = "Licenses.CC_BY_SA_4_0" 

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/dehanalkautsar/IndoToD/main/IndoCamRest/IndoCamRest676.json",
}

_SUPPORTED_TASKS = [Tasks.DIALOGUE_SYSTEM]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


# Name the dataset class to match the script name using CamelCase instead of snake_case
class IndoCamRest(datasets.GeneratorBasedBuilder):
    """IndoToD: A Multi-Domain Indonesian Benchmark For End-to-End Task-Oriented Dialogue Systems"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    # You will be able to load the "source" or "seacrowd" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_seacrowd = datasets.load_dataset('my_dataset', name='seacrowd')

    # For each dataset, implement Config for Source and SEACrowd;
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
            name="indocamrest_source",
            version=SOURCE_VERSION,
            description="IndoToD: IndoCamRest source schema",
            schema="source",
            subset_id="indocamrest",
        ),
        SEACrowdConfig(
            name="indocamrest_seacrowd_ds",
            version=SEACROWD_VERSION,
            description="IndoToD: IndoCamRest SEACrowd schema",
            schema="seacrowd_ds",
            subset_id="indocamrest",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indocamrest_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "dialogue_id": datasets.Value("int32"),
                    "finished": datasets.Value("string"),
                    "goal": {
                        "constraints": [[datasets.Value("string")]],
                        "request-slots": [datasets.Value("string")],
                        "text": datasets.Value("string")
                    },
                    "dial": [
                        {
                            "turn": datasets.Value("int32"),
                            "usr": {
                                "transcript": datasets.Value("string"),
                                "delex_transcript": datasets.Value("string"),
                                "slu": [
                                    {
                                        "act": datasets.Value("string"),
                                        "slots": [[
                                            datasets.Value("string")
                                        ]]
                                    }
                                ],
                            },
                            "sys": {
                                "sent": datasets.Value("string"),
                                "delex_sent": datasets.Value("string"),
                                "DA": [datasets.Value("string")]
                            }
                        }
                    ]
                }
            )
        elif self.config.schema == 'seacrowd_ds':
            pass
        else:
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
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access the "source" or "seacrowd" config choice, that will be in self.config.name

        # PUBLIC DATASETS: Assign your data-dir based on the dl_manager.

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and will give back the same structure with the url replaced with the path to local files.

        # TODO: KEEP if your dataset is PUBLIC; remove if not
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    # "filepath": os.path.join(data_dir, "train.jsonl"),
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]


    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, "r+") as fw:
            data = json.loads(fw.read())

        if self.config.schema == "source":
            # yield (key, example) tuples in the original dataset schema
            for idx, example in enumerate(data):
                example["index"] = str(idx)
                yield str(idx), example

        elif self.config.schema == "seacrowd_ds":
            # yield (key, example) tuples in the seacrowd schema
            # for key, example in thing:
            #     yield key, example
            for idx, example in enumerate(data):
                example["index"] = str(idx)
                yield str(idx), example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
