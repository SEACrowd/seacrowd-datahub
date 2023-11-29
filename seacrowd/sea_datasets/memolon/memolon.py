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

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants Tasks, Licenses

_CITATION = """\
@inproceedings{buechel-etal-2020-learning-evaluating,
    title = "Learning and Evaluating Emotion Lexicons for 91 Languages",
    author = {Buechel, Sven  and
      R{\"u}cker, Susanna  and
      Hahn, Udo},
    editor = "Jurafsky, Dan  and
      Chai, Joyce  and
      Schluter, Natalie  and
      Tetreault, Joel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.112",
    doi = "10.18653/v1/2020.acl-main.112",
    pages = "1202--1217",
    abstract = "Emotion lexicons describe the affective meaning of words and thus constitute a centerpiece for advanced sentiment and emotion analysis. Yet, manually curated lexicons are only available for a handful of languages, leaving most languages of the world without such a precious resource for downstream applications. Even worse, their coverage is often limited both in terms of the lexical units they contain and the emotional variables they feature. In order to break this bottleneck, we here introduce a methodology for creating almost arbitrarily large emotion lexicons for any target language. Our approach requires nothing but a source language emotion lexicon, a bilingual word translation model, and a target language embedding model. Fulfilling these requirements for 91 languages, we are able to generate representationally rich high-coverage lexicons comprising eight emotional variables with more than 100k lexical entries each. We evaluated the automatically generated lexicons against human judgment from 26 datasets, spanning 12 typologically diverse languages, and found that our approach produces results in line with state-of-the-art monolingual approaches to lexicon creation and even surpasses human reliability for some languages and variables. Code and data are available at \url{https://github.com/JULIELab/MEmoLon} archived under DOI 10.5281/zenodo.3779901.",
}
"""


_DATASETNAME = "memolon"

_DESCRIPTION = """\
MEmoLon is an emotion lexicons for 91 languages, each one covers eight emotional variables and comprises over 100k word entries. There are several versions of the lexicons, the difference being the choice of the expansion model.
"""

_HOMEPAGE = "https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1"

_LICENSE = Licenses.MIT.value

_URLS = {
    _DATASETNAME: "https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1",
}

_SUPPORTED_TASKS = [Tasks.EMOTION_CLASSIFICATION]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

_LANGUAGE_MAP = {
    "eng": "English", 
    "ceb": "Cebuano",
    "tl": "Tagalog", 
    "id": "Indonesian", 
    "su": "Sundanese", 
    "jv": "Javanese",
    "ms": "Malay", 
    "vi": "Vietnamese", 
    "th": "Thai", 
    "mya": "Burmese",
    "zh": "Chinese" 
}



def seacrowd_config_constructor(lang, schema, version):
    if lang not in _LANGUAGE_MAP:
        raise ValueError(f"Invalid lang {lang}")

    if schema != "source" and schema != "seacrowd_text_multi":
        raise ValueError(f"Invalid schema: {schema}")

    return SEACrowdConfig(
        name="memolon_{lang}_{schema}".format(lang=lang, schema=schema),
        version=datasets.Version(version),
        description="MEmoLon {schema} schema for {lang} language".format(lang=_LANGUAGE_MAP[lang], schema=schema),
        schema=schema,
        subset_id="memolon",
    )

class Memolon(datasets.GeneratorBasedBuilder):
    """MEmoLon is an emotion lexicons for 91 languages, each one covers eight emotional variables and comprises over 100k word entries."""

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
    BUILDER_CONFIGS = [seacrowd_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGUAGE_MAP] + [seacrowd_config_constructor(lang, "seacrowd_text_multi", _SEACROWD_VERSION) for lang in _LANGUAGE_MAP]
    # BUILDER_CONFIGS = [
    #     SEACrowdConfig(
    #         name=f"{_DATASETNAME}_source",
    #         version=datasets.Version(_SOURCE_VERSION),
    #         description="MEmoLon source schema",
    #         schema="source",
    #         subset_id=f"{_DATASETNAME}",
    #     ),
    #     SEACrowdConfig(
    #         name=f"{_DATASETNAME}_seacrowd_text_multi",
    #         version=SEACROWD_VERSION,
    #         description="MEmoLon SEACrowd schema",
    #         schema="seacrowd_text_multi",
    #         subset_id=f"{_DATASETNAME}",
    #     ),
    # ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "word": datasets.Value("string"), 
                    "valence": datasets.Value("float16"), 
                    "arousal": datasets.Value("float16"),
                    "dominance": datasets.Value("float16"), 
                    "joy": datasets.Value("float16"),
                    "anger": datasets.Value("float16"), 
                    "sadness": datasets.Value("float16"),
                    "fear": datasets.Value("float16"), 
                    "disgust": datasets.Value("float16")
                }
            )
        elif self.config.schema == "seacrowd_[seacrowdschema_name]":
            features = schemas.text_multi_features(["valence", "arousal", "dominance", "joy", "anger", "sadness", "fear", "disgust"])
            
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
        base_path = Path(dl_manager.download_and_extract(urls))
        lang = self.config.name.split("_")[1]
        train_data_path = base_path / f"{lang}.tsv"



        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": train_data_path,
                    "split": "train",
                },
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files
        rows = []
        with open("filename.tsv") as file:
            for line in file:
                l=line.split('\t')
                rows.append(l)

        # DAN!!!! REFER TO THIS https://www.geeksforgeeks.org/simple-ways-to-read-tsv-files-in-python/
        if self.config.schema == "source":
            # TODO: yield (key, example) tuples in the original dataset schema
            for key, row in enumerate(rows[1:]):
                example = {
                    "word": row[0],
                    "valence": row[1],
                    "arousal": row[2],
                    "dominance": row[3],
                    "joy": row[4],
                    "anger": row[5],
                    "sadness": row[6],
                    "fear": row[7],							
                    "disgust": row[8]
                }
                yield key, example

        elif self.config.schema == "seacrowd_text_multi":
            # TODO: yield (key, example) tuples in the seacrowd schema
            for key, example in enumerate(rows[1:]):
                example = {
                    "id": key,
                    "text": row[0],
                    "labels": [row[i] for i in range(1,9)]
                }
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
