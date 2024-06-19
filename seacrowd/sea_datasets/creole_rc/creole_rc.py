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

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{lent2023creoleval,
    title={CreoleVal: Multilingual Multitask Benchmarks for Creoles},
    author={Heather Lent and Kushal Tatariya and Raj Dabre and Yiyi Chen and Marcell Fekete and Esther Ploeger and Li Zhou and Hans Erik Heje and Diptesh Kanojia and Paul Belony and Marcel Bollmann and \
    Loïc Grobol and Miryam de Lhoneux and Daniel Hershcovich and Michel DeGraff and Anders Søgaard and Johannes Bjerva},
    year={2023},
    eprint={2310.19567},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DATASETNAME = "creole_rc"
_DESCRIPTION = """\
CreoleRC is a subset created by the CreoleVal paper. Relation classification (RC) aims to identify semantic associations between entities within a text, essential for applications like knowledge base \
completion and question answering. The dataset is sourced from Wikipedia and manually annotated. CreoleRC contains 5 creoles, but SEACrowd is interested specifically in the Chavacano subset.
"""
_HOMEPAGE = "https://github.com/hclent/CreoleVal/tree/main/nlu/relation_classification"
_LANGUAGES = ["cbk"]
_LICENSE = Licenses.CC_BY_SA_4_0.value
_LOCAL = False
_URLS = {
    "csv": "https://raw.githubusercontent.com/hclent/CreoleVal/main/nlu/relation_classification/data/relation_extraction/cbk-zam.csv",
    "json": "https://raw.githubusercontent.com/hclent/CreoleVal/main/nlu/relation_classification/data/relation_extraction/cbk-zam.json",
}
_SUPPORTED_TASKS = [Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class CreoleRC(datasets.GeneratorBasedBuilder):
    """Creole relation classification dataset, Chavacano subset, from https://github.com/hclent/CreoleVal/tree/main/nlu/relation_classification."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    # You will be able to load the "source" or "seacrowd" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_seacrowd = datasets.load_dataset('my_dataset', name='seacrowd')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_seacrowd = datasets.load_dataset('my_dataset', name='seacrowd', data_dir="/path/to/data/files")

    SEACROWD_SCHEMA_NAME = "kb"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "ent1": datasets.Value("string"),
                    "ent2": datasets.Value("string"),
                    "ent1_qcode": datasets.Value("string"),
                    "ent2_qcode": datasets.Value("string"),
                    "property": datasets.Value("string"),
                    "property_desc": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "edgeset_left": datasets.Sequence(datasets.Value("int32")),
                    "edgeset_right": datasets.Sequence(datasets.Value("int32")),
                    "edgeset_triple": datasets.Sequence(datasets.Value("string")),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        data_paths = {
            "csv": Path(dl_manager.download_and_extract(_URLS["csv"])),
            "json": Path(dl_manager.download_and_extract(_URLS["json"])),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "csv_filepath": data_paths["csv"],
                    "json_filepath": data_paths["json"],
                    "split": "train",
                },
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    def _generate_examples(self, csv_filepath: Path, json_filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # read csv file
        with open(csv_filepath, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)
            csv_data = [row for row in csv_reader]
        csv_data = csv_data[1:]  # remove header

        # read json file
        with open(json_filepath, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)

        # properties descriptions from https://github.com/hclent/CreoleVal/tree/main/nlu/relation_classification
        # for properties present in Chavacano subset
        properties_desc = {"P17": "country", "P30": "continent", "P106": "occupation", "P131": "located in the administrative territorial entity", "P361": "part of", "P1376": " capital of country"}

        num_sample = len(csv_data)

        for i in range(num_sample):
            if self.config.schema == "source":
                example = {
                    "id": str(i),
                    "sentence": csv_data[i][0],
                    "ent1": csv_data[i][1],
                    "ent2": csv_data[i][2],
                    "ent1_qcode": csv_data[i][3],
                    "ent2_qcode": csv_data[i][4],
                    "property": csv_data[i][5],
                    "property_desc": properties_desc[csv_data[i][5]],
                    "tokens": json_data[i]["tokens"],
                    "edgeset_left": json_data[i]["edgeSet"]["left"],
                    "edgeset_right": json_data[i]["edgeSet"]["right"],
                    "edgeset_triple": json_data[i]["edgeSet"]["triple"],
                }
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                offset_entity1 = csv_data[i][0].find(csv_data[i][1])
                offset_entity2 = csv_data[i][0].find(csv_data[i][2])

                if (offset_entity1 == -1) or (offset_entity2 == -1):
                    continue

                example = {
                    "id": str(i),
                    "passages": [{"id": f"passage-{i}", "type": "text", "text": [csv_data[i][0]], "offsets": [[0, len(csv_data[i][0])]]}],
                    "entities": [
                        {"id": f"{i}-entity-{csv_data[i][3]}", "type": "text", "text": [csv_data[i][1]], "normalized": [{"db_name": csv_data[i][1], "db_id": csv_data[i][3]}], "offsets": [[offset_entity1, offset_entity1 + len(csv_data[i][1])]]},
                        {"id": f"{i}-entity-{csv_data[i][4]}", "type": "text", "text": [csv_data[i][2]], "normalized": [{"db_name": csv_data[i][2], "db_id": csv_data[i][4]}], "offsets": [[offset_entity2, offset_entity2 + len(csv_data[i][2])]]},
                    ],
                    "events": [],
                    "coreferences": [],
                    "relations": [
                        {
                            "id": f"{i}-relation-{csv_data[i][5]}",
                            "type": properties_desc[csv_data[i][5]],
                            "arg1_id": f"{i}-entity-{csv_data[i][3]}",
                            "arg2_id": f"{i}-entity-{csv_data[i][4]}",
                            "normalized": [{"db_name": properties_desc[csv_data[i][5]], "db_id": csv_data[i][5]}],
                        }
                    ],
                }

            yield i, example
