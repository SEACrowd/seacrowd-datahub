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

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@article{Chen2012,
    title = {Creating a live, public short message service corpus: the NUS SMS corpus},
    ISSN = {1574-0218},
    url = {http://dx.doi.org/10.1007/s10579-012-9197-9},
    DOI = {10.1007/s10579-012-9197-9},
    journal = {Language Resources and Evaluation},
    publisher = {Springer Science and Business Media LLC},
    author = {Chen, Tao and Kan, Min-Yen},
    year = {2012},
    month = aug
}
"""

_DATASETNAME = "nus_sms_corpus"

_DESCRIPTION = """\
This is a corpus of SMS (Short Message Service) messages collected for research
at the Department of Computer Science at the National University of Singapore.
This dataset consists of 67,093 SMS messages taken from the corpus on Mar 9,
2015. The messages largely originate from Singaporeans and mostly from students
attending the University. These messages were collected from volunteers who were
made aware that their contributions were going to be made publicly available.
The data collectors opportunistically collected as much metadata about the
messages and their senders as possible, so as to enable different types of
analyses.
"""

_HOMEPAGE = "https://github.com/kite1988/nus-sms-corpus"

_LANGUAGES = ["eng", "cmn"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "eng": "https://github.com/kite1988/nus-sms-corpus/raw/master/smsCorpus_en_xml_2015.03.09_all.zip",
    "cmn": "https://github.com/kite1988/nus-sms-corpus/raw/master/smsCorpus_zh_xml_2015.03.09.zip",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # ssp

_SOURCE_VERSION = "1.2.0"  # inside the dataset

_SEACROWD_VERSION = "1.0.0"


class NusSmsCorpusDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in _LANGUAGES:
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

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{_LANGUAGES[0]}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            # all values from xml data are strings
            features = datasets.Features(
                {
                    "@id": datasets.Value("string"),
                    "text": {"$": datasets.Value("string")},
                    "source": {
                        "srcNumber": {"$": datasets.Value("string")},
                        "phoneModel": {
                            "@manufactuer": datasets.Value("string"),
                            "@smartphone": datasets.Value("string"),
                        },
                        "userProfile": {
                            "userID": {"$": datasets.Value("string")},
                            "age": {"$": datasets.Value("string")},
                            "gender": {"$": datasets.Value("string")},
                            "nativeSpeaker": {"$": datasets.Value("string")},
                            "country": {"$": datasets.Value("string")},
                            "city": {"$": datasets.Value("string")},
                            "experience": {"$": datasets.Value("string")},
                            "frequency": {"$": datasets.Value("string")},
                            "inputMethod": {"$": datasets.Value("string")},
                        },
                    },
                    "destination": {
                        "@country": datasets.Value("string"),
                        "destNumber": {"$": datasets.Value("string")},
                    },
                    "messageProfile": {
                        "@language": datasets.Value("string"),
                        "@time": datasets.Value("string"),
                        "@type": datasets.Value("string"),
                    },
                    "collectionMethod": {
                        "@collector": datasets.Value("string"),
                        "@method": datasets.Value("string"),
                        "@time": datasets.Value("string"),
                    },
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]]  # ssp_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        lang = self.config.subset_id
        data_path = Path(dl_manager.download_and_extract(_URLS[lang]))
        data_file = list(data_path.glob("*.xml"))[0]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": data_file,
                },
            ),
        ]

    def xml_element_to_dict(self, element: ET.Element) -> Dict:
        """Converts an xml element to a dictionary."""
        element_dict = {}

        # add text with key '$', attributes with '@' prefix
        element_dict["$"] = element.text
        for attrib, value in element.attrib.items():
            element_dict[f"@{attrib}"] = value

        # recursively
        for child in element:
            child_dict = self.xml_element_to_dict(child)
            element_dict[child.tag] = child_dict

        return element_dict

    def _generate_examples(self, data_file: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data_root = ET.parse(data_file).getroot()
        data_list = [self.xml_element_to_dict(child) for child in data_root]

        for idx, example in enumerate(data_list):
            if self.config.schema == "source":
                yield idx, example
            elif self.config.schema == _SEACROWD_SCHEMA:
                yield idx, {
                    "id": str(idx),  # example["@id"] is not unique
                    "text": example["text"]["$"],
                }
