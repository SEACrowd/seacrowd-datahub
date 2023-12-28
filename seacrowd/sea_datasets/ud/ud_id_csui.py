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

from seacrowd.utils.common_parser import load_ud_data, load_ud_data_as_seacrowd_kb
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks
from seacrowd.sea_datasets.ud.ud import UDDataset, _DATASETNAME

_CITATION = ""

_LANGUAGES = ["ind", "vie"]
_LOCAL = False

_SUBSET = "id_csui"

_DESCRIPTION = """\
Universal Dependencies (UD) is a project that is developing cross-linguistically consistent treebank annotation
 for many languages, with the goal of facilitating multilingual parser development, cross-lingual learning, and 
 parsing research from a language typology perspective. The annotation scheme is based on an evolution of (universal)
   Stanford dependencies (de Marneffe et al., 2006, 2008, 2014), Google universal part-of-speech tags 
   (Petrov et al., 2012), and the Interset interlingua for morphosyntactic tagsets (Zeman, 2008). 
   The general philosophy is to provide a universal inventory of categories and guidelines to facilitate consistent
     annotation of similar constructions across languages, while allowing language-specific extensions when necessary.
"""

_HOMEPAGE = "https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5287"

_LICENSE = "Apache license 2.0 (apache-2.0)"

_URLS = {
    "id_csui": {
        "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-CSUI/master/id_csui-ud-train.conllu",
        "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-CSUI/master/id_csui-ud-test.conllu",
    },
}

_SUPPORTED_TASKS = [Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"



class UdIdCSUIDataset(UDDataset):

    # def __init__(self, subset):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{_SUBSET}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME}_{_SUBSET} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_{_SUBSET}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{_SUBSET}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME}_{_SUBSET} SEACrowd Seq Label schema",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}_{_SUBSET}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{_SUBSET}_source"

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_SUBSET]
        data_path = dl_manager.download(urls)
        print(data_path)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path["train"]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_path["test"],
                },
            )
        ]


if __name__ == "__main__":
    data = datasets.load_dataset(__file__)