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
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{elkishky_ccaligned_2020,
    author = {El-Kishky, Ahmed and Chaudhary, Vishrav and Guzm{\'a}n, Francisco and Koehn, Philipp},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)},
    month = {November},
    title = {{CCAligned}: A Massive Collection of Cross-lingual Web-Document Pairs},
    year = {2020}
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.480",
    doi = "10.18653/v1/2020.emnlp-main.480",
    pages = "5960--5969"
}
"""

_DATASETNAME = "cc_aligned_doc"

_DESCRIPTION = """\
CCAligned consists of parallel or comparable web-document pairs in 137 languages aligned with English\
(10 languages are from Southeast Asia; Burmese has two document collection with different scripts).\
These web-document pairs were constructed by performing language identification on raw web-documents, \
and ensuring corresponding language codes were corresponding in the URLs of web documents. This pattern \
matching approach yielded more than 100 million aligned documents paired with English.
"""

_HOMEPAGE = "https://www2.statmt.org/cc-aligned/"

_LANGUAGES = ["ind", "sun", "tha", "vie", "zlm", "lao", "khm", "mya", "ceb", "war"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False
_SUBSETS = {"id_ID": "ind", "su_ID": "sun", "th_TH": "tha", "vi_VN": "vie", "ms_MY": "zlm", "lo_LA": "lao", "km_KH": "khm", "my_MM": "mya", "my_MM_zaw": "mya", "cx_PH": "ceb", "wy_PH": "war"}
_URLS = {_DATASETNAME: "https://data.statmt.org/cc-aligned/en_XX-{subset}.tsv.xz"}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class CCAlignedDocDataset(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "t2t"

    BUILDER_CONFIGS = [SEACrowdConfig(name=f"{_DATASETNAME}_{subset}_source", version=datasets.Version(_SOURCE_VERSION), description=f"{_DATASETNAME} source schema", schema="source", subset_id=f"{_DATASETNAME}",) for subset in _SUBSETS.keys()] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_seacrowd_{schema_name}",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{schema_name}",
            subset_id=f"{_DATASETNAME}",
        )
        for subset, schema_name in zip(_SUBSETS.keys(), len(_SUBSETS.keys()) * [SEACROWD_SCHEMA_NAME])
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_id_ID_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "Domain": datasets.Value("string"),
                    "Source_URL": datasets.Value("string"),
                    "Source_Content": datasets.Value("string"),
                    "Target_URL": datasets.Value("string"),
                    "Target_Content": datasets.Value("string"),
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
        subset = "_".join([self.config.name.split("_")[3], self.config.name.split("_")[4]])
        urls = _URLS[_DATASETNAME].format(subset=subset)
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        subset = "_".join([self.config.name.split("_")[3], self.config.name.split("_")[4]])
        lines = open(filepath, "r").readlines()
        if self.config.schema == "source":
            idx = 0
            for line in lines:
                content = line.split("\t")
                example = {
                    "Domain": content[0],
                    "Source_URL": content[1],
                    "Source_Content": content[2],
                    "Target_URL": content[3],
                    "Target_Content": content[4],
                }
                yield idx, example
                idx += 1
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            idx = 0
            for line in lines:
                content = line.split("\t")
                example = {
                    "id": str(idx),
                    "text_1": content[2],
                    "text_2": content[4],
                    "text_1_name": "en",
                    "text_2_name": _SUBSETS[subset],
                }
                yield idx, example
                idx += 1
