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
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.sea_datasets.alt_burmese_treebank.utils.alt_burmese_treebank_utils import extract_data
from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{
    10.1145/3373268,
    author = {Ding, Chenchen and Yee, Sann Su Su and Pa, Win Pa and Soe, Khin Mar and Utiyama, Masao and Sumita, Eiichiro},
    title = {A Burmese (Myanmar) Treebank: Guideline and Analysis},
    year = {2020},
    issue_date = {May 2020},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {19},
    number = {3},
    issn = {2375-4699},
    url = {https://doi.org/10.1145/3373268},
    doi = {10.1145/3373268},
    abstract = {A 20,000-sentence Burmese (Myanmar) treebank on news articles has been released under a CC BY-NC-SA license.\
               Complete phrase structure annotation was developed for each sentence from the morphologically annotated data\
               prepared in previous work of Ding et&nbsp;al. [1]. As the final result of the Burmese component in the Asian\
               Language Treebank Project, this is the first large-scale, open-access treebank for the Burmese language.\
               The annotation details and features of this treebank are presented.\
               },
    journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
    month = {jan},
    articleno = {40},
    numpages = {13},
    keywords = {Burmese (Myanmar), phrase structure, treebank}
}
"""

_DATASETNAME = "alt_burmese_treebank"

_DESCRIPTION = """\
A 20,000-sentence Burmese (Myanmar) treebank on news articles containing complete phrase structure annotation.\
As the final result of the Burmese component in the Asian Language Treebank Project, this is the first large-scale,\
open-access treebank for the Burmese language.
"""

_HOMEPAGE = "https://zenodo.org/records/3463010"

_LANGUAGES = ["mya"]

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://zenodo.org/records/3463010/files/my-alt-190530.zip?download=1",
}

_SUPPORTED_TASKS = [Tasks.CONSTITUENCY_PARSING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class AltBurmeseTreebank(datasets.GeneratorBasedBuilder):
    """A 20,000-sentence Burmese (Myanmar) treebank on news articles containing complete phrase structure annotation.\
       As the final result of the Burmese component in the Asian Language Treebank Project, this is the first large-scale,\
       open-access treebank for the Burmese language."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_tree",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_tree",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "text": datasets.Value("string")})
        elif self.config.schema == "seacrowd_tree":
            features = schemas.tree_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "my-alt-190530/data"),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath, "r") as f:
                for idx, line in enumerate(f):
                    example = {"id": line.split("\t")[0], "text": line.split("\t")[1]}
                    yield idx, example

        elif self.config.schema == "seacrowd_tree":
            with open(filepath, "r") as f:
                for idx, line in enumerate(f):
                    example = extract_data(line)
                    yield idx, example
