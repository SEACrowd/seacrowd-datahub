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

""" \
The dataset contribution of this study is a compilation of short fictional stories \
written in Bikol for readability assessment. The data was combined other collected \
Philippine language corpora, such as Tagalog and Cebuano. The data from these languages \
are all distributed across the Philippine elementary system's first three grade \
levels (L1, L2, L3). We sourced this dataset from Let's Read Asia (LRA), Bloom Library, \
Department of Education, and Adarna House.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{imperial-kochmar-2023-automatic,
    title = "Automatic Readability Assessment for Closely Related Languages",
    author = "Imperial, Joseph Marvin  and
      Kochmar, Ekaterina",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.331",
    doi = "10.18653/v1/2023.findings-acl.331",
    pages = "5371--5386",
    abstract = "In recent years, the main focus of research on automatic readability assessment (ARA) \
    has shifted towards using expensive deep learning-based methods with the primary goal of increasing models{'} accuracy. \
    This, however, is rarely applicable for low-resource languages where traditional handcrafted features are still \
    widely used due to the lack of existing NLP tools to extract deeper linguistic representations. In this work, \
    we take a step back from the technical component and focus on how linguistic aspects such as mutual intelligibility \
    or degree of language relatedness can improve ARA in a low-resource setting. We collect short stories written in three \
    languages in the Philippines{---}Tagalog, Bikol, and Cebuano{---}to train readability assessment models and explore the \
    interaction of data and features in various cross-lingual setups. Our results show that the inclusion of CrossNGO, \
    a novel specialized feature exploiting n-gram overlap applied to languages with high mutual intelligibility, \
    significantly improves the performance of ARA models compared to the use of off-the-shelf large multilingual \
    language models alone. Consequently, when both linguistic representations are combined, we achieve state-of-the-art \
    results for Tagalog and Cebuano, and baseline scores for ARA in Bikol.",
}
"""

_DATASETNAME = "ara_close"

_DESCRIPTION = """\
The dataset contribution of this study is a compilation of short fictional stories \
written in Bikol for readability assessment. The data was combined other collected \
Philippine language corpora, such as Tagalog and Cebuano. The data from these languages \
are all distributed across the Philippine elementary system's first three grade \
levels (L1, L2, L3). We sourced this dataset from Let's Read Asia (LRA), Bloom Library, \
Department of Education, and Adarna House. \
"""

_HOMEPAGE = "https://github.com/imperialite/ara-close-lang"

_LANGUAGES = ["bcl", "ceb"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_4_0.value  # example: Licenses.MIT.value, Licenses.CC_BY_NC_SA_4_0.value, Licenses.UNLICENSE.value, Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "bcl": "https://raw.githubusercontent.com/imperialite/ara-close-lang/main/data/bikol/bik_all_data.txt",
    # 'tgl': '', # file for tgl language was deleted
    "ceb": "https://raw.githubusercontent.com/imperialite/ara-close-lang/main/data/cebuano/ceb_all_data.txt",
}

_SUPPORTED_TASKS = [Tasks.READABILITY_ASSESSMENT]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class AraCloseDataset(datasets.GeneratorBasedBuilder):
    f"""{_DESCRIPTION}"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [SEACrowdConfig(name=f"{_DATASETNAME}_{lang}_source", version=datasets.Version(_SOURCE_VERSION), description=f"{_DATASETNAME} source schema", schema="source", subset_id=f"{_DATASETNAME}",) for lang in _LANGUAGES] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}",
        )
        for lang in _LANGUAGES
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "title": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(["1", "2", "3"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        lang = self.config.name.split("_")[2]
        if lang in _LANGUAGES:
            data_path = Path(dl_manager.download_and_extract(_URLS[lang]))
        else:
            data_path = [Path(dl_manager.download_and_extract(_URLS[lang])) for lang in _LANGUAGES]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        lang = self.config.name.split("_")[2]
        if lang in _LANGUAGES:
            file_content = open(filepath, "r").readlines()
        else:
            file_content = []
            for path in filepath:
                lines = open(path, "r").readlines()
                file_content.extend(lines)

        if self.config.schema == "source":
            idx = 0
            for line in file_content:
                split_data = line.strip().split(",")
                title = split_data[0]
                label = split_data[1]
                text = ",".join(split_data[2:])
                ex = {"title": title, "text": text, "label": label}
                yield idx, ex
                idx += 1

        elif self.config.schema == "seacrowd_text":
            idx = 0
            for line in file_content:
                split_data = line.strip().split(",")
                title = split_data[0]
                label = split_data[1]
                text = ",".join(split_data[2:])
                ex = {
                    "id": idx,
                    "text": text,
                    "label": label,
                }
                yield idx, ex
                idx += 1
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
