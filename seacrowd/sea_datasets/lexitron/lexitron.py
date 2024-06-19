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
Corpus-based dictionary of Thai and English languages. \
    This dataset contains frequently-used words from trusted \
    publications such as novels, academic documents and newspaper. \
    The dataset link contains Thai-English and English-Thai lexicons. \
    Thai-English vocabulary consists of vocabulary, type of word \
    (part of speech), translation, synonym (synonym) and sample sentences \
    with a list of Thai-> English words, 53,000 words and English vocabulary \
    list -> Thai, 83,000 words.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

# There are no citations available for this dataset.
_CITATION = ""

_DATASETNAME = "lexitron"

_DESCRIPTION = """
Corpus-based dictionary of Thai and English languages. \
    This dataset contains frequently-used words from trusted \
    publications such as novels, academic documents and newspaper. \
    The dataset link contains Thai-English and English-Thai lexicons. \
    Thai-English vocabulary consists of vocabulary, type of word \
    (part of speech), translation, synonym (synonym) and sample sentences \
    with a list of Thai-> English words, 53,000 words and English vocabulary \
    list -> Thai, 83,000 words.
"""

_HOMEPAGE = "https://opend-portal.nectec.or.th/dataset/lexitron-2-0"

_LANGUAGES = ["tha"]

_LICENSE = Licenses.OTHERS.value

_LOCAL = False

_URLS = {
    "telex": "https://opend-portal.nectec.or.th/dataset/bdd85296-9398-499f-b3a7-aab85042d3f9/resource/761924ea-937f-4be3-afe1-c031c754fa39/download/lexitron_2.0.zip",
    "etlex": "https://opend-portal.nectec.or.th/dataset/bdd85296-9398-499f-b3a7-aab85042d3f9/resource/761924ea-937f-4be3-afe1-c031c754fa39/download/lexitron_2.0.zip",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class LEXiTRONDataset(datasets.GeneratorBasedBuilder):
    """
    Corpus-based dictionary of Thai and English languages. \
    This dataset contains frequently-used words from trusted \
    publications such as novels, academic documents and newspaper. \
    The dataset link contains Thai-English and English-Thai lexicons. \
    Thai-English vocabulary consists of vocabulary, type of word \
    (part of speech), translation, synonym (synonym) and sample sentences \
    with a list of Thai-> English words, 53,000 words and English vocabulary \
    list -> Thai, 83,000 words.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "t2t"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_telex_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_telex",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_telex_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_telex",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_etlex_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_etlex",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_etlex_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_etlex",
        ),
    ]

    DEFAULT_CONFIG_NAME = "[dataset_name]_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            translation_type = self.config.name.split("_")[1]

            if translation_type == "telex":
                features = datasets.Features(
                    {
                        "id": datasets.Value("int64"),
                        "tsearch": datasets.Value("string"),
                        "tentry": datasets.Value("string"),
                        "eentry": datasets.Value("string"),
                        "tcat": datasets.Value("string"),
                        "tsyn": datasets.Value("string"),
                        "tsample": datasets.Value("string"),
                        "tdef": datasets.Value("string"),
                    }
                )

            elif translation_type == "etlex":
                features = datasets.Features(
                    {"id": datasets.Value("int64"), "esearch": datasets.Value("string"), "eentry": datasets.Value("string"), "tentry": datasets.Value("string"), "ecat": datasets.Value("string"), "esyn": datasets.Value("string")}
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

        translation_type = self.config.name.split("_")[1]
        data_dir = dl_manager.download_and_extract(_URLS[translation_type])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"LEXiTRON_2.0/{translation_type}"),
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        translation_type = self.config.name.split("_")[1]

        if translation_type == "telex":

            with open(filepath, "r", encoding="latin-1") as file:
                data = file.read()

            pattern = r"<Doc>(.*?)</Doc>"
            docs = re.findall(pattern, data, re.DOTALL)

            doc_data = []

            for doc in docs:
                tsearch = tentry = eentry = tcat = tsyn = tsample = tdef = id = None

                tsearch_match = re.search(r"<tsearch>(.*?)</tsearch>", doc)
                if tsearch_match:
                    tsearch = tsearch_match.group(1)

                tentry_match = re.search(r"<tentry>(.*?)</tentry>", doc)
                if tentry_match:
                    tentry = tentry_match.group(1)

                eentry_match = re.search(r"<eentry>(.*?)</eentry>", doc)
                if eentry_match:
                    eentry = eentry_match.group(1)

                tcat_match = re.search(r"<tcat>(.*?)</tcat>", doc)
                if tcat_match:
                    tcat = tcat_match.group(1)

                tsyn_match = re.search(r"<tsyn>(.*?)</tsyn>", doc)
                if tsyn_match:
                    tsyn = tsyn_match.group(1)

                tsample_match = re.search(r"<tsample>(.*?)</tsample>", doc)
                if tsample_match:
                    tsample = tsample_match.group(1)

                tdef_match = re.search(r"<tdef>(.*?)</tdef>", doc)
                if tdef_match:
                    tdef = tdef_match.group(1)

                id_match = re.search(r"<id>(.*?)</id>", doc)
                if id_match:
                    id = id_match.group(1)

                doc_data.append({"id": id, "tsearch": tsearch, "tentry": tentry, "eentry": eentry, "tcat": tcat, "tsyn": tsyn, "tsample": tsample, "tdef": tdef})

            df = pd.DataFrame(doc_data)

        if translation_type == "etlex":

            with open(filepath, "r", encoding="latin-1") as file:
                data = file.read()

            pattern = r"<Doc>(.*?)</Doc>"
            docs = re.findall(pattern, data, re.DOTALL)

            doc_data = []

            for doc in docs:
                esearch = eentry = tentry = ecat = esyn = id = None

                esearch_match = re.search(r"<esearch>(.*?)</esearch>", doc)
                if esearch_match:
                    esearch = esearch_match.group(1)

                eentry_match = re.search(r"<eentry>(.*?)</eentry>", doc)
                if eentry_match:
                    eentry = eentry_match.group(1)

                tentry_match = re.search(r"<tentry>(.*?)</tentry>", doc)
                if tentry_match:
                    tentry = tentry_match.group(1)

                ecat_match = re.search(r"<ecat>(.*?)</ecat>", doc)
                if ecat_match:
                    ecat = ecat_match.group(1)

                esyn_match = re.search(r"<esyn>(.*?)</esyn>", doc)
                if esyn_match:
                    esyn = esyn_match.group(1)

                id_match = re.search(r"<id>(.*?)</id>", doc)
                if id_match:
                    id = id_match.group(1)

                doc_data.append({"id": id, "esearch": esearch, "eentry": eentry, "tentry": tentry, "ecat": ecat, "esyn": esyn})

            df = pd.DataFrame(doc_data)

        for index, row in df.iterrows():

            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":

                if translation_type == "telex":
                    example = {
                        "id": str(index),
                        "text_1": str(row["tentry"]),
                        "text_2": str(row["eentry"]),
                        "text_1_name": "tentry",
                        "text_2_name": "eentry",
                    }

                if translation_type == "etlex":
                    example = {
                        "id": str(index),
                        "text_1": str(row["eentry"]),
                        "text_2": str(row["tentry"]),
                        "text_1_name": "eentry",
                        "text_2_name": "tentry",
                    }

            yield index, example
