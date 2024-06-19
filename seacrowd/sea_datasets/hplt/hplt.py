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

import io
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import requests
import zstandard as zstd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import SCHEMA_TO_FEATURES, TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = r"""\
@inproceedings{aulamo-etal-2023-hplt,
    title = "{HPLT}: High Performance Language Technologies",
    author = {Aulamo, Mikko  and
        Bogoychev, Nikolay  and
        Ji, Shaoxiong  and
        Nail, Graeme  and
        Ram{\'\i}rez-S{\'a}nchez, Gema  and
        Tiedemann, J{\"o}rg  and
        van der Linde, Jelmer  and
        Zaragoza, Jaume},
    editor = "Nurminen, Mary  and
        Brenner, Judith  and
        Koponen, Maarit  and
        Latomaa, Sirkku  and
        Mikhailov, Mikhail  and
        Schierl, Frederike  and
        Ranasinghe, Tharindu  and
        Vanmassenhove, Eva  and
        Vidal, Sergi Alvarez  and
        Aranberri, Nora  and
        Nunziatini, Mara  and
        Escart{\'\i}n, Carla Parra  and
        Forcada, Mikel  and
        Popovic, Maja  and
        Scarton, Carolina  and
        Moniz, Helena",
    booktitle = "Proceedings of the 24th Annual Conference of the European
    Association for Machine Translation",
    month = jun,
    year = "2023",
    address = "Tampere, Finland",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2023.eamt-1.61",
    pages = "517--518",

    abstract = "We describe the High Performance Language Technologies project
    (HPLT), a 3-year EU-funded project started in September 2022. HPLT will
    build a space combining petabytes of natural language data with large-scale
    model training. It will derive monolingual and bilingual datasets from the
    Internet Archive and CommonCrawl and build efficient and solid machine
    translation (MT) as well as large language models (LLMs). HPLT aims at
    providing free, sustainable and reusable datasets, models and workflows at
    scale using high-performance computing (HPC).",
}
"""

_DATASETNAME = "hplt"

_DESCRIPTION = """\
The dataset is part of the High Performance Language Technologies project
(HPLT), a 3-year EU-funded project started in September 2022. HPLT derives
monolingual and bilingual datasets from the Internet Archive and CommonCrawl and
builds efficient and solid machine translation (MT) as well as large language
models (LLMs). HPLT aims at providing free, sustainable and reusable datasets,
models and workflows at scale using high-performance computing (HPC).
"""

_HOMEPAGE = "https://hplt-project.org/datasets/v1.2"

_LANGUAGES = {
    "ind": "id",
    "zlm": "ms",
    "tha": "th",
    "mya": "my",
    "fil": "tl",
    "vie": "vi"
}

_LICENSE = Licenses.CC0_1_0.value

_LOCAL = False

_URLS = {
    "raw": "https://data.hplt-project.org/one/monotext/{lang}_map.txt",
    "deduplicated": "https://data.hplt-project.org/one/monotext/deduplicated/{lang}_map.txt",
    "cleaned": "https://data.hplt-project.org/one/monotext/cleaned/{lang}_map.txt",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # ssp

_SOURCE_VERSION = "1.2.0"

_SEACROWD_VERSION = "2024.06.20"


class HpltDataset(datasets.GeneratorBasedBuilder):
    """HPLT derives monolingual and bilingual datasets from the Internet Archive and CommonCrawl"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SUBSETS = ["raw", "deduplicated", "cleaned"]

    BUILDER_CONFIGS = []
    for lang, subset in list(itertools.product(_LANGUAGES.keys(), SUBSETS)):
        subset_id = f"{lang}_{subset}"
        BUILDER_CONFIGS += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset_id}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} {subset_id} source schema",
                schema="source",
                subset_id=subset_id,
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset_id}_{_SEACROWD_SCHEMA}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {subset_id} SEACrowd schema",
                schema=_SEACROWD_SCHEMA,
                subset_id=subset_id,
            ),
        ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_mya_cleaned_source"  # smallest w.r.t. size

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "document_lang": datasets.Value("string"),
                    "scores": datasets.Sequence(datasets.Value("float")),
                    "langs": datasets.Sequence(datasets.Value("string")),
                    "text": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "collection": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[
                TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]
            ]  # ssp_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators. Data is not yet extracted for efficient generation."""
        lang, subset = self.config.subset_id.split("_")
        lang = _LANGUAGES[lang]
        map_url = _URLS[subset].format(lang=lang)

        response = requests.get(map_url, timeout=10)
        if response:
            data_urls = response.text.strip().split("\n")
            data_urls = [url for url in data_urls if url.endswith(".jsonl.zst")]
        else:
            raise requests.exceptions.HTTPError(
                f"Non-success status code: {response.status_code}"
            )

        data_paths = list(map(Path, dl_manager.download(data_urls)))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_paths": data_paths,
                },
            ),
        ]

    def _generate_examples(self, data_paths: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        key = 0
        for data_path in data_paths:
            with open(data_path, "rb") as f:
                # Zstandard decompression
                dctx = zstd.ZstdDecompressor()
                reader = dctx.stream_reader(f)
                text_io = io.TextIOWrapper(reader, encoding="utf-8")

                # read jsonl file by line and yield
                for line in text_io:
                    data = json.loads(line)
                    if self.config.schema == "source":
                        yield key, {
                            "id": key,
                            "document_lang": data["document_lang"],
                            "scores": data["scores"],
                            "langs": data["langs"],
                            "text": data["text"],
                            "url": data["url"],
                            "collection": data["collection"],
                        }
                    elif self.config.schema == _SEACROWD_SCHEMA:
                        yield key, {
                            "id": str(key),
                            "text": data["text"],
                        }
                    key += 1
