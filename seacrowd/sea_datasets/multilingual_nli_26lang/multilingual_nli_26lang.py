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

import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd
from huggingface_hub import HfFileSystem

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@article{laurer_less_2022,
    title = {Less Annotating, More Classifying: Addressing the Data Scarcity
    Issue of Supervised Machine Learning with Deep Transfer Learning and
    BERT-NLI},
    url = {https://osf.io/74b8k},
    language = {en-us},
    urldate = {2022-07-28},
    journal = {Preprint},
    author = {Laurer, Moritz and
        Atteveldt, Wouter van and
        Casas, Andreu Salleras and
        Welbers, Kasper},
    month = jun,
    year = {2022},
    note = {Publisher: Open Science Framework},
}
"""

_DATASETNAME = "multilingual_nli_26lang"

_DESCRIPTION = """\
This dataset contains 2 730 000 NLI text pairs in 26 languages spoken by more
than 4 billion people. The dataset can be used to train models for multilingual
NLI (Natural Language Inference) or zero-shot classification. The dataset is
based on the English datasets MultiNLI, Fever-NLI, ANLI, LingNLI and WANLI and
was created using the latest open-source machine translation models.
"""

_HOMEPAGE = "https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7"

_LANGUAGES = ["ind", "vie"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_BASE_URL = "https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7/resolve/main/data/{file_name}"

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # pairs

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MultilingualNLI26LangDataset(datasets.GeneratorBasedBuilder):
    """NLI dataset in 26 languages, created using machine translation models"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SUBSETS = ["anli", "fever", "ling", "mnli", "wanli"]

    BUILDER_CONFIGS = []
    for lang, subset in list(itertools.product(_LANGUAGES, SUBSETS)):
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

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_ind_anli_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "premise_original": datasets.Value("string"),
                    "hypothesis_original": datasets.Value("string"),
                    "label": datasets.Value("int64"),
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = schemas.pairs_features(label_names=["entailment", "neutral", "contradiction"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        file_list = HfFileSystem().ls("datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7/data", detail=False)

        subset_config = self.config.subset_id
        if "ind" in subset_config:
            subset_config = subset_config.replace("ind", "id")
        if "vie" in subset_config:
            subset_config = subset_config.replace("vie", "vi")

        data_urls = []
        for file_path in file_list:
            file_name = file_path.split("/")[-1]
            subset_id = file_name.split("-")[0]
            if subset_id == subset_config:
                if file_path.endswith(".parquet"):
                    url = _BASE_URL.format(file_name=file_name)
                    data_urls.append(url)

        data_paths = list(map(Path, dl_manager.download_and_extract(data_urls)))
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
            df = pd.read_parquet(data_path)

            for _, row in df.iterrows():
                if self.config.schema == "source":
                    yield key, {
                        "premise_original": row["premise_original"],
                        "hypothesis_original": row["hypothesis_original"],
                        "label": row["label"],
                        "premise": row["premise"],
                        "hypothesis": row["hypothesis"],
                    }
                    key += 1
                elif self.config.schema == _SEACROWD_SCHEMA:
                    yield key, {
                        "id": str(key),
                        "text_1": row["premise"],
                        "text_2": row["hypothesis"],
                        "label": row["label"],
                    }
                    key += 1
