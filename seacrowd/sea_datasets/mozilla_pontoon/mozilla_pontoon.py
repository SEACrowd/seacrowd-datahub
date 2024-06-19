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
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

# Keep blank; dataset has no associated paper
_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_LOCAL = False
_LANGUAGES = ["mya", "ceb", "gor", "hil", "ilo", "ind", "jav", "khm", "lao", "zlm", "nia", "tgl", "tha", "vie"]

_DATASETNAME = "mozilla_pontoon"
_DESCRIPTION = """
This dataset contains crowdsource translations of more than 200 languages for
different Mozilla open-source projects from Mozilla's Pontoon localization platform.
Source sentences are in English.
"""

_HOMEPAGE = "https://huggingface.co/datasets/ayymen/Pontoon-Translations"
_LICENSE = Licenses.BSD_3_CLAUSE.value
_URL = "https://huggingface.co/datasets/ayymen/Pontoon-Translations"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class MozillaPontoonDataset(datasets.GeneratorBasedBuilder):
    """Dataset of translations from Mozilla's Pontoon platform."""

    # Two-letter ISO code is used when available
    # otherwise 3-letter one is used
    LANG_CODE_MAPPER = {"mya": "my", "ceb": "ceb", "gor": "gor", "hil": "hil", "ilo": "ilo", "ind": "id", "jav": "jv", "khm": "km", "lao": "lo", "zlm": "ms", "nia": "nia", "tgl": "tl", "tha": "th", "vie": "vi"}

    # Config to load individual datasets per language
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_eng_{lang}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema for {lang} language",
            schema="source",
            subset_id=f"{_DATASETNAME}_eng_{lang}",
        )
        for lang in _LANGUAGES
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_eng_{lang}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema for {lang} language",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_eng_{lang}",
        )
        for lang in _LANGUAGES
    ]

    # Config to load all datasets
    BUILDER_CONFIGS.extend(
        [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} source schema for all languages",
                schema="source",
                subset_id=_DATASETNAME,
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_t2t",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} SEACrowd schema for all languages",
                schema="seacrowd_t2t",
                subset_id=_DATASETNAME,
            ),
        ]
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "source_string": datasets.Value("string"),
                    "target_string": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_t2t":
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
        # dl_manager not used since dataloader uses HF 'load_dataset'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train"},
            ),
        ]

    def _load_hf_data_from_remote(self, language: str) -> datasets.DatasetDict:
        """Load dataset from HuggingFace."""
        hf_lang_code = self.LANG_CODE_MAPPER[language]
        hf_remote_ref = "/".join(_URL.split("/")[-2:])
        return datasets.load_dataset(hf_remote_ref, f"en-{hf_lang_code}", split="train")

    def _generate_examples(self, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        languages = []
        pontoon_datasets = []

        lang = self.config.subset_id.split("_")[-1]
        if lang in _LANGUAGES:
            languages.append(lang)
            pontoon_datasets.append(self._load_hf_data_from_remote(lang))
        else:
            for lang in _LANGUAGES:
                languages.append(lang)
                pontoon_datasets.append(self._load_hf_data_from_remote(lang))

        index = 0
        for lang, lang_subset in zip(languages, pontoon_datasets):
            for row in lang_subset:
                if self.config.schema == "source":
                    example = row

                elif self.config.schema == "seacrowd_t2t":
                    example = {
                        "id": str(index),
                        "text_1": row["source_string"],
                        "text_2": row["target_string"],
                        "text_1_name": "eng",
                        "text_2_name": lang,
                    }
                yield index, example
                index += 1
