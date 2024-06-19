# coding=utf-8
# Copyright 2024 The HuggingFace Datasets Authors and the current dataset script contributor.
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
SIB-200 is the largest publicly available topic classification dataset based on Flores-200 covering 205 languages and dialects.
The train/validation/test sets are available for all the 205 languages.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@misc{adelani2023sib200,
      title={SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects},
      author={David Ifeoluwa Adelani and Hannah Liu and Xiaoyu Shen and Nikita Vassilyev and Jesujoba O. Alabi and Yanke Mao and Haonan Gao and Annie En-Shiun Lee},
      year={2023},
      eprint={2309.07445},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "sib_200"

_DESCRIPTION = """\
SIB-200 is the largest publicly available topic classification dataset based on Flores-200 covering 205 languages and dialects.
The train/validation/test sets are available for all the 205 languages.
"""

_HOMEPAGE = "https://github.com/dadelani/sib-200"

_LANGUAGES = [
    "ace",
    "ban",
    "bjn",
    "bug",
    "ceb",
    "ilo",
    "ind",
    "jav",
    "kac",
    "khm",
    "lao",
    "lus",
    "min",
    "mya",
    "pag",
    "shn",
    "sun",
    "tgl",
    "tha",
    "vie",
    "war",
    "zsm",
]

_SUPPORTED_LANGUAGE_CODES = [
    "ace_Arab",
    "ace_Latn",
    "ban_Latn",
    "bjn_Arab",
    "bjn_Latn",
    "bug_Latn",
    "ceb_Latn",
    "ilo_Latn",
    "ind_Latn",
    "jav_Latn",
    "kac_Latn",
    "khm_Khmr",
    "lao_Laoo",
    "lus_Latn",
    "min_Arab",
    "min_Latn",
    "mya_Mymr",
    "pag_Latn",
    "shn_Mymr",
    "sun_Latn",
    "tgl_Latn",
    "tha_Thai",
    "vie_Latn",
    "war_Latn",
    "zsm_Latn",
]

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "base_url": "https://huggingface.co/datasets/Davlan/sib200/raw/main/data"
}

_SUPPORTED_TASKS = [Tasks.TOPIC_MODELING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_SEACROWD_SCHEMA = f"seacrowd_text"


def _sib_config_constructor(lang: str, schema: str = _SEACROWD_SCHEMA, version: str = _SEACROWD_VERSION) -> SEACrowdConfig:
    return SEACrowdConfig(
        name=f"{_DATASETNAME}_{lang}_{schema}",
        version=version,
        description=f"SIB-200 {schema} schema",
        schema=schema,
        subset_id=f"SIB-200 {lang}",
    )


class Sib200Dataset(datasets.GeneratorBasedBuilder):
    """
    SIB-200 is the largest publicly available topic classification dataset based on Flores-200 covering 205 languages and dialects.
    The train/validation/test sets are available for all the 205 languages.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    def _populate_configs():
        configs = [_sib_config_constructor(lang, schema="source", version=_SOURCE_VERSION) for lang in _SUPPORTED_LANGUAGE_CODES] + [_sib_config_constructor(lang, schema=_SEACROWD_SCHEMA, version=_SEACROWD_VERSION) for lang in _SUPPORTED_LANGUAGE_CODES]

        all_lang_source_config = SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=_SOURCE_VERSION,
            description=f"SIB-200 source schema",
            schema="source",
            subset_id=f"SIB-200 SEA",
        )

        all_lang_t2t_config = SEACrowdConfig(
            name=f"{_DATASETNAME}_{_SEACROWD_SCHEMA}",
            version=_SEACROWD_VERSION,
            description=f"SIB-200 {_SEACROWD_SCHEMA} schema",
            schema=_SEACROWD_SCHEMA,
            subset_id=f"SIB-200 SEA",
        )

        configs.append(all_lang_source_config)
        configs.append(all_lang_t2t_config)
        return configs

    BUILDER_CONFIGS = _populate_configs()

    DEFAULT_CONFIG_NAME = "sib_200_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index_id": datasets.Value("int64"),
                    "text": datasets.Value("string"),
                    "category": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(["geography", "science/technology", "health", "travel", "entertainment", "politics", "sports"])

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
        lang = self.config.subset_id.split(" ")[-1]
        if lang in _SUPPORTED_LANGUAGE_CODES:
            train_paths = [Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/{lang}/train.tsv"))]
            valid_paths = [Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/{lang}/dev.tsv"))]
            test_paths = [Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/{lang}/test.tsv"))]
            lang_codes = [lang]
        elif lang == "SEA":
            train_paths, valid_paths, test_paths, lang_codes = [], [], [], []
            for lang in _SUPPORTED_LANGUAGE_CODES:                
                train_paths.append(Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/{lang}/train.tsv")))
                valid_paths.append(Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/{lang}/dev.tsv")))
                test_paths.append(Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/{lang}/test.tsv")))
                lang_codes.append(lang)
        else:
            raise ValueError(f"Language {lang} not a SEA language in the dataset")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"file_paths": train_paths, "lang_codes": lang_codes}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={"file_paths": valid_paths, "lang_codes": lang_codes}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"file_paths": test_paths, "lang_codes": lang_codes}
            )
        ]

    def _generate_examples(self, file_paths: List[str], lang_codes: List[str]) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        index = 0
        for file_path, lang_code in zip(file_paths, lang_codes):
            lang_df = pd.read_csv(file_path, sep='\t')
            for row in lang_df.itertuples():
                if self.config.schema == "source":
                    example = {
                        "index_id": row.index_id,
                        "text": row.text,
                        "category": row.category,
                    }

                elif self.config.schema == "seacrowd_text":
                    example = {
                        "id": f'{lang_code}_{row.index_id}',
                        "text": row.text,
                        "label": row.category,
                    }
                yield index, example
                index += 1
