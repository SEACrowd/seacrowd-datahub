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
A parallel corpus of KDE4 localization files. The corpus is available in 92 languages in total, with 4099 bitexts.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{tiedemann2012parallel,
  title={Parallel Data, Tools and Interfaces in OPUS},
  author={Tiedemann, J{\"o}rg},
  booktitle={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
  pages={2214--2218},
  year={2012}
}
"""

_DATASETNAME = "kde4"

_DESCRIPTION = """\
A parallel corpus of KDE4 localization files. The corpus is available in 92 languages in total, with 4099 bitexts.
"""

_HOMEPAGE = "https://opus.nlpl.eu/KDE4.php"

# TODO: Add languages related to this dataset
_LANGUAGES = ["ind", "khm", "zlm", "tha", "vie"]

_LICENSE = Licenses.UNKNOWN.value
_LOCAL = False

_URL_TEMPLATE = "https://opus.nlpl.eu/download.php?f=KDE4/v2/moses/{src}-{tgt}.txt.zip"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "2.0.0"
_SEACROWD_VERSION = "1.0.0"

configs = {
    "af": ["ms", "th", "km", "id", "vi"],
    "ar": ["ms", "th", "km", "id", "vi"],
    "as": ["ms", "th", "km", "id", "vi"],
    "ast": ["ms", "th", "km", "id", "vi"],
    "be": ["ms", "th", "km", "id", "vi"],
    "bg": ["ms", "th", "km", "id", "vi"],
    "bn": ["ms", "th", "km", "id", "vi"],
    "br": ["ms", "th", "km", "id", "vi"],
    "ca": ["ms", "th", "km", "id", "vi"],
    "crh": ["ms", "th", "km", "id", "vi"],
    "cs": ["ms", "th", "km", "id", "vi"],
    "csb": ["ms", "th", "km", "id", "vi"],
    "cy": ["ms", "th", "km", "id", "vi"],
    "da": ["ms", "th", "km", "id", "vi"],
    "de": ["ms", "th", "km", "id", "vi"],
    "el": ["ms", "th", "km", "id", "vi"],
    "en": ["ms", "th", "km", "id", "vi"],
    "eo": ["ms", "th", "km", "id", "vi"],
    "es": ["ms", "th", "km", "id", "vi"],
    "et": ["ms", "th", "km", "id", "vi"],
    "eu": ["ms", "th", "km", "id", "vi"],
    "fa": ["ms", "th", "km", "id", "vi"],
    "fi": ["ms", "th", "km", "id", "vi"],
    "fr": ["ms", "th", "km", "id", "vi"],
    "fy": ["ms", "th", "km", "id", "vi"],
    "ga": ["ms", "th", "km", "id", "vi"],
    "gl": ["ms", "th", "km", "id", "vi"],
    "gu": ["ms", "th", "km", "id", "vi"],
    "ha": ["ms", "th", "km", "id", "vi"],
    "he": ["ms", "th", "km", "id", "vi"],
    "hi": ["ms", "th", "km", "id", "vi"],
    "hne": ["ms", "th", "km", "id", "vi"],
    "hr": ["ms", "th", "km", "id", "vi"],
    "hsb": ["ms", "th", "km", "id", "vi"],
    "hu": ["ms", "th", "km", "id", "vi"],
    "hy": ["ms", "th", "km", "id", "vi"],
    "id": [
        "kn",
        "ps",
        "ms",
        "sl",
        "tr",
        "ru",
        "nl",
        "mk",
        "ja",
        "or",
        "ne",
        "xh",
        "nds",
        "lv",
        "uk",
        "vi",
        "mai",
        "ta",
        "lb",
        "is",
        "uz",
        "se",
        "lt",
        "tg",
        "ka",
        "ml",
        "sr",
        "wa",
        "pt",
        "oc",
        "ku",
        "mr",
        "si",
        "sk",
        "ko",
        "kk",
        "ro",
        "nn",
        "th",
        "km",
        "te",
        "it",
        "pl",
        "sv",
        "pa",
        "nb",
    ],
    "is": ["km", "ms", "th", "vi"],
    "it": ["km", "ms", "th", "vi"],
    "ja": ["km", "ms", "th", "vi"],
    "ka": ["km", "ms", "th", "vi"],
    "kk": ["km", "ms", "th", "vi"],
    "km": [
        "kn",
        "ps",
        "ms",
        "sl",
        "tr",
        "ru",
        "rw",
        "nl",
        "mk",
        "or",
        "xh",
        "nso",
        "ne",
        "nds",
        "lv",
        "uk",
        "vi",
        "mai",
        "ta",
        "lb",
        "uz",
        "se",
        "lt",
        "tg",
        "mt",
        "ml",
        "sr",
        "wa",
        "pt",
        "oc",
        "ku",
        "mr",
        "si",
        "sk",
        "ko",
        "ro",
        "nn",
        "th",
        "te",
        "pl",
        "sv",
        "pa",
        "nb",
    ],
    "kn": ["ms", "th", "vi"],
    "ko": ["ms", "th", "vi"],
    "ku": ["ms", "th", "vi"],
    "lb": ["ms", "th", "vi"],
    "lt": ["ms", "th", "vi"],
    "lv": ["ms", "th", "vi"],
    "mai": ["ms", "th", "vi"],
    "mk": ["ms", "th", "vi"],
    "ml": ["ms", "th", "vi"],
    "mr": ["ms", "th", "vi"],
    "ms": ["ps", "sl", "tr", "ru", "rw", "nl", "or", "xh", "nso", "ne", "nds", "uk", "vi", "ta", "uz", "se", "tg", "mt", "sr", "wa", "pt", "oc", "sk", "si", "ro", "nn", "th", "te", "pl", "sv", "pa", "nb"],
    "mt": ["th"],
    "nb": ["th", "vi"],
    "nds": ["th", "vi"],
    "ne": ["th", "vi"],
    "nl": ["th", "vi"],
    "nn": ["th", "vi"],
    "oc": ["th", "vi"],
    "or": ["th", "vi"],
    "pa": ["th", "vi"],
    "pl": ["th", "vi"],
    "ps": ["th", "vi"],
    "pt": ["th", "vi"],
    "ro": ["th", "vi"],
    "ru": ["th", "vi"],
    "rw": ["th"],
    "se": ["th", "vi"],
    "si": ["th", "vi"],
    "sk": ["th", "vi"],
    "sl": ["th", "vi"],
    "sr": ["th", "vi"],
    "sv": ["th", "vi"],
    "ta": ["th", "vi"],
    "te": ["th", "vi"],
    "tg": ["th", "vi"],
    "th": ["wa", "tr", "uz", "xh", "vi", "uk"],
    "tr": ["vi"],
    "uk": ["vi"],
    "uz": ["vi"],
    "vi": ["wa", "xh"],
}


class KDE4Dataset(datasets.GeneratorBasedBuilder):
    """TODO: A parallel corpus of KDE4 localization files. The corpus is available in 92 languages in total, with 4099 bitexts."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = (
        [
            SEACrowdConfig(name="kde4_source", version=datasets.Version(_SOURCE_VERSION), description="kde4 source schema for af to ms", schema="source", subset_id="af_ms"),
            SEACrowdConfig(name="kde4_seacrowd_t2t", version=datasets.Version(_SOURCE_VERSION), description="kde4 seacrowd_t2t schema for af to ms", schema="seacrowd_t2t", subset_id="af_ms"),
        ]
        + [
            SEACrowdConfig(name=f"kde4_{src_lang}_{tgt_lang}_source", version=datasets.Version(_SOURCE_VERSION), description=f"kde4 source schema for {src_lang} to {tgt_lang}", schema="source", subset_id=f"{src_lang}_{tgt_lang}")
            for src_lang in configs
            for tgt_lang in configs[src_lang]
        ]
        + [
            SEACrowdConfig(name=f"kde4_{src_lang}{tgt_lang}_seacrowd_t2t", version=datasets.Version(_SEACROWD_VERSION), description=f"kde4 seacrowd_t2t schema for {src_lang} to {tgt_lang}", schema="seacrowd_t2t", subset_id=f"{src_lang}_{tgt_lang}")
            for src_lang in configs
            for tgt_lang in configs[src_lang]
        ]
    )

    DEFAULT_CONFIG_NAME = "kde4_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "src_text": datasets.Value("string"), "tgt_text": datasets.Value("string")})

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

        src_lang, tgt_lang = self.config.subset_id.split("_")
        url = _URL_TEMPLATE.format(src=src_lang, tgt=tgt_lang)
        data_dir = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "src_filepath": os.path.join(data_dir, f"KDE4.{src_lang}-{tgt_lang}.{src_lang}"),
                    "tgt_filepath": os.path.join(data_dir, f"KDE4.{src_lang}-{tgt_lang}.{tgt_lang}"),
                },
            )
        ]

    def _generate_examples(self, src_lang: str, tgt_lang: str, src_filepath: Path, tgt_filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            for row_id, (src_text, tgt_text) in enumerate(zip(open(src_filepath), open(tgt_filepath))):
                yield row_id, {"id": row_id, "src_text": src_text.strip(), "tgt_text": tgt_text.strip()}

        elif self.config.schema == "seacrowd_t2t":
            for row_id, (src_text, tgt_text) in enumerate(zip(open(src_filepath), open(tgt_filepath))):
                yield row_id, {"id": row_id, "text_1": src_text.strip(), "text_2": tgt_text.strip(), "text_1_name": src_lang, "text_2_name": tgt_lang}
