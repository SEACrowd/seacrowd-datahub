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
NTREX-128, a data set for machine translation (MT) evaluation, includes 123 documents \
(1,997 sentences, 42k words) translated from English into 128 target languages. \
9 languages are natively spoken in Southeast Asia, i.e., Burmese, Filipino, \
Hmong, Indonesian, Khmer, Lao, Malay, Thai, and Vietnamese.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{federmann-etal-2022-ntrex,
    title = "{NTREX}-128 {--} News Test References for {MT} Evaluation of 128 Languages",
    author = "Federmann, Christian  and
      Kocmi, Tom  and
      Xin, Ying",
    editor = "Ahuja, Kabir  and
      Anastasopoulos, Antonios  and
      Patra, Barun  and
      Neubig, Graham  and
      Choudhury, Monojit  and
      Dandapat, Sandipan  and
      Sitaram, Sunayana  and
      Chaudhary, Vishrav",
    booktitle = "Proceedings of the First Workshop on Scaling Up Multilingual Evaluation",
    month = nov,
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.sumeval-1.4",
    pages = "21--24",
}
"""

_DATASETNAME = "ntrex_128"

_DESCRIPTION = """\
NTREX-128, a data set for machine translation (MT) evaluation, includes 123 documents \
(1,997 sentences, 42k words) translated from English into 128 target languages. \
9 languages are natively spoken in Southeast Asia, i.e., Burmese, Filipino, \
Hmong, Indonesian, Khmer, Lao, Malay, Thai, and Vietnamese.
"""

_HOMEPAGE = "https://github.com/MicrosoftTranslator/NTREX"

_LANGUAGES = ["mya", "fil", "ind", "khm", "lao", "zlm", "tha", "vie", "hmv", "eng"]

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

# _MAPPING = {"mya": "mya", "fil": "fil", "ind": "ind", "khm": "khm", "lao": "lao", "zlm": "msa", "tha": "tha", "vie": "vie", "hmv": "hmn"}
_MAPPING = {
    "afr": "afr",
    "amh": "amh",
    "arb": "arb",
    "aze-Latn": "aze-Latn",
    "bak": "bak",
    "bel": "bel",
    "bem": "bem",
    "ben": "ben",
    "bod": "bod",
    "bos": "bos",
    "bul": "bul",
    "cat": "cat",
    "ces": "ces",
    "ckb-Arab": "ckb-Arab",
    "cym": "cym",
    "dan": "dan",
    "deu": "deu",
    "div": "div",
    "dzo": "dzo",
    "ell": "ell",
    "eng-GB": "eng-GB",
    "eng-IN": "eng-IN",
    "eng-US": "eng-US",
    "est": "est",
    "eus": "eus",
    "ewe": "ewe",
    "fao": "fao",
    "fas": "fas",
    "fij": "fij",
    "fil": "fil",
    "fin": "fin",
    "fra": "fra",
    "fra-CA": "fra-CA",
    "fuc": "fuc",
    "gle": "gle",
    "glg": "glg",
    "guj": "guj",
    "hau": "hau",
    "heb": "heb",
    "hin": "hin",
    "hmv": "hmn",
    "hrv": "hrv",
    "hun": "hun",
    "hye": "hye",
    "ibo": "ibo",
    "ind": "ind",
    "isl": "isl",
    "ita": "ita",
    "jpn": "jpn",
    "kan": "kan",
    "kat": "kat",
    "kaz": "kaz",
    "khm": "khm",
    "kin": "kin",
    "kir": "kir",
    "kmr": "kmr",
    "kor": "kor",
    "lao": "lao",
    "lav": "lav",
    "lit": "lit",
    "ltz": "ltz",
    "mal": "mal",
    "mar": "mar",
    "mey": "mey",
    "mkd": "mkd",
    "mlg": "mlg",
    "mlt": "mlt",
    "mon": "mon",
    "mri": "mri",
    "zlm": "msa",
    "mya": "mya",
    "nde": "nde",
    "nep": "nep",
    "nld": "nld",
    "nno": "nno",
    "nob": "nob",
    "nso": "nso",
    "nya": "nya",
    "orm": "orm",
    "pan": "pan",
    "pol": "pol",
    "por": "por",
    "por-BR": "por-BR",
    "prs": "prs",
    "pus": "pus",
    "ron": "ron",
    "rus": "rus",
    "shi": "shi",
    "sin": "sin",
    "slk": "slk",
    "slv": "slv",
    "smo": "smo",
    "sna-Latn": "sna-Latn",
    "snd-Arab": "snd-Arab",
    "som": "som",
    "spa": "spa",
    "spa-MX": "spa-MX",
    "sqi": "sqi",
    "srp-Cyrl": "srp-Cyrl",
    "srp-Latn": "srp-Latn",
    "ssw": "ssw",
    "swa": "swa",
    "swe": "swe",
    "tah": "tah",
    "tam": "tam",
    "tat": "tat",
    "tel": "tel",
    "tgk-Cyrl": "tgk-Cyrl",
    "tha": "tha",
    "tir": "tir",
    "ton": "ton",
    "tsn": "tsn",
    "tuk": "tuk",
    "tur": "tur",
    "uig": "uig",
    "ukr": "ukr",
    "urd": "urd",
    "uzb": "uzb",
    "ven": "ven",
    "vie": "vie",
    "wol": "wol",
    "xho": "xho",
    "yor": "yor",
    "yue": "yue",
    "zho-CN": "zho-CN",
    "zho-TW": "zho-TW",
    "zul": "zul",
}
_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-ref.{lang}.txt",
}

_ALL_LANG = [
    "afr",
    "amh",
    "arb",
    "aze-Latn",
    "bak",
    "bel",
    "bem",
    "ben",
    "bod",
    "bos",
    "bul",
    "cat",
    "ces",
    "ckb-Arab",
    "cym",
    "dan",
    "deu",
    "div",
    "dzo",
    "ell",
    "eng-GB",
    "eng-IN",
    "eng-US",
    "est",
    "eus",
    "ewe",
    "fao",
    "fas",
    "fij",
    "fil",
    "fin",
    "fra",
    "fra-CA",
    "fuc",
    "gle",
    "glg",
    "guj",
    "hau",
    "heb",
    "hin",
    "hmv",
    "hrv",
    "hun",
    "hye",
    "ibo",
    "ind",
    "isl",
    "ita",
    "jpn",
    "kan",
    "kat",
    "kaz",
    "khm",
    "kin",
    "kir",
    "kmr",
    "kor",
    "lao",
    "lav",
    "lit",
    "ltz",
    "mal",
    "mar",
    "mey",
    "mkd",
    "mlg",
    "mlt",
    "mon",
    "mri",
    "zlm",
    "mya",
    "nde",
    "nep",
    "nld",
    "nno",
    "nob",
    "nso",
    "nya",
    "orm",
    "pan",
    "pol",
    "por",
    "por-BR",
    "prs",
    "pus",
    "ron",
    "rus",
    "shi",
    "sin",
    "slk",
    "slv",
    "smo",
    "sna-Latn",
    "snd-Arab",
    "som",
    "spa",
    "spa-MX",
    "sqi",
    "srp-Cyrl",
    "srp-Latn",
    "ssw",
    "swa",
    "swe",
    "tah",
    "tam",
    "tat",
    "tel",
    "tgk-Cyrl",
    "tha",
    "tir",
    "ton",
    "tsn",
    "tuk",
    "tur",
    "uig",
    "ukr",
    "urd",
    "uzb",
    "ven",
    "vie",
    "wol",
    "xho",
    "yor",
    "yue",
    "zho-CN",
    "zho-TW",
    "zul",
]

# aze-Latn: Azerbaijani (Latin)
# ckb-Arab: Central Kurdish (Sorani)
# eng-GB: English (British), eng-IN: English (India), eng-US: English (US)
# fra: French, fra-CA: French (Canada)
# mya: Myanmar
# por: Portuguese, por-BR: Portuguese (Brazil)
# shi: Shilha
# sna-Latn: Shona (Latin)
# snd-Arab: Sindhi (Arabic)
# spa: Spanish, spa-MX: Spanish (Mexico)
# srp-Cyrl: Serbian (Cyrillic), srp-Latn: Serbian (Latin)
# tgk-Cyrl: Tajik (Cyrillic)
# yue: Cantonese
# zho-CN: Chinese (Simplified), zho-TW: Chinese (Traditional)

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "11.24.2022"

_SEACROWD_VERSION = "2024.06.20"


class Ntrex128Dataset(datasets.GeneratorBasedBuilder):
    """NTREX-128, a data set for machine translation (MT) evaluation, includes 123 documents \
    (1,997 sentences, 42k words) translated from English into 128 target languages. \
    9 languages are natively spoken in Southeast Asia, i.e., Burmese, Filipino, \
    Hmong, Indonesian, Khmer, Lao, Malay, Thai, and Vietnamese."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset1}_{subset2}_source",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} {subset1}2{subset2} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_{subset1}_{subset2}",
        )
        for subset2 in _ALL_LANG
        for subset1 in _ALL_LANG
        if subset1 != subset2 and (subset1 in _LANGUAGES or subset2 in _LANGUAGES)
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset1}_{subset2}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} {subset1}2{subset2} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_{subset1}_{subset2}",
        )
        for subset2 in _ALL_LANG
        for subset1 in _ALL_LANG
        if subset1 != subset2 and (subset1 in _LANGUAGES or subset2 in _LANGUAGES)
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_mya_fil_source"

    def _info(self):
        # The format of the source is just texts in different .txt files (each file corresponds to one language).
        # Decided make source schema the same as the seacrowd_t2t schema.
        if self.config.schema == "source" or self.config.schema == "seacrowd_t2t":
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
        lang1 = self.config.name.split("_")[2]
        lang2 = self.config.name.split("_")[3]
        lang1_txt_path = Path(dl_manager.download_and_extract(_URLS[_DATASETNAME].format(lang=_MAPPING[lang1])))
        lang2_txt_path = Path(dl_manager.download_and_extract(_URLS[_DATASETNAME].format(lang=_MAPPING[lang2])))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": [lang1_txt_path, lang2_txt_path]},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        lang1 = self.config.name.split("_")[2]
        lang2 = self.config.name.split("_")[3]

        texts1 = []
        texts2 = []
        texts1 = open(filepath[0], "r").readlines()
        texts2 = open(filepath[1], "r").readlines()

        if self.config.schema == "source" or self.config.schema == "seacrowd_t2t":
            idx = 0
            for line1, line2 in zip(texts1, texts2):
                ex = {
                    "id": str(idx),
                    "text_1": line1,
                    "text_2": line2,
                    "text_1_name": lang1,
                    "text_2_name": lang2,
                }
                yield idx, ex
                idx += 1
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
