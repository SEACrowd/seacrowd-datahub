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

_HOMEPAGE = "https://opus.nlpl.eu/KDE4/corpus/version/KDE4"

_LANGUAGES = ["ind", "khm", "zlm", "tha", "vie"]

_LICENSE = Licenses.UNKNOWN.value
_LOCAL = False

_URL_TEMPLATE = "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/{src}-{tgt}.txt.zip"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "2.0.0"
_SEACROWD_VERSION = "2024.06.20"

kde4_language_codes = {
    "aar": "aa",
    "abk": "ab",
    "ave": "ae",
    "afr": "af",
    "aka": "ak",
    "amh": "am",
    "arg": "an",
    "ara": "ar",
    "asm": "as",
    "ava": "av",
    "aym": "ay",
    "aze": "az",
    "bak": "ba",
    "bel": "be",
    "bul": "bg",
    "bis": "bi",
    "bam": "bm",
    "ben": "bn",
    "bod": "bo",
    "bre": "br",
    "bos": "bs",
    "cat": "ca",
    "che": "ce",
    "cha": "ch",
    "cos": "co",
    "cre": "cr",
    "ces": "cs",
    "chu": "cu",
    "chv": "cv",
    "cym": "cy",
    "dan": "da",
    "deu": "de",
    "div": "dv",
    "dzo": "dz",
    "ewe": "ee",
    "ell": "el",
    "eng": "en",
    "epo": "eo",
    "spa": "es",
    "est": "et",
    "eus": "eu",
    "fas": "fa",
    "ful": "ff",
    "fin": "fi",
    "fij": "fj",
    "fao": "fo",
    "fra": "fr",
    "fry": "fy",
    "gle": "ga",
    "gla": "gd",
    "glg": "gl",
    "grn": "gn",
    "guj": "gu",
    "glv": "gv",
    "hau": "ha",
    "heb": "he",
    "hin": "hi",
    "hmo": "ho",
    "hrv": "hr",
    "hat": "ht",
    "hun": "hu",
    "hye": "hy",
    "her": "hz",
    "ina": "ia",
    "ind": "id",
    "ile": "ie",
    "ibo": "ig",
    "iii": "ii",
    "ipk": "ik",
    "ido": "io",
    "isl": "is",
    "ita": "it",
    "iku": "iu",
    "jpn": "ja",
    "jav": "jv",
    "kat": "ka",
    "kon": "kg",
    "kik": "ki",
    "kua": "kj",
    "kaz": "kk",
    "kal": "kl",
    "khm": "km",
    "kan": "kn",
    "kor": "ko",
    "kau": "kr",
    "kas": "ks",
    "kur": "ku",
    "kom": "kv",
    "cor": "kw",
    "kir": "ky",
    "lat": "la",
    "ltz": "lb",
    "lug": "lg",
    "lim": "li",
    "lin": "ln",
    "lao": "lo",
    "lit": "lt",
    "lub": "lu",
    "lav": "lv",
    "mlg": "mg",
    "mah": "mh",
    "mri": "mi",
    "mkd": "mk",
    "mal": "ml",
    "mon": "mn",
    "mar": "mr",
    "msa": "ms",
    "mlt": "mt",
    "mya": "my",
    "nau": "na",
    "nob": "nb",
    "nde": "nd",
    "nep": "ne",
    "ndo": "ng",
    "nld": "nl",
    "nno": "nn",
    "nor": "no",
    "nbl": "nr",
    "nav": "nv",
    "nya": "ny",
    "oci": "oc",
    "oji": "oj",
    "orm": "om",
    "ori": "or",
    "oss": "os",
    "pan": "pa",
    "pli": "pi",
    "pol": "pl",
    "pus": "ps",
    "por": "pt",
    "que": "qu",
    "roh": "rm",
    "run": "rn",
    "ron": "ro",
    "rus": "ru",
    "kin": "rw",
    "san": "sa",
    "srd": "sc",
    "snd": "sd",
    "sme": "se",
    "sag": "sg",
    "hbs": "sh",
    "sin": "si",
    "slk": "sk",
    "slv": "sl",
    "smo": "sm",
    "sna": "sn",
    "som": "so",
    "sqi": "sq",
    "srp": "sr",
    "ssw": "ss",
    "sot": "st",
    "sun": "su",
    "swe": "sv",
    "swa": "sw",
    "tam": "ta",
    "tel": "te",
    "tgk": "tg",
    "tha": "th",
    "tir": "ti",
    "tuk": "tk",
    "tgl": "tl",
    "tsn": "tn",
    "ton": "to",
    "tur": "tr",
    "tso": "ts",
    "tat": "tt",
    "twi": "tw",
    "tah": "ty",
    "uig": "ug",
    "ukr": "uk",
    "urd": "ur",
    "uzb": "uz",
    "ven": "ve",
    "vie": "vi",
    "vol": "vo",
    "wln": "wa",
    "wol": "wo",
    "xho": "xh",
    "yid": "yi",
    "yor": "yo",
    "zha": "za",
    "zho": "zh",
    "zul": "zu",
    "nds": "nds",
    "mai": "mai",
    "nso": "nso",
    "ast": "ast",
    "crh": "crh",
    "csb": "csb",
    "hne": "hne",
    "hsb": "hsb",
}

configs = {
    "afr": ["msa", "tha", "khm", "ind", "vie"],
    "ara": ["msa", "tha", "khm", "ind", "vie"],
    "asm": ["msa", "tha", "khm", "ind", "vie"],
    "ast": ["msa", "tha", "khm", "ind", "vie"],
    "bel": ["msa", "tha", "khm", "ind", "vie"],
    "bul": ["msa", "tha", "khm", "ind", "vie"],
    "ben": ["msa", "tha", "khm", "ind", "vie"],
    "bre": ["msa", "tha", "khm", "ind", "vie"],
    "cat": ["msa", "tha", "khm", "ind", "vie"],
    "crh": ["msa", "tha", "khm", "ind", "vie"],
    "ces": ["msa", "tha", "khm", "ind", "vie"],
    "csb": ["msa", "tha", "khm", "ind", "vie"],
    "cym": ["msa", "tha", "khm", "ind", "vie"],
    "dan": ["msa", "tha", "khm", "ind", "vie"],
    "deu": ["msa", "tha", "khm", "ind", "vie"],
    "ell": ["msa", "tha", "khm", "ind", "vie"],
    "eng": ["msa", "tha", "khm", "ind", "vie"],
    "epo": ["msa", "tha", "khm", "ind", "vie"],
    "spa": ["msa", "tha", "khm", "ind", "vie"],
    "est": ["msa", "tha", "khm", "ind", "vie"],
    "eus": ["msa", "tha", "khm", "ind", "vie"],
    "fas": ["msa", "tha", "khm", "ind", "vie"],
    "fin": ["msa", "tha", "khm", "ind", "vie"],
    "fra": ["msa", "tha", "khm", "ind", "vie"],
    "fry": ["msa", "tha", "khm", "ind", "vie"],
    "gle": ["msa", "tha", "khm", "ind", "vie"],
    "glg": ["msa", "tha", "khm", "ind", "vie"],
    "guj": ["msa", "tha", "khm", "ind", "vie"],
    "hau": ["msa", "tha", "khm", "ind", "vie"],
    "heb": ["msa", "tha", "khm", "ind", "vie"],
    "hin": ["msa", "tha", "khm", "ind", "vie"],
    "hne": ["msa", "tha", "khm", "ind", "vie"],
    "hrv": ["msa", "tha", "khm", "ind", "vie"],
    "hsb": ["msa", "tha", "khm", "ind", "vie"],
    "hun": ["msa", "tha", "khm", "ind", "vie"],
    "hye": ["msa", "tha", "khm", "ind", "vie"],
    "ind": [
        "kan",
        "pus",
        "msa",
        "slv",
        "tur",
        "rus",
        "nld",
        "mkd",
        "jpn",
        "ori",
        "nep",
        "xho",
        "nds",
        "lav",
        "ukr",
        "vie",
        "mai",
        "tam",
        "ltz",
        "isl",
        "uzb",
        "sme",
        "lit",
        "tgk",
        "kat",
        "mal",
        "srp",
        "wln",
        "por",
        "oci",
        "kur",
        "mar",
        "sin",
        "slk",
        "kor",
        "kaz",
        "ron",
        "nno",
        "tha",
        "khm",
        "tel",
        "ita",
        "pol",
        "swe",
        "pan",
        "nob",
    ],
    "isl": ["khm", "msa", "tha", "vie"],
    "ita": ["khm", "msa", "tha", "vie"],
    "jpn": ["khm", "msa", "tha", "vie"],
    "kat": ["khm", "msa", "tha", "vie"],
    "kaz": ["khm", "msa", "tha", "vie"],
    "khm": [
        "kan",
        "pus",
        "msa",
        "slv",
        "tur",
        "rus",
        "kin",
        "nld",
        "mkd",
        "ori",
        "xho",
        "nso",
        "nep",
        "nds",
        "lav",
        "ukr",
        "vie",
        "mai",
        "tam",
        "ltz",
        "uzb",
        "sme",
        "lit",
        "tgk",
        "mlt",
        "mal",
        "srp",
        "wln",
        "por",
        "oci",
        "kur",
        "mar",
        "sin",
        "slk",
        "kor",
        "ron",
        "nno",
        "tha",
        "tel",
        "pol",
        "swe",
        "pan",
        "nob",
    ],
    "kan": ["msa", "tha", "vie"],
    "kor": ["msa", "tha", "vie"],
    "kur": ["msa", "tha", "vie"],
    "ltz": ["msa", "tha", "vie"],
    "lit": ["msa", "tha", "vie"],
    "lav": ["msa", "tha", "vie"],
    "mai": ["msa", "tha", "vie"],
    "mkd": ["msa", "tha", "vie"],
    "mal": ["msa", "tha", "vie"],
    "mar": ["msa", "tha", "vie"],
    "msa": [
        "pus",
        "slv",
        "tur",
        "rus",
        "kin",
        "nld",
        "ori",
        "xho",
        "nso",
        "nep",
        "nds",
        "ukr",
        "vie",
        "tam",
        "uzb",
        "sme",
        "tgk",
        "mlt",
        "srp",
        "wln",
        "por",
        "oci",
        "slk",
        "sin",
        "ron",
        "nno",
        "tha",
        "tel",
        "pol",
        "swe",
        "pan",
        "nob",
    ],
    "mlt": ["tha"],
    "nob": ["tha", "vie"],
    "nds": ["tha", "vie"],
    "nep": ["tha", "vie"],
    "nld": ["tha", "vie"],
    "nno": ["tha", "vie"],
    "oci": ["tha", "vie"],
    "ori": ["tha", "vie"],
    "pan": ["tha", "vie"],
    "pol": ["tha", "vie"],
    "pus": ["tha", "vie"],
    "por": ["tha", "vie"],
    "ron": ["tha", "vie"],
    "rus": ["tha", "vie"],
    "kin": ["tha"],
    "sme": ["tha", "vie"],
    "sin": ["tha", "vie"],
    "slk": ["tha", "vie"],
    "slv": ["tha", "vie"],
    "srp": ["tha", "vie"],
    "swe": ["tha", "vie"],
    "tam": ["tha", "vie"],
    "tel": ["tha", "vie"],
    "tgk": ["tha", "vie"],
    "tha": ["wln", "tur", "uzb", "xho", "vie", "ukr"],
    "tur": ["vie"],
    "ukr": ["vie"],
    "uzb": ["vie"],
    "vie": ["wln", "xho"],
}


class KDE4Dataset(datasets.GeneratorBasedBuilder):
    """A parallel corpus of KDE4 localization files. The corpus is available in 92 languages in total, with 4099 bitexts."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = (
        [
            SEACrowdConfig(
                name="kde4_source",
                version=datasets.Version(_SOURCE_VERSION),
                description="kde4 source schema for afr to msa",
                schema="source",
                subset_id="afr_msa",
            ),
            SEACrowdConfig(
                name="kde4_seacrowd_t2t",
                version=datasets.Version(_SOURCE_VERSION),
                description="kde4 seacrowd_t2t schema for afr to msa",
                schema="seacrowd_t2t",
                subset_id="afr_msa",
            ),
        ]
        + [
            SEACrowdConfig(
                name=f"kde4_{src_lang}_{tgt_lang}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"kde4 source schema for {src_lang} to {tgt_lang}",
                schema="source",
                subset_id=f"{src_lang}_{tgt_lang}",
            )
            for src_lang in configs
            for tgt_lang in configs[src_lang]
        ]
        + [
            SEACrowdConfig(
                name=f"kde4_{src_lang}_{tgt_lang}_seacrowd_t2t",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"kde4 seacrowd_t2t schema for {src_lang} to {tgt_lang}",
                schema="seacrowd_t2t",
                subset_id=f"{src_lang}_{tgt_lang}",
            )
            for src_lang in configs
            for tgt_lang in configs[src_lang]
        ]
    )

    DEFAULT_CONFIG_NAME = "kde4_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "src_text": datasets.Value("string"),
                    "tgt_text": datasets.Value("string"),
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

        src_lang, tgt_lang = self.config.subset_id.split("_")
        kde4_src_lang, kde4_tgt_lang = kde4_language_codes[src_lang], kde4_language_codes[tgt_lang]

        url = _URL_TEMPLATE.format(src=kde4_src_lang, tgt=kde4_tgt_lang)
        data_dir = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "src_filepath": os.path.join(data_dir, f"KDE4.{kde4_src_lang}-{kde4_tgt_lang}.{kde4_src_lang}"),
                    "tgt_filepath": os.path.join(data_dir, f"KDE4.{kde4_src_lang}-{kde4_tgt_lang}.{kde4_tgt_lang}"),
                },
            )
        ]

    def _generate_examples(self, src_lang: str, tgt_lang: str, src_filepath: Path, tgt_filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            for row_id, (src_text, tgt_text) in enumerate(zip(open(src_filepath), open(tgt_filepath))):
                yield row_id, {
                    "id": row_id,
                    "src_text": src_text.strip(),
                    "tgt_text": tgt_text.strip(),
                }

        elif self.config.schema == "seacrowd_t2t":
            for row_id, (src_text, tgt_text) in enumerate(zip(open(src_filepath), open(tgt_filepath))):
                yield row_id, {
                    "id": row_id,
                    "text_1": src_text.strip(),
                    "text_2": tgt_text.strip(),
                    "text_1_name": src_lang,
                    "text_2_name": tgt_lang,
                }
