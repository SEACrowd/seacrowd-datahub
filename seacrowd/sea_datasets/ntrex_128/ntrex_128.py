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

_LANGUAGES = ["mya", "fil", "ind", "khm", "lao", "zlm", "tha", "vie", "hmv"] 

_LICENSE = Licenses.CC_BY_SA_4_0.value  

_LOCAL = False

_MAPPING = {"mya": "mya", "fil": "fil", "ind": "ind", "khm": "khm", "lao": "lao", "zlm": "msa", "tha": "tha", "vie": "vie", "hmv": "hmn"}
_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-ref.{lang}.txt",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION] 

_SOURCE_VERSION = "11.24.2022"

_SEACROWD_VERSION = "1.0.0"


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
        for subset2 in _LANGUAGES
        for subset1 in _LANGUAGES
        if subset1 != subset2
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset1}_{subset2}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} {subset1}2{subset2} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_{subset1}_{subset2}",
        )
        for subset2 in _LANGUAGES
        for subset1 in _LANGUAGES
        if subset1 != subset2
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
                name=datasets.Split.TRAIN,
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
