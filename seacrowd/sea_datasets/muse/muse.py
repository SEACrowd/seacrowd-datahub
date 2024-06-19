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
Contains 110 large-scale ground-truth bilingual dictionaries created and released by Meta using an internal translation tool.
The dictionaries account for polysemy. The data comprises of a train and test split of 5000 and 1500 unique source words, as well as a larger set of up to 100k pairs.
It comprises of Europeans languages in every direction, and SEA languages to and from English.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{lample2018word,
  title={Word translation without parallel data},
  author={Lample, Guillaume and Conneau, Alexis and Ranzato, Marc'Aurelio and Denoyer, Ludovic and J{\'e}gou, Herv{\'e}},
  booktitle={International Conference on Learning Representations},
  year={2018}}
}
"""

_DATASETNAME = "muse"

_DESCRIPTION = """\
Contains 110 large-scale ground-truth bilingual dictionaries created and released by Meta using an internal translation tool.
The dictionaries account for polysemy. The data comprises of a train and test split of 5000 and 1500 unique source words, as well as a larger set of up to 100k pairs.
It comprises of Europeans languages in every direction, and SEA languages to and from English.
"""

_HOMEPAGE = "https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries"

_LANGUAGES = ["tgl", "ind", "zlm", "tha", "vie"]

_LICENSE = Licenses.CC_BY_NC_ND_4_0.value

_LOCAL = False

_TRAIN_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/arrival/dictionaries/{src}-{tgt}.0-5000.txt"
_TEST_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/arrival/dictionaries/{src}-{tgt}.5000-6500.txt"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

configs = {
    "tgl": ["eng"],
    "ind": ["eng"],
    "zlm": ["eng"],
    "tha": ["eng"],
    "vie": ["eng"],
    "eng": ["tha", "vie", "tgl", "zlm", "ind"],
}

langid_dict = {
    "eng": "en",
    "tgl": "tl",
    "ind": "id",
    "zlm": "ms",
    "tha": "th",
    "vie": "vi",
}


class MUSEDataset(datasets.GeneratorBasedBuilder):
    """Large-scale ground-truth bilingual dictionaries"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = (
        [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} source schema",
                schema="source",
                subset_id=f"{_DATASETNAME}_tgl_eng",
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_t2t",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} SEACrowd schema",
                schema="seacrowd_t2t",
                subset_id=f"{_DATASETNAME}_tgl_eng",
            ),
        ]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{src_lang}_{tgt_lang}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} source schema",
                schema="source",
                subset_id=f"{_DATASETNAME}_{src_lang}_{tgt_lang}",
            )
            for src_lang in configs
            for tgt_lang in configs[src_lang]
        ]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{src_lang}_{tgt_lang}_seacrowd_t2t",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} SEACrowd schema",
                schema="seacrowd_t2t",
                subset_id=f"{_DATASETNAME}_{src_lang}_{tgt_lang}",
            )
            for src_lang in configs
            for tgt_lang in configs[src_lang]
        ]
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

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

        _, src_lang, tgt_lang = self.config.subset_id.split("_")
        train_url = _TRAIN_URL_TEMPLATE.format(src=langid_dict[src_lang], tgt=langid_dict[tgt_lang])
        test_url = _TEST_URL_TEMPLATE.format(src=langid_dict[src_lang], tgt=langid_dict[tgt_lang])

        train_file = dl_manager.download_and_extract(train_url)
        test_file = dl_manager.download_and_extract(test_url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "filepath": train_file,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "filepath": test_file,
                },
            ),
        ]

    def _generate_examples(self, src_lang: str, tgt_lang: str, filepath: Path) -> Tuple[int, Dict]:
        if self.config.schema == "source":
            for row_id, line in enumerate(open(filepath)):
                src_text, tgt_text = line.strip().split("\t")
                yield row_id, {"id": row_id, "src_text": src_text, "tgt_text": tgt_text}

        elif self.config.schema == "seacrowd_t2t":
            for row_id, line in enumerate(open(filepath)):
                src_text, tgt_text = line.strip().split("\t")
                yield row_id, {
                    "id": row_id,
                    "text_1": src_text,
                    "text_2": tgt_text,
                    "text_1_name": src_lang,
                    "text_2_name": tgt_lang,
                }
