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
The dataset is split into two:
1. Monolingual (ends with .txt) [Indonesian, Javanese]
2. Bilingual (ends with .tsv) [Indonesian-Javanese, Indonesian-Balinese, Indonesian-Minangkabau, Indonesian-Sundanese]
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{susanto2023replicable,
      title={Replicable Benchmarking of Neural Machine Translation (NMT) on Low-Resource Local Languages in Indonesia},
      author={Lucky Susanto and Ryandito Diandaru and Adila Krisnadhi and Ayu Purwarianti and Derry Wijaya},
      year={2023},
      eprint={2311.00998},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""
_DATASETNAME = "indonesiannmt"

_DESCRIPTION = """\
This dataset is used on the paper "Replicable Benchmarking of Neural Machine Translation (NMT) on Low-Resource Local Languages in Indonesia". This repository contains two types of data:
1. Monolingual (*.txt) [Indonesian, Javanese]
2. Bilingual (*.tsv) [Indonesian-Javanese, Indonesian-Balinese, Indonesian-Minangkabau, Indonesian-Sundanese]
Only the Bilingual dataset is available for this dataloader
"""

_HOMEPAGE = "https://huggingface.co/datasets/Exqrch/IndonesianNMT"

_LANGUAGES = ["ind", "jav", "ban", "min", "sun"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_LOCAL = False

_URLS = {
    "ind_jav": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/id-jv.tsv?download=true",
    "ind_sun": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/id-su.tsv?download=true",
    "ind_ban": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/id-ban.tsv?download=true",
    "ind_min": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/id-min.tsv?download=true",
    "ind": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/bt-id-jv.id.txt?download=true",
    "jav": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/bt-id-jv.jv.txt?download=true",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION, Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


def seacrowd_config_constructor(modifier, schema, version):
    return SEACrowdConfig(
        name=f"indonesiannmt_{modifier}_{schema}",
        version=version,
        description=f"indonesiannmt_{modifier} {schema} schema",
        schema=f"{schema}",
        subset_id="indonesiannmt",
    )


class IndonesianNMT(datasets.GeneratorBasedBuilder):
    """IndonesianNMT consists of 4 parallel datasets and 2 monolingual datasets,
    all obtained synthetically from either gpt-3.5-turbo or text-davinci-003"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = (
        [seacrowd_config_constructor(x, "source", _SOURCE_VERSION) for x in ["ind", "jav"]]
        + [seacrowd_config_constructor(x, "seacrowd_ssp", _SOURCE_VERSION) for x in ["ind", "jav"]]
        + [seacrowd_config_constructor(x, "source", _SOURCE_VERSION) for x in ["ind_jav", "ind_min", "ind_sun", "ind_ban"]]
        + [seacrowd_config_constructor(x, "seacrowd_t2t", _SEACROWD_VERSION) for x in ["ind_jav", "ind_min", "ind_sun", "ind_ban"]]
    )

    DEFAULT_CONFIG_NAME = "indonesiannmt_ind_source"

    def is_mono(self):
        if self.config.schema == "seacrowd_ssp":
            return True
        if "source" in self.config.schema:
            if len(self.config.name.split("_")) == 3:
                return True
        return False

    def _info(self) -> datasets.DatasetInfo:
        # ex mono: indonesiannmt_ind_source OR indonesiannmt_ind_seacrowd_ssp
        # ex para: indonesiannmt_ind_jav_source OR indonesiannmt_ind_jav_seacrowd_t2t
        is_mono = self.is_mono()
        if is_mono and self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "lang_1": datasets.Value("string"),
                    "lang_2": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text_to_text.features
        elif self.config.schema == "seacrowd_ssp":
            features = schemas.self_supervised_pretraining.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # ex mono: indonesiannmt_ind_source OR indonesiannmt_ind_seacrowd_ssp
        # ex para: indonesiannmt_ind_jav_source OR indonesiannmt_ind_jav_seacrowd_t2t
        is_mono = self.is_mono()
        if "seacrowd_ssp" in self.config.schema or is_mono:
            lang = self.config.name.split("_")[1]
            path = dl_manager.download_and_extract(_URLS[lang])
        else:
            target = "_".join(self.config.name.split("_")[1:3])
            url = _URLS[target]
            path = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        is_mono = self.is_mono()
        STR_TO_ISO = {"Indonesian": "ind", "Javanese": "jav", "Minangkabau": "min", "Sundanese": "sun", "Balinese": "ban"}

        with open(filepath, encoding="utf-8") as f:
            flag = True
            if "seacrowd_ssp" in self.config.schema or is_mono:
                for counter, row in enumerate(f):
                    if row.strip != "":
                        yield (
                            counter,
                            {
                                "id": str(counter),
                                "text": row.strip(),
                            },
                        )
            elif self.config.schema == "source":
                for counter, row in enumerate(f):
                    if flag:
                        src, tgt = row.split("\t")
                        tgt = tgt.strip()
                        flag = False
                    else:
                        if row.strip() != "":
                            yield (
                                counter,
                                {
                                    "id": str(counter),
                                    "text_1": row.split("\t")[0].strip(),
                                    "text_2": row.split("\t")[1].strip(),
                                    "lang_1": STR_TO_ISO[src],
                                    "lang_2": STR_TO_ISO[tgt],
                                },
                            )
            elif self.config.schema == "seacrowd_t2t":
                for counter, row in enumerate(f):
                    if flag:
                        src, tgt = row.split("\t")
                        tgt = tgt.strip()
                        flag = False
                    else:
                        if row.strip() != "":
                            yield (
                                counter,
                                {
                                    "id": str(counter),
                                    "text_1": row.split("\t")[0].strip(),
                                    "text_2": row.split("\t")[1].strip(),
                                    "text_1_name": STR_TO_ISO[src],
                                    "text_2_name": STR_TO_ISO[tgt],
                                },
                            )
