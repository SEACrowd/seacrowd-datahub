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
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

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

"""

_HOMEPAGE = "https://huggingface.co/datasets/Exqrch/IndonesianNMT"

_LANGUAGES = ['ind', 'jav', 'ban', 'min', 'sun']  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value 

_LOCAL = False

_URLS = {
    "parallel_ind_jav": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/id-jv.tsv?download=true",
    "parallel_ind_sun": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/id-su.tsv?download=true",
    "parallel_ind_ban": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/id-ban.tsv?download=true",
    "parallel_ind_min": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/id-min.tsv?download=true",
    "mono_ind": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/bt-id-jv.id.txt?download=true",
    "mono_jav": "https://huggingface.co/datasets/Exqrch/IndonesianNMT/resolve/main/bt-id-jv.jv.txt?download=true",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]  

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

def seacrowd_config_constructor(modifier, schema, version):
    return SEACrowdConfig(
            name=f"indonesiannmt-{modifier}_{schema}",
            version=version,
            description=f"indonesiannmt-{modifier} {schema} schema",
            schema=f"{schema}",
            subset_id="indonesiannmt",
        )

class IndonesianNMT(datasets.GeneratorBasedBuilder):
    """IndonesianNMT consists of 4 parallel datasets and 2 monolingual datasets, 
    all obtained synthetically from either gpt-3.5-turbo or text-davinci-003"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = (
        [SEACrowdConfig(
            name=f"indonesiannmt_source",
            version=_SOURCE_VERSION,
            description=f"indonesiannmt source schema",
            schema=f"source",
            subset_id="indonesiannmt",
        )] + 
        [SEACrowdConfig(
            name=f"indonesiannmt_seacrowd_t2t",
            version=_SEACROWD_VERSION,
            description=f"indonesiannmt seacrowd_t2t schema",
            schema=f"seacrowd_t2t",
            subset_id="indonesiannmt",
        )] +
        [seacrowd_config_constructor(x, 'source', _SOURCE_VERSION) for x in ['mono_ind', 'mono_jav', 'parallel_ind_jav', 'parallel_ind_min', 'parallel_ind_sun', 'parallel_ind_ban']]
        + [seacrowd_config_constructor(x, 'seacrowd_t2t', _SEACROWD_VERSION) for x in ['mono_ind', 'mono_jav', 'parallel_ind_jav', 'parallel_ind_min', 'parallel_ind_sun', 'parallel_ind_ban']])

    DEFAULT_CONFIG_NAME = "indonesiannmt_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
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

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # ex mono: indonesiannmt-mono_ind_source OR indonesiannmt-mono_ind_seacrowd_t2t
        # ex para: indonesiannmt-parallel_ind_jav_source OR indonesiannmt-parallel_ind_jav_seacrowd_t2t
        if self.config.name in ['indonesiannmt_source', 'indonesiannmt_seacrowd_t2t']:
            path = dl_manager.download_and_extract(_URLS['parallel_ind_jav'])
        else:         
            mono_or_para = self.config.name.split('-')[1].split('_')[0]
            target = ""
            if mono_or_para == "mono":
                target = '_'.join(self.config.name.split('-')[1].split('_')[:2])
            elif mono_or_para == "parallel":
                target = '_'.join(self.config.name.split('-')[1].split('_')[:3])
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
        with open(filepath, encoding="utf-8") as f:
            flag = True  
            mono = True
            if self.config.schema == "source":
                for counter, row in enumerate(f):
                    if mono and not row.startswith('Indonesian'):
                        if row.strip() != "":
                            yield (
                                counter,
                                {
                                    "id": str(counter),
                                    "text_1": row.strip(),
                                    "text_2": None,
                                    "lang_1": None,
                                    "lang_2": None,
                                },
                            )
                    else:
                        mono = False 
                        if flag:
                            src, tgt = row.split('\t')
                            flag = False
                        else:
                            if row.strip() != "":
                                yield(
                                    counter,
                                    {
                                        "id": str(counter),
                                        "text_1": row.split('\t')[0].strip(),
                                        "text_2": row.split('\t')[1].strip(),
                                        "lang_1": src,
                                        "lang_2": tgt,
                                    }
                                )
            else:
                for counter, row in enumerate(f):
                    if mono and not row.startswith('Indonesian'):
                        if row.strip() != "":
                            yield (
                                counter,
                                {
                                    "id": str(counter),
                                    "text_1": row.strip(),
                                    "text_2": None,
                                    "text_1_name": None,
                                    "text_2_name": None,
                                },
                            )
                    else:
                        mono = False 
                        if flag:
                            src, tgt = row.split('\t')
                            flag = False
                        else:
                            if row.strip() != "":
                                yield(
                                    counter,
                                    {
                                        "id": str(counter),
                                        "text_1": row.split('\t')[0].strip(),
                                        "text_2": row.split('\t')[1].strip(),
                                        "text_1_name": src,
                                        "text_2_name": tgt,
                                    }
                                )
