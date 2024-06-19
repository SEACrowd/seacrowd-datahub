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
https://github.com/ir-nlp-csui/indoler/tree/main
The dataset contains 993 annotated court decission document. 
The document was taken from Decision of the Supreme Court of Indonesia. 
The documents have also been tokenized and cleaned
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@INPROCEEDINGS{9263157,
  author={Nuranti, Eka Qadri and Yulianti, Evi},
  booktitle={2020 International Conference on Advanced Computer Science and Information Systems (ICACSIS)}, 
  title={Legal Entity Recognition in Indonesian Court Decision Documents Using Bi-LSTM and CRF Approaches}, 
  year={2020},
  volume={},
  number={},
  pages={429-434},
  keywords={Xenon;6G mobile communication;legal processing;legal entity recognition;legal document;name entity recognition;ner;bi-lstm;lstm;crf},
  doi={10.1109/ICACSIS51025.2020.9263157}}
"""

_DATASETNAME = "indoler"

_DESCRIPTION = """\
https://github.com/ir-nlp-csui/indoler/tree/main
The data can be used for NER Task in legal documents.
The dataset contains 993 annotated court decission document. 
The document was taken from Decision of the Supreme Court of Indonesia. 
The documents have also been tokenized and cleaned
"""

_HOMEPAGE = "https://github.com/ir-nlp-csui/indoler/tree/main"

_LANGUAGES = ['ind']  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.UNKNOWN.value 

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "test_idx": "https://raw.githubusercontent.com/ir-nlp-csui/indoler/main/test.ids.csv",
        "train_idx": "https://raw.githubusercontent.com/ir-nlp-csui/indoler/main/train.ids.csv",
        "valid_idx": "https://raw.githubusercontent.com/ir-nlp-csui/indoler/main/val.ids.csv",
        "full_data": "https://raw.githubusercontent.com/ir-nlp-csui/indoler/main/data.json"
    },
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]  

_SOURCE_VERSION = "2.0.0"

_SEACROWD_VERSION = "2024.06.20"



class IndoLer(datasets.GeneratorBasedBuilder):
    """https://github.com/ir-nlp-csui/indoler/tree/main
The data can be used for NER Task in legal documents
The dataset contains 993 annotated court decission document. 
The document was taken from Decision of the Supreme Court of Indonesia. 
The documents have also been tokenized and cleaned"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="indoler_source",
            version=SOURCE_VERSION,
            description="indoler source schema",
            schema="source",
            subset_id="indoler",
        ),
        SEACrowdConfig(
            name="indoler_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="indoler SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id="indoler",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indoler_source"

    def _info(self) -> datasets.DatasetInfo:

        NAMED_ENTITIES = ['O', 'B-Jenis Amar', 'B-Jenis Dakwaan', 'B-Jenis Perkara', 'B-Melanggar UU (Dakwaan)', 
                    'B-Melanggar UU (Pertimbangan Hukum)', 'B-Melanggar UU (Tuntutan)', 'B-Nama Hakim Anggota', 'B-Nama Hakim Ketua', 
                    'B-Nama Jaksa', 'B-Nama Panitera', 'B-Nama Pengacara', 'B-Nama Pengadilan', 
                    'B-Nama Saksi', 'B-Nama Terdakwa', 'B-Nomor Putusan', 'B-Putusan Hukuman', 
                    'B-Tanggal Kejadian', 'B-Tanggal Putusan', 'B-Tingkat Kasus', 'B-Tuntutan Hukuman', 
                    'I-Jenis Amar', 'I-Jenis Dakwaan', 'I-Jenis Perkara', 'I-Melanggar UU (Dakwaan)', 
                    'I-Melanggar UU (Pertimbangan Hukum)', 'I-Melanggar UU (Tuntutan)', 'I-Nama Hakim Anggota', 'I-Nama Hakim Ketua', 
                    'I-Nama Jaksa', 'I-Nama Panitera', 'I-Nama Pengacara', 'I-Nama Pengadilan', 
                    'I-Nama Saksi', 'I-Nama Terdakwa', 'I-Nomor Putusan', 'I-Putusan Hukuman', 
                    'I-Tanggal Kejadian', 'I-Tanggal Putusan', 'I-Tingkat Kasus', 'I-Tuntutan Hukuman']

        if self.config.schema == "source":
            features = datasets.Features({
                "id": datasets.Value("string"),
                "owner": datasets.Value("string"),
                "lawyer": datasets.ClassLabel(names=[False, True]),
                "verdict": datasets.ClassLabel(names=["guilty", "bebas", "lepas"]),
                "indictment": datasets.ClassLabel(names=["NA", "tunggal", "subsider", "komul", "alternatif", "kombinasi", "gabungan"]),
                "text-tags": datasets.Sequence(datasets.ClassLabel(names=NAMED_ENTITIES)),
                "text": datasets.Sequence(datasets.Value("string")),
            })
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label.features(NAMED_ENTITIES)
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        test_path = dl_manager.download_and_extract(urls['test_idx'])
        train_path = dl_manager.download_and_extract(urls['train_idx'])
        valid_path = dl_manager.download_and_extract(urls['valid_idx'])
        data_path = dl_manager.download_and_extract(urls['full_data'])
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path,
                    "idx_path": train_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_path,
                    "idx_path": test_path,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_path,
                    "idx_path": valid_path,
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, idx_path: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        split_idxs = []
        with open(idx_path, 'r', encoding="utf-8") as indexes:
            for index in indexes.readlines():
                split_idxs.append(int(index))
        with open(filepath, 'r', encoding="utf-8") as file:
            contents = json.load(file)
            counter = 0
            for content in contents:
                if int(content['id']) in split_idxs:
                    if self.config.schema == "source":
                        if content['indictment'] not in ["NA", "tunggal", "subsider", "komul", "alternatif", "kombinasi", "gabungan"]:
                            content['indictment'] = "NA"
                        yield(
                            counter,
                            {
                                "id"        : content['id'],
                                "owner"     : content['owner'],
                                "lawyer"    : content['lawyer'],
                                "verdict"   : content['verdict'],
                                "indictment": content['indictment'],
                                "text-tags" : content['text-tags'],
                                "text"      : content['text'],
                            }
                        )        
                        counter += 1
                    elif self.config.schema == "seacrowd_seq_label":
                        yield(
                            counter,
                            {
                                "id": content['id'],
                                "tokens": content['text'],
                                "labels": content['text-tags'],
                            }
                        )
                        counter += 1
