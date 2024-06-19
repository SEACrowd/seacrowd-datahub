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

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.common_parser import load_conll_data
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{abdillah2023pengenalan,
  title={Pengenalan Entitas Biomedis dalam Teks Konsultasi Kesehatan Online Berbahasa Indonesia Berbasis Arsitektur Transformers},
  author={Abdillah, Abid Famasya and Purwitasari, Diana and Juanita, Safitri and Purnomo, Mauridhi Hery},
  year={2023},
  month=feb,
  journal={Jurnal Teknologi Informasi dan Ilmu Komputer},
  volume={10},
  number={1},
  pages={131--140}
}
"""

_DATASETNAME = "bioner_id"

_DESCRIPTION = """\
This dataset taken from online health consultation platform Alodokter.com which has been annotated by two medical doctors. Data were annotated using IOB in CoNLL format.

Dataset contains 2600 medical answers by doctors from 2017-2020. Two medical experts were assigned to annotate the data into two entity types: DISORDERS and ANATOMY.
The topics of answers are: diarrhea, HIV-AIDS, nephrolithiasis and TBC, which marked as high-risk dataset from WHO.
"""

_HOMEPAGE = "https://huggingface.co/datasets/abid/indonesia-bioner-dataset"

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.BSD_3_CLAUSE_CLEAR.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {k: f"https://huggingface.co/datasets/abid/indonesia-bioner-dataset/raw/main/{k}.conll" for k in ["train", "valid", "test"]},
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class BioNERIdDataset(datasets.GeneratorBasedBuilder):
    """2600 conversations of patioent and medical doctors between 2017-2020.
    Two medical annotated the data into two entity types: DISORDERS and ANATOMY"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    label_classes = ["B-ANAT", "B-DISO", "I-ANAT", "I-DISO", "O"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sentence": [datasets.Value("string")],
                    "label": [datasets.Value("string")],
                }
            )

        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(self.label_classes)

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
        data_paths = dl_manager.download(urls)

        # def _assert_data(msg):
        #     cur_data = list(map(
        #         lambda d: d.split(" "),
        #         open(fp, "r", encoding="utf8").readlines()
        #     ))
        #     assert {1, 4} == set(map(len, cur_data)), msg    # length of 4 is due to uncommon delimiter of " _ _ "
        #     assert {('_', '_')} == set(map(lambda _: (_[1], _[2]), filter(lambda _: len(_) == 4, cur_data))), msg

        # Convert to tab-seperated value
        for subset in ["train", "valid", "test"]:
            fp = data_paths[subset]
            # _assert_data(f"Invalid file for subset '{subset}'")
            data = open(fp, "r", encoding="utf8").read()
            # data_paths[subset] = f"{fp}.tsv"
            open(data_paths[subset], "w", encoding="utf8").write(data.replace(" _ _ ", "\t"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_paths["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_paths["test"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_paths["valid"]},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = load_conll_data(filepath)

        if self.config.schema == "source":
            for key, ex in enumerate(data):
                yield key, ex

        elif self.config.schema == "seacrowd_seq_label":
            for key, ex in enumerate(data):
                yield key, {
                    "id": str(key),
                    "tokens": ex["sentence"],
                    "labels": ex["label"],
                }
