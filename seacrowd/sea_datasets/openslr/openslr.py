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

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{kjartansson18_sltu,
  author={Oddur Kjartansson and Supheakmungkol Sarin and Knot Pipatsrisawat and Martin Jansche and Linne Ha},
  title={{Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali}},
  year=2018,
  booktitle={Proc. 6th Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU 2018)},
  pages={52--55},
  doi={10.21437/SLTU.2018-11}
}
"""

_DATASETNAME = "openslr"

_DESCRIPTION = """\
This data set contains transcribed high-quality audio of Javanese, Sundanese, Burmese, Khmer. This data set\
come from 3 different projects under OpenSLR initiative
"""

_HOMEPAGE = "https://www.openslr.org/resources.php"

_LANGUAGES = ["mya", "jav", "sun", "khm"]

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

_RESOURCES = {
    "SLR35": {
        "language": "jav",
        "files": [
            "asr_javanese_0.zip",
            "asr_javanese_1.zip",
            "asr_javanese_2.zip",
            "asr_javanese_3.zip",
            "asr_javanese_4.zip",
            "asr_javanese_5.zip",
            "asr_javanese_6.zip",
            "asr_javanese_7.zip",
            "asr_javanese_8.zip",
            "asr_javanese_9.zip",
            "asr_javanese_a.zip",
            "asr_javanese_b.zip",
            "asr_javanese_c.zip",
            "asr_javanese_d.zip",
            "asr_javanese_e.zip",
            "asr_javanese_f.zip",
        ],
        "index_files": ["asr_javanese/utt_spk_text.tsv"] * 16,
        "data_dirs": ["asr_javanese/data"] * 16,
    },
    "SLR36": {
        "language": "sun",
        "files": [
            "asr_sundanese_0.zip",
            "asr_sundanese_1.zip",
            "asr_sundanese_2.zip",
            "asr_sundanese_3.zip",
            "asr_sundanese_4.zip",
            "asr_sundanese_5.zip",
            "asr_sundanese_6.zip",
            "asr_sundanese_7.zip",
            "asr_sundanese_8.zip",
            "asr_sundanese_9.zip",
            "asr_sundanese_a.zip",
            "asr_sundanese_b.zip",
            "asr_sundanese_c.zip",
            "asr_sundanese_d.zip",
            "asr_sundanese_e.zip",
            "asr_sundanese_f.zip",
        ],
        "index_files": ["asr_sundanese/utt_spk_text.tsv"] * 16,
        "data_dirs": ["asr_sundanese/data"] * 16,
    },
    "SLR41": {
        "language": "jav",
        "files": ["jv_id_female.zip", "jv_id_male.zip"],
        "index_files": ["jv_id_female/line_index.tsv", "jv_id_male/line_index.tsv"],
        "data_dirs": ["jv_id_female/wavs", "jv_id_male/wavs"],
    },
    "SLR42": {
        "language": "khm",
        "files": ["km_kh_male.zip"],
        "index_files": ["km_kh_male/line_index.tsv"],
        "data_dirs": ["km_kh_male/wavs"],
    },
    "SLR44": {
        "language": "sun",
        "files": ["su_id_female.zip", "su_id_male.zip"],
        "index_files": ["su_id_female/line_index.tsv", "su_id_male/line_index.tsv"],
        "data_dirs": ["su_id_female/wavs", "su_id_male/wavs"],
    },
    "SLR80": {
        "language": "mya",
        "files": ["my_mm_female.zip"],
        "index_files": ["line_index.tsv"],
        "data_dirs": [""],
    },
}
_URLS = {_DATASETNAME: "https://openslr.org/resources/{subset}"}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class OpenSLRDataset(datasets.GeneratorBasedBuilder):
    """This data set contains transcribed high-quality audio of Javanese, Sundanese, Burmese, Khmer. This data set
    come from 3 different projects under OpenSLR initiative"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(name=f"{_DATASETNAME}_{subset}_{_RESOURCES[subset]['language']}_source", version=datasets.Version(_SOURCE_VERSION), description=f"{_DATASETNAME} source schema", schema="source", subset_id=f"{_DATASETNAME}")
        for subset in _RESOURCES.keys()
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_{_RESOURCES[subset]['language']}_seacrowd_sptext", version=datasets.Version(_SEACROWD_VERSION), description=f"{_DATASETNAME} SEACrowd schema", schema="seacrowd_sptext", subset_id=f"{_DATASETNAME}"
        )
        for subset in _RESOURCES.keys()
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_SLR41_jav_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=48_000),
                    "sentence": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_sptext":
            features = schemas.speech_text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        subset = self.config.name.split("_")[1]
        urls = [f"{_URLS[_DATASETNAME].format(subset=subset[3:])}/{file}" for file in _RESOURCES[subset]["files"]]
        data_dir = dl_manager.download_and_extract(urls)

        path_to_indexs = [os.path.join(path, f"{_RESOURCES[subset]['index_files'][i]}") for i, path in enumerate(data_dir)]
        path_to_datas = [os.path.join(path, f"{_RESOURCES[subset]['data_dirs'][i]}") for i, path in enumerate(data_dir)]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": [path_to_indexs, path_to_datas],
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        subset = self.config.name.split("_")[1]
        path_to_indexs, path_to_datas = filepath[0], filepath[1]
        counter = -1
        if subset in ["SLR35", "SLR36"]:
            sentence_index = {}
            for i, path_to_index in enumerate(path_to_indexs):
                with open(path_to_index, encoding="utf-8") as f:
                    lines = f.readlines()
                    for id_, line in enumerate(lines):
                        field_values = re.split(r"\t\t?", line.strip())
                        filename, user_id, sentence = field_values
                        sentence_index[filename] = sentence
                for path_to_data in sorted(Path(path_to_datas[i]).rglob("*.flac")):
                    filename = path_to_data.stem
                    if path_to_data.stem not in sentence_index:
                        continue
                    path = str(path_to_data.resolve())
                    sentence = sentence_index[filename]
                    counter += 1
                    if self.config.schema == "source":
                        example = {"path": path, "audio": path, "sentence": sentence}
                    elif self.config.schema == "seacrowd_sptext":
                        example = {
                            "id": counter,
                            "path": path,
                            "audio": path,
                            "text": sentence,
                            "speaker_id": user_id,
                            "metadata": {
                                "speaker_age": None,
                                "speaker_gender": None,
                            },
                        }
                    yield counter, example
        else:
            for i, path_to_index in enumerate(path_to_indexs):
                geneder = "female" if "female" in path_to_index else "male"
                with open(path_to_index, encoding="utf-8") as f:
                    lines = f.readlines()
                    for id_, line in enumerate(lines):
                        # Following regexs are needed to normalise the lines, since the datasets
                        # are not always consistent and have bugs:
                        line = re.sub(r"\t[^\t]*\t", "\t", line.strip())
                        field_values = re.split(r"\t\t?", line)
                        if len(field_values) != 2:
                            continue
                        filename, sentence = field_values
                        path = os.path.join(path_to_datas[i], f"{filename}.wav")
                        counter += 1
                        if self.config.schema == "source":
                            example = {"path": path, "audio": path, "sentence": sentence}
                        elif self.config.schema == "seacrowd_sptext":
                            example = {
                                "id": counter,
                                "path": path,
                                "audio": path,
                                "text": sentence,
                                "speaker_id": None,
                                "metadata": {
                                    "speaker_age": None,
                                    "speaker_gender": geneder,
                                },
                            }
                        yield counter, example
