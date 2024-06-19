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
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@inproceedings{mazumder2021mswc,
    author = {Mazumder, Mark and Chitlangia, Sharad and Banbury, Colby and Kang, Yiping and Ciro, Juan and Achorn, Keith and Galvez,
    Daniel and Sabini, Mark and Mattson, Peter and Kanter, David and Diamos, Greg and Warden, Pete and Meyer, Josh and Janapa Reddi,
    Vijay},
    booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
    editor = {J. Vanschoren and S. Yeung},
    pages = {},
    publisher = {Curran},
    title = {Multilingual Spoken Words Corpus},
    url = {https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/fe131d7f5a6b38b23cc967316c13dae2-Paper-round2.pdf},
    volume = {1},
    year = {2021}
}
"""

_DATASETNAME = "mswc"

_DESCRIPTION = """\
Multilingual Spoken Words Corpus is a large and growing audio dataset of spoken words in 50 languages collectively spoken by over 5 billion people, for academic research and commercial applications in keyword spotting and spoken term search.
"""

_HOMEPAGE = "https://huggingface.co/datasets/MLCommons/ml_spoken_words"

_LANGUAGES = ["cnh", "ind", "vie"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LANGUAGE_NAME_MAP = {
    "cnh": "cnh",
    "ind": "id",
    "vie": "vi",
}

_FORMATS = ["wav", "opus"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = "https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/refs%2Fconvert%2Fparquet/{lang}_{format}/{split}/0000.parquet?download=true"

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MSWC(datasets.GeneratorBasedBuilder):
    """
    Multilingual Spoken Words Corpus is a large and growing audio dataset of spoken words in 50 languages collectively spoken by over 5 billion people, for academic research and commercial applications in keyword spotting and spoken term search.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []

    for language in _LANGUAGES:
        for format in _FORMATS:
            subset_id = f"{_DATASETNAME}_{language}_{format}"
            BUILDER_CONFIGS.append(
                SEACrowdConfig(name=f"{subset_id}_source", version=SOURCE_VERSION, description=f"{_DATASETNAME} source schema", schema="source", subset_id=subset_id),
            )

    seacrowd_schema_config: list[SEACrowdConfig] = []

    for seacrowd_schema in _SUPPORTED_SCHEMA_STRINGS:
        for language in _LANGUAGES:
            for format in _FORMATS:
                subset_id = f"{_DATASETNAME}_{language}_{format}"
                seacrowd_schema_config.append(
                    SEACrowdConfig(
                        name=f"{subset_id}_{seacrowd_schema}",
                        version=SEACROWD_VERSION,
                        description=f"{_DATASETNAME} {seacrowd_schema} schema",
                        schema=f"{seacrowd_schema}",
                        subset_id=subset_id,
                    )
                )

    BUILDER_CONFIGS.extend(seacrowd_schema_config)

    DEFAULT_CONFIG_NAME = f"{_LANGUAGES[0]}_{_FORMATS[0]}_source"

    def _info(self) -> datasets.DatasetInfo:

        _, _, format = str(self.config.subset_id).split("_")

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "is_valid": datasets.Value("bool"),
                    "language": datasets.ClassLabel(num_classes=3),
                    "speaker_id": datasets.Value("string"),
                    "gender": datasets.ClassLabel(num_classes=4),
                    "keyword": datasets.Value("string"),
                    "audio": datasets.Audio(decode=False, sampling_rate=16000 if format == "wav" else 48000),
                }
            )

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SPEECH_RECOGNITION]).lower()}":
            features = schemas.speech_text_features

        else:
            raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        split_names = ["train", "validation", "test"]

        result = []

        _, language, format = str(self.config.subset_id).split("_")

        for split_name in split_names:
            path = dl_manager.download_and_extract(_URLS.format(split=split_name, lang=_LANGUAGE_NAME_MAP[language], format=format))

            result.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={
                        "path": path,
                        "split": split_name,
                        "language": language,
                        "format": format,
                    },
                ),
            )

        return result

    def _generate_examples(self, path: Path, split: str, language: str, format: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        idx = 0

        if self.config.schema == "source":
            df = pd.read_parquet(path)

            for _, row in df.iterrows():
                yield idx, row.to_dict()
                idx += 1

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SPEECH_RECOGNITION]).lower()}":
            df = pd.read_parquet(path)

            base_folder = os.path.dirname(path)
            base_folder = os.path.join(base_folder, _DATASETNAME, language, format, split)

            if not os.path.exists(base_folder):
                os.makedirs(base_folder)

            audio_paths = []

            for _, row in df.iterrows():
                audio_dict = row["audio"]
                file_name = audio_dict["path"]

                path = os.path.join(base_folder, file_name)

                audio_dict["path"] = path

                with open(path, "wb") as f:
                    f.write(audio_dict["bytes"])

                audio_paths.append(path)

            df.rename(columns={"label": "text"}, inplace=True)

            df["path"] = audio_paths

            df["id"] = df.index + idx
            df = df.assign(text="").astype({"text": "str"})
            df = df.assign(metadata=[{"speaker_age": 0, "speaker_gender": gender} for gender in df["gender"]]).astype({"metadata": "object"})

            df.drop(columns=["file", "is_valid", "language", "gender", "keyword"], inplace=True)

            for _, row in df.iterrows():
                yield idx, row.to_dict()
                idx += 1

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
