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

import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = ""  # no dataset/paper citation found

_DATASETNAME = "thai_elderly_speech"

_DESCRIPTION = """\
The Thai Elderly Speech dataset by Data Wow and VISAI Version 1 dataset aims at
advancing Automatic Speech Recognition (ASR) technology specifically for the
elderly population. Researchers can use this dataset to advance ASR technology
for healthcare and smart home applications. The dataset consists of 19,200 audio
files, totaling 17 hours and 11 minutes of recorded speech. The files are
divided into 2 categories: Healthcare (relating to medical issues and services
in 30 medical categories) and Smart Home (relating to smart home devices in 7
household contexts). The dataset contains 5,156 unique sentences spoken by 32
seniors (10 males and 22 females), aged 57-60 years old (average age of 63
years).
"""

_HOMEPAGE = "https://github.com/VISAI-DATAWOW/Thai-Elderly-Speech-dataset/releases/tag/v1.0.0"

_LANGUAGES = ["tha"]
_SUBSETS = ["healthcare", "smarthome"]

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

_URLS = [
    "https://github.com/VISAI-DATAWOW/Thai-Elderly-Speech-dataset/releases/download/v1.0.0/Dataset.zip.001",
    "https://github.com/VISAI-DATAWOW/Thai-Elderly-Speech-dataset/releases/download/v1.0.0/Dataset.zip.002",
    "https://github.com/VISAI-DATAWOW/Thai-Elderly-Speech-dataset/releases/download/v1.0.0/Dataset.zip.003",
]

_SUPPORTED_TASKS = [Tasks.SPEECH_TO_TEXT_TRANSLATION]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # sptext

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class ThaiElderlySpeechDataset(datasets.GeneratorBasedBuilder):
    """A speech dataset from elderly Thai speakers."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in _SUBSETS:
        BUILDER_CONFIGS += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} {subset} source schema",
                schema="source",
                subset_id=subset,
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset}_{_SEACROWD_SCHEMA}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {subset} SEACrowd schema",
                schema=_SEACROWD_SCHEMA,
                subset_id=subset,
            ),
        ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_healthcare_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "filename": datasets.Value("string"),
                    "transcription": datasets.Value("string"),
                    "speaker": {
                        "id": datasets.Value("string"),
                        "age": datasets.Value("int32"),
                        "gender": datasets.Value("string"),
                    },
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]]  # ssp_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        zip_files = list(map(Path, dl_manager.download(_URLS)))
        zip_combined = zip_files[0].parent / "thai_elderly_speech.zip"

        with open(str(zip_combined), "wb") as out_file:
            for zip_file in zip_files:
                with open(str(zip_file), "rb") as in_file:
                    out_file.write(in_file.read())

        data_file = Path(dl_manager.extract(zip_combined)) / "Dataset"
        subset_id = self.config.subset_id

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "speaker_file": data_file / "speaker_demography.json",
                    "audio_dir": data_file / subset_id.title() / "Record",
                    "transcript_file": data_file / subset_id.title() / "transcription.json",
                },
            ),
        ]

    def _generate_examples(self, speaker_file: Path, audio_dir: Path, transcript_file: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # read speaker information
        with open(speaker_file, "r", encoding="utf-8") as f:
            speaker_info = json.load(f)
            speaker_dict = {speaker["speaker_id"]: {"age": speaker["age"], "gender": speaker["gender"]} for speaker in speaker_info}

        # read transcript information
        with open(transcript_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        for idx, instance in enumerate(annotations):
            transcript = instance["transcript"]

            speaker_id = instance["speaker_id"]
            speaker_info = speaker_dict[int(speaker_id)]

            filename = instance["filename"]
            audio_file = str(audio_dir / (filename + ".wav"))

            if self.config.schema == "source":
                yield idx, {
                    "audio": audio_file,
                    "filename": filename,
                    "transcription": transcript,
                    "speaker": {
                        "id": speaker_id,
                        "age": speaker_info["age"],
                        "gender": speaker_info["gender"],
                    },
                }
            elif self.config.schema == _SEACROWD_SCHEMA:
                yield idx, {
                    "id": idx,
                    "path": audio_file,
                    "audio": audio_file,
                    "text": transcript,
                    "speaker_id": speaker_id,
                    "metadata": {
                        "speaker_age": speaker_info["age"],
                        "speaker_gender": speaker_info["gender"],
                    },
                }
