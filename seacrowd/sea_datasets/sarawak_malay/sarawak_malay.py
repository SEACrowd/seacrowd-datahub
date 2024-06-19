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

try:
    import audiosegment
except:
    print("Install the `audiosegment` package to use.")

try:
    import textgrid
except:
    print("Install the `textgrid` package to use.")

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@INPROCEEDINGS{
10337314,
author={Rahim, Mohd Zulhafiz and Juan, Sarah Samson and Mohamad, Fitri Suraya},
booktitle={2023 International Conference on Asian Language Processing (IALP)},
title={Improving Speaker Diarization for Low-Resourced Sarawak Malay Language Conversational Speech Corpus},
year={2023},
pages={228-233},
keywords={Training;Oral communication;Data models;Usability;Speech processing;Testing;Speaker diarization;x-vectors;clustering;low-resource;auto-labeling;pseudo-labeling;unsupervised},
doi={10.1109/IALP61005.2023.10337314}}
"""

_DATASETNAME = "sarawak_malay"

_DESCRIPTION = """\
This is a Sarawak Malay conversation data for the purpose of speech technology research. \
At the moment, this is an experimental data and currently used for investigating \
speaker diarization. The data was collected by Faculty of Computer Science and \
Information Technology, Universiti Malaysia Sarawak. The data consists of 38 conversations \
that have been transcribed using Transcriber (see TextGrid folder), where each file \
contains two speakers. Each conversation was recorded by different individuals using microphones \
from mobile devices or laptops thus, different file formats were collected from the data collectors. \
All data was then standardized to mono, 16000Khz, wav format.
"""

_HOMEPAGE = "https://github.com/sarahjuan/sarawakmalay"

_LANGUAGES = ["zlm"]

_LICENSE = Licenses.CC0_1_0.value
_LOCAL = False

_URLS = {
    _DATASETNAME: "https://github.com/sarahjuan/sarawakmalay/archive/refs/heads/main.zip",
}
_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION, Tasks.TEXT_TO_SPEECH]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class SarawakMalayDataset(datasets.GeneratorBasedBuilder):
    """This is experimental Sarawak Malay conversation data collected by \
        Universiti Malaysia Sarawak for speech technology research, \
        specifically speaker diarization. The data includes 38 conversations, \
        each with two speakers, recorded on various devices and then standardized to mono, \
        16000Khz, wav format."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "sptext"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "metadata": {
                        "malay_text": datasets.Value("string"),
                    },
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.speech_text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "sarawakmalay-main"),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        id_counter = 0
        filenames = filter(lambda x: x.endswith(".wav"), os.listdir(f"{filepath}/wav"))
        filenames = map(lambda x: x.replace(".wav", ""), filenames)

        os.makedirs(f"{filepath}/segmented", exist_ok=True)
        for i, filename in enumerate(filenames):
            info = textgrid.TextGrid.fromFile(f"{filepath}/TextGrid/{filename}.TextGrid")
            if len(info) == 3:
                sarawak_conversation, malay_conversation, speakers = info
            else:
                sarawak_conversation, malay_conversation, speakers, _ = info

            audio_file = audiosegment.from_file(f"{filepath}/wav/{filename}.wav").resample(sample_rate_Hz=16000)

            for sarawak_tg, malay_tg, speaker in zip(sarawak_conversation, malay_conversation, speakers):
                start, end, text = sarawak_tg.minTime, sarawak_tg.maxTime, sarawak_tg.mark
                malay_text = malay_tg.mark
                speaker_id = speaker.mark

                start_sec, end_sec = int(start * 1000), int(end * 1000)
                segment = audio_file[start_sec:end_sec]
                segement_filename = f"{filepath}/segmented/{filename}-{round(start, 0)}-{round(end, 0)}.wav"
                segment.export(segement_filename, format="wav")

                if self.config.schema == "source":
                    yield id_counter, {
                        "id": id_counter,
                        "speaker_id": speaker_id,
                        "path": f"{filepath}/wav/{filename}.wav",
                        "audio": segement_filename,
                        "text": text,
                        "metadata": {
                            "malay_text": malay_text,
                        },
                    }

                elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                    yield id_counter, {"id": id_counter, "speaker_id": speaker_id, "path": f"{filepath}/wav/{filename}.wav", "audio": segement_filename, "text": text, "metadata": {"speaker_age": None, "speaker_gender": None}}

                id_counter += 1
