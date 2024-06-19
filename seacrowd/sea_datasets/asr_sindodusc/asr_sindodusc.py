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

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

# no bibtex citation
_CITATION = ""
_DATASETNAME = "asr_sindodusc"
_DESCRIPTION = """\
This open-source dataset consists of 3.5 hours of transcribed Indonesian
scripted speech focusing on daily use sentences, where 3,296 utterances
contributed by ten speakers were contained.
"""

_HOMEPAGE = "https://magichub.com/datasets/indonesian-scripted-speech-corpus-daily-use-sentence/"
_LANGUAGES = ["ind"]
_LICENSE = Licenses.CC_BY_NC_ND_4_0.value
_LOCAL = False
_URLS = {
    _DATASETNAME: "https://magichub.com/df/df.php?file_name=Indonesian_Scripted_Speech_Corpus_Daily_Use_Sentence.zip",
}
_SUPPORTED_TASKS = [Tasks.TEXT_TO_SPEECH, Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ASRSindodusc(datasets.GeneratorBasedBuilder):
    """ASR-Sindodusc consists transcribed Indonesian scripted speech focusing on daily use sentences"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "sptext"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "channel": datasets.Value("string"),
                    "uttrans_id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "transcription": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "speaker_gender": datasets.Value("string"),
                    "speaker_age": datasets.Value("int64"),
                    "speaker_region": datasets.Value("string"),
                    "speaker_device": datasets.Value("string"),
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
        """Returns SplitGenerators."""

        data_paths = {
            _DATASETNAME: Path(dl_manager.download_and_extract(_URLS[_DATASETNAME])),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_paths[_DATASETNAME],
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        # read UTTRANSINFO file
        # columns: channel, uttrans_id, speaker_id, prompt (empty field), transcription
        uttransinfo_filepath = os.path.join(filepath, "UTTRANSINFO.txt")
        with open(uttransinfo_filepath, "r", encoding="utf-8") as uttransinfo_file:
            uttransinfo_data = uttransinfo_file.readlines()
        uttransinfo_data = uttransinfo_data[1:]  # remove header
        uttransinfo_data = [s.strip("\n").split("\t") for s in uttransinfo_data]

        # read SPKINFO file
        # columns: channel, speaker_id, gender, age, region, device
        spkinfo_filepath = os.path.join(filepath, "SPKINFO.txt")
        with open(spkinfo_filepath, "r", encoding="utf-8") as spkinfo_file:
            spkinfo_data = spkinfo_file.readlines()
        spkinfo_data = spkinfo_data[1:]  # remove header
        spkinfo_data = [s.strip("\n").split("\t") for s in spkinfo_data]
        for i, s in enumerate(spkinfo_data):
            if s[2] == "M":
                s[2] = "male"
            elif s[2] == "F":
                s[2] = "female"
            else:
                s[2] = None
        # dictionary of metadata of each speaker
        spkinfo_dict = {s[1]: {"speaker_gender": s[2], "speaker_age": int(s[3]), "speaker_region": s[4], "speaker_device": s[5]} for s in spkinfo_data}

        num_sample = len(uttransinfo_data)

        for i in range(num_sample):
            wav_path = os.path.join(filepath, "WAV", uttransinfo_data[i][2], uttransinfo_data[i][1])

            if self.config.schema == "source":
                example = {
                    "id": str(i),
                    "channel": uttransinfo_data[i][0],
                    "uttrans_id": uttransinfo_data[i][1],
                    "speaker_id": uttransinfo_data[i][2],
                    "transcription": uttransinfo_data[i][4],
                    "path": wav_path,
                    "audio": wav_path,
                    "speaker_gender": spkinfo_dict[uttransinfo_data[i][2]]["speaker_gender"],
                    "speaker_age": spkinfo_dict[uttransinfo_data[i][2]]["speaker_age"],
                    "speaker_region": spkinfo_dict[uttransinfo_data[i][2]]["speaker_region"],
                    "speaker_device": spkinfo_dict[uttransinfo_data[i][2]]["speaker_device"],
                }
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                example = {
                    "id": str(i),
                    "speaker_id": uttransinfo_data[i][2],
                    "path": wav_path,
                    "audio": wav_path,
                    "text": uttransinfo_data[i][4],
                    "metadata": {"speaker_age": spkinfo_dict[uttransinfo_data[i][2]]["speaker_age"], "speaker_gender": spkinfo_dict[uttransinfo_data[i][2]]["speaker_gender"]},
                }

            yield i, example
