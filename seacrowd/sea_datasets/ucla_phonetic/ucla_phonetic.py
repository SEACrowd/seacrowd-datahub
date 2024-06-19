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
This dataset contains audio recordings and phonetic transcriptions of word utterances for various low-resource SEA languages.
Each language has a directory of text and audio files, with the latter forming one data subset.
The dataset is prepared from the online UCLA phonetic dataset, which contains 7000 utterances across 100 low-resource languages, phonetically aligned using various automatic approaches, and manually fixed for misalignments.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{li2021multilingual,
  title={Multilingual phonetic dataset for low resource speech recognition},
  author={Li, Xinjian and Mortensen, David R and Metze, Florian and Black, Alan W},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6958--6962},
  year={2021},
  organization={IEEE}
}
"""

_DATASETNAME = "ucla_phonetic"

_DESCRIPTION = """\
This dataset contains audio recordings and phonetic transcriptions of word utterances for various low-resource SEA languages.
Each language has a directory of text and audio files, with the latter forming one data subset.
The dataset is prepared from the online UCLA phonetic dataset, which contains 7000 utterances across 100 low-resource languages, phonetically aligned using various automatic approaches, and manually fixed for misalignments.
"""

_HOMEPAGE = "https://github.com/xinjli/ucla-phonetic-corpus"

_LANGUAGES = ["ace", "brv", "hil", "hni", "ilo", "khm", "mak", "mya", "pam"]

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_LOCAL = False

_DATA_URL = "https://github.com/xinjli/ucla-phonetic-corpus/releases/download/v1.0/data.tar.gz"

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


def seacrowd_config_constructor(lang, schema, version):
    if lang not in _LANGUAGES:
        raise ValueError(f"Invalid lang {lang}")

    if schema not in ["source", "seacrowd_sptext"]:
        raise ValueError(f"Invalid schema: {schema}")

    return SEACrowdConfig(
        name=f"ucla_phonetic_{lang}_{schema}",
        version=datasets.Version(version),
        description=f"UCLA Phonetic {schema} for {lang}",
        schema=schema,
        subset_id=f"{lang}_{schema}",
    )


class UCLAPhoneticDataset(datasets.GeneratorBasedBuilder):
    """This dataset contains audio recordings and phonetic transcriptions of word utterances for various low-resource SEA languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = (
        [
            SEACrowdConfig(
                name="ucla_phonetic_source",
                version=datasets.Version(_SOURCE_VERSION),
                description="UCLA Phonetic source for ace",
                schema="source",
                subset_id="ace_source",
            ),
            SEACrowdConfig(
                name="ucla_phonetic_seacrowd_sptext",
                version=datasets.Version(_SOURCE_VERSION),
                description="UCLA Phonetic seacrowd+sptext for ace",
                schema="seacrowd_sptext",
                subset_id="ace_seacrowd_sptext",
            ),
        ]
        + [seacrowd_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGUAGES]
        + [seacrowd_config_constructor(lang, "seacrowd_sptext", _SEACROWD_VERSION) for lang in _LANGUAGES]
    )

    DEFAULT_CONFIG_NAME = "ucla_phonetic_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "text": datasets.Value("string"), "audio": datasets.Audio(sampling_rate=16_000)})
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

        lang, schema = self.config.subset_id.split("_", maxsplit=1)
        data_dir = dl_manager.download_and_extract(_DATA_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data", lang, "text.txt"),
                    "audiopath": Path(os.path.join(data_dir, "data", lang, "audio")),
                },
            )
        ]

    def _generate_examples(self, filepath: Path, audiopath: Path) -> Tuple[int, Dict]:

        audiofiles = {}
        for audiofile in audiopath.iterdir():
            audio_idx = os.path.basename(audiofile).split(".")[0]
            audiofiles[audio_idx] = audiofile

        if self.config.schema == "source":
            for line_idx, line in enumerate(open(filepath)):
                audio_idx, text = line.strip().split(maxsplit=1)
                yield line_idx, {"id": line_idx, "text": text, "audio": str(audiofiles[audio_idx])}

        elif self.config.schema == "seacrowd_sptext":
            for line_idx, line in enumerate(open(filepath)):
                audio_idx, text = line.strip().split(maxsplit=1)
                yield line_idx, {"id": line_idx, "path": str(audiofiles[audio_idx]), "audio": str(audiofiles[audio_idx]), "text": text, "speaker_id": None, "metadata": {"speaker_age": None, "speaker_gender": None}}
