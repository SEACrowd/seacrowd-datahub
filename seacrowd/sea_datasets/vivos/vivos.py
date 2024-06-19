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
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{luong-vu-2016-non,
    title = "A non-expert {K}aldi recipe for {V}ietnamese Speech Recognition System",
    author = "Luong, Hieu-Thi  and
      Vu, Hai-Quan",
    editor = "Murakami, Yohei  and
      Lin, Donghui  and
      Ide, Nancy  and
      Pustejovsky, James",
    booktitle = "Proceedings of the Third International Workshop on Worldwide Language Service
    Infrastructure and Second Workshop on Open Infrastructures and Analysis Frameworks for
    Human Language Technologies ({WLSI}/{OIAF}4{HLT}2016)",
    month = dec,
    year = "2016",
    address = "Osaka, Japan",
    publisher = "The COLING 2016 Organizing Committee",
    url = "https://aclanthology.org/W16-5207",
    pages = "51--55",
    abstract = "In this paper we describe a non-expert setup for Vietnamese speech recognition
    system using Kaldi toolkit. We collected a speech corpus over fifteen hours from about fifty
    Vietnamese native speakers and using it to test the feasibility of our setup. The essential
    linguistic components for the Automatic Speech Recognition (ASR) system was prepared basing
    on the written form of the language instead of expertise knowledge on linguistic and phonology
    as commonly seen in rich resource languages like English. The modeling of tones by integrating
    them into the phoneme and using the phonetic decision tree is also discussed. Experimental
    results showed this setup for ASR systems does yield competitive results while still have
    potentials for further improvements.",
}
"""

_DATASETNAME = "vivos"

_DESCRIPTION = """\
VIVOS is a Vietnamese speech corpus consisting of 15 hours of recording speech prepared for
Automatic Speech Recognition task. This speech corpus is collected by recording speech data
from more than 50 native Vietnamese volunteers.
"""

_HOMEPAGE = "https://zenodo.org/records/7068130"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

_URLS = {
    "audio": "https://huggingface.co/datasets/vivos/resolve/main/data/vivos.tar.gz",
    "train_prompt": "https://huggingface.co/datasets/vivos/resolve/main/data/prompts-train.txt.gz",
    "test_prompt": "https://huggingface.co/datasets/vivos/resolve/main/data/prompts-test.txt.gz",
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

logger = datasets.logging.get_logger(__name__)


class VIVOSDataset(datasets.GeneratorBasedBuilder):
    """
    VIVOS is a Vietnamese speech corpus from https://zenodo.org/records/7068130.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_sptext",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_sptext",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "speaker_id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
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
        """
        Returns SplitGenerators.
        """

        audio_path = dl_manager.download(_URLS["audio"])
        train_prompt_path = dl_manager.download_and_extract(_URLS["train_prompt"])
        test_prompt_path = dl_manager.download_and_extract(_URLS["test_prompt"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "prompts_path": train_prompt_path,
                    "clips_path": "vivos/train/waves",
                    "audio_files": dl_manager.iter_archive(audio_path),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "prompts_path": test_prompt_path,
                    "clips_path": "vivos/test/waves",
                    "audio_files": dl_manager.iter_archive(audio_path),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, prompts_path: Path, clips_path: Path, audio_files, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """
        examples = {}
        with open(prompts_path, encoding="utf-8") as f:
            if self.config.schema == "source":
                for row in f:
                    data = row.strip().split(" ", 1)
                    speaker_id = data[0].split("_")[0]
                    audio_path = "/".join([clips_path, speaker_id, data[0] + ".wav"])
                    examples[audio_path] = {
                        "speaker_id": speaker_id,
                        "path": audio_path,
                        "sentence": data[1],
                    }
            elif self.config.schema == "seacrowd_sptext":
                audio_id = 0
                for row in f:
                    data = row.strip().split(" ", 1)
                    speaker_id = data[0].split("_")[0]
                    audio_path = "/".join([clips_path, speaker_id, data[0] + ".wav"])
                    examples[audio_path] = {
                        "id": audio_id,
                        "path": audio_path,
                        "text": data[1],
                        "speaker_id": speaker_id,
                        "metadata": {
                            "speaker_age": None,
                            "speaker_gender": None,
                        },
                    }
                    audio_id += 1

        idx = 0
        for path, f in audio_files:
            if path.startswith(clips_path):
                if path in examples:
                    audio = {"path": path, "bytes": f.read()}
                    yield idx, {**examples[path], "audio": audio}
                    idx += 1
            else:
                continue