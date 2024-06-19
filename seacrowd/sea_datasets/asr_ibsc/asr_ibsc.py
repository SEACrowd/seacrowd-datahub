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
import fsspec
import pandas as pd
from fsspec.callbacks import TqdmCallback

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@inproceedings{Juan14,
    Title = {Semi-supervised G2P bootstrapping and its application to ASR for a very under-resourced language: Iban},
    Author = {Sarah Samson Juan and Laurent Besacier and Solange Rossato},
    Booktitle = {Proceedings of Workshop for Spoken Language Technology for Under-resourced (SLTU)},
    Year = {2014}}
    Month = {May},

@inproceedings{Juan2015,
    Title = {Using Resources from a closely-Related language to develop ASR for a very under-resourced Language: A case study for Iban},
    Author = {Sarah Samson Juan and Laurent Besacier and Benjamin Lecouteux and Mohamed Dyab},
    Booktitle = {Proceedings of INTERSPEECH},
    Year = {2015},
    Month = {September}}
    Address = {Dresden, Germany},
"""

_DATASETNAME = "asr_ibsc"

_DESCRIPTION = """\
This package contains Iban language text and speech suitable for Automatic
Speech Recognition (ASR) experiments. In addition, transcribed speech, 2M tokens
corpus crawled from an online newspaper site is provided. News data was provided
by a local radio station in Sarawak, Malaysia.
"""

_HOMEPAGE = "https://github.com/sarahjuan/iban"

_LANGUAGES = ["iba"]

_LICENSE = Licenses.CC_BY_SA_3_0.value

_LOCAL = False

_URL = "https://github.com/sarahjuan/iban/tree/master/data"

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # sptext

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class ASRIbanDataset(datasets.GeneratorBasedBuilder):
    """Iban language text and speech suitable for ASR experiments"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{_SEACROWD_SCHEMA}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=_SEACROWD_SCHEMA,
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "transcription": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]]  # speech_text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # prepare data directory
        data_dir = Path.cwd() / "data" / "asr_ibsc"
        data_dir.mkdir(parents=True, exist_ok=True)

        # download data
        # if rate limiting is an issue, pass github username and token
        username = None
        token = None
        fs = fsspec.filesystem("github", org="sarahjuan", repo="iban", ref="master", username=username, token=token)
        fs.clear_instance_cache()

        # download annotation
        print("Downloading annotation...")
        fs.get(fs.ls("data/train/"), (data_dir / "train").as_posix(), recursive=True)
        fs.get(fs.ls("data/test/"), (data_dir / "test").as_posix(), recursive=True)

        # download audio files
        print("Downloading audio files (~1GB). It may take several minutes...")
        for idx, folder in enumerate(fs.ls("data/wav/")):
            folder_name = folder.split("/")[-1]
            pbar = TqdmCallback(tqdm_kwargs={"desc": f"-> {folder_name} [{idx+1:2d}/{len(fs.ls('data/wav/'))}]", "unit": "file"})
            fs.get(fs.ls(f"data/wav/{folder_name}/"), (data_dir / "wav" / folder_name).as_posix(), recursive=True, callback=pbar)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, data_dir: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        text_file = data_dir / split / f"{split}_text"
        utt2spk_file = data_dir / split / f"{split}_utt2spk"
        wav_scp_file = data_dir / split / f"{split}_wav.scp"

        # load the data
        text_df = pd.read_csv(text_file, sep="  ", header=None, names=["utt_id", "text"])
        utt2spk_df = pd.read_csv(utt2spk_file, sep="\t", header=None, names=["utt_id", "speaker"])
        wav_df = pd.read_csv(wav_scp_file, sep="\t", header=None, names=["utt_id", "wav_path"])
        merged_df = pd.merge(text_df, utt2spk_df, on="utt_id")
        merged_df = pd.merge(merged_df, wav_df, on="utt_id")

        for _, row in merged_df.iterrows():
            wav_file = data_dir / "wav" / row["speaker"] / row["wav_path"].split("/")[-1]

            if self.config.schema == "source":
                yield row["utt_id"], {
                    "audio": str(wav_file.as_posix()),
                    "transcription": row["text"],
                    "speaker_id": row["speaker"],
                }
            elif self.config.schema == _SEACROWD_SCHEMA:
                yield row["utt_id"], {
                    "id": row["utt_id"],
                    "path": str(wav_file),
                    "audio": str(wav_file.as_posix()),
                    "text": row["text"],
                    "speaker_id": row["speaker"],
                    "metadata": None,
                }
