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
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """
@techreport{gowajee,
    title = {{Gowajee Corpus}},
    author = {Ekapol Chuangsuwanich and Atiwong Suchato and Korrawe Karunratanakul and Burin Naowarat and Chompakorn CChaichot
and Penpicha Sangsa-nga and Thunyathon Anutarases and Nitchakran Chaipojjana and Yuatyong Chaichana},
    year = {2020},
    institution = {Chulalongkorn University, Faculty of Engineering, Computer Engineering Department},
    month = {12},
    Date-Added = {2023-07-30},
    url = {https://github.com/ekapolc/gowajee_corpus}
    note = {Version 0.9.3}
}
"""

_DATASETNAME = "gowajee"

_DESCRIPTION = """
The Gowajee corpus was collected in the Automatic Speech Recognition class offered at
Chulalongkorn University as a homework assignment. Each group was asked to come up with an
example smart home application.
"""

_HOMEPAGE = "https://github.com/ekapolc/gowajee_corpus"

_LANGUAGES = ["tha"]

_LICENSE = Licenses.MIT.value

_LOCAL = False

_URL = "https://drive.google.com/file/d/1soriRMMuZI5w5RZOjAnbpocBZxT6i1-l/view"  # ~1.5GB

_SUPPORTED_TASKS = [Tasks.SPEECH_TO_TEXT_TRANSLATION]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # sptext

_SOURCE_VERSION = "0.9.3"

_SEACROWD_VERSION = "2024.06.20"


class GowajeeDataset(datasets.GeneratorBasedBuilder):
    """Automatic Speech Recognition dataset on smart home application where the wakeword is "Gowajee"."""

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
        # check if gdown is installed
        try:
            import gdown
        except ImportError as err:
            raise ImportError("Please install `gdown` to enable downloading data from google drive.") from err

        # download data from gdrive
        output_dir = Path.cwd() / "data" / "gowajee"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "gowajee_v0-9-3.zip"
        if not output_file.exists():
            gdown.download(_URL, str(output_file), fuzzy=True)
        else:
            print(f"File already downloaded: {str(output_file)}")

        # extract data
        data_dir = Path(dl_manager.extract(output_file)) / "v0.9.2"

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
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, data_dir: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        text_file = data_dir / split / "text"
        utt2spk_file = data_dir / split / "utt2spk"
        wav_scp_file = data_dir / split / "wav.scp"

        # load the data
        with open(text_file, "r", encoding="utf-8") as f:
            text_lines = f.readlines()
            text_lines = [line.strip().split(" ", 1) for line in text_lines]
        with open(utt2spk_file, "r", encoding="utf-8") as f:
            utt2spk_lines = f.readlines()
            utt2spk_lines = [line.strip().split(" ") for line in utt2spk_lines]
        with open(wav_scp_file, "r", encoding="utf-8") as f:
            wav_scp_lines = f.readlines()
            wav_scp_lines = [line.strip().split(" ", 1) for line in wav_scp_lines]

        assert len(text_lines) == len(utt2spk_lines) == len(wav_scp_lines), f"Length of text_lines: {len(text_lines)}, utt2spk_lines: {len(utt2spk_lines)}, wav_scp_lines: {len(wav_scp_lines)}"

        text_df = pd.DataFrame(text_lines, columns=["utt_id", "text"])
        utt2spk_df = pd.DataFrame(utt2spk_lines, columns=["utt_id", "speaker"])
        wav_df = pd.DataFrame(wav_scp_lines, columns=["utt_id", "wav_path"])
        merged_df = pd.merge(text_df, utt2spk_df, on="utt_id")
        merged_df = pd.merge(merged_df, wav_df, on="utt_id")

        for _, row in merged_df.iterrows():
            wav_file = data_dir / row["wav_path"]

            if self.config.schema == "source":
                yield row["utt_id"], {
                    "audio": str(wav_file),
                    "transcription": row["text"],
                    "speaker_id": row["speaker"],
                }
            elif self.config.schema == _SEACROWD_SCHEMA:
                yield row["utt_id"], {
                    "id": row["utt_id"],
                    "path": str(wav_file),
                    "audio": str(wav_file),
                    "text": row["text"],
                    "speaker_id": row["speaker"],
                    "metadata": None,
                }
