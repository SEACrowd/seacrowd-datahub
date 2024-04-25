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
from huggingface_hub import HfFileSystem
from pyarrow import parquet as pq

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import SCHEMA_TO_FEATURES, TASK_TO_SCHEMA, Licenses, Tasks

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

_BASE_URL = "https://huggingface.co/datasets/meisin123/iban_speech_corpus/resolve/main/data/{filename}"

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # sptext

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


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
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[
                TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]
            ]  # speech_text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        file_list = HfFileSystem().ls("datasets/meisin123/iban_speech_corpus/data", detail=False)
        data_urls = []
        for filename in file_list:
            if filename.endswith(".parquet"):
                filename = filename.split("/")[-1]
                url = _BASE_URL.format(filename=filename)
                data_urls.append(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_paths": list(map(Path, dl_manager.download(sorted(data_urls))))
                },
            ),
        ]

    def _generate_examples(self, data_paths: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        key = 0
        for data_path in data_paths:
            with open(data_path, "rb") as f:
                pf = pq.ParquetFile(f)

                for row_group in range(pf.num_row_groups):
                    df = pf.read_row_group(row_group).to_pandas()

                    for row in df.itertuples():
                        if self.config.schema == "source":
                            yield key, {
                                "audio": row.audio,
                                "transcription": row.transcription,
                            }
                        elif self.config.schema == _SEACROWD_SCHEMA:
                            yield key, {
                                "id": str(key),
                                "path": None,
                                "audio": row.audio,
                                "text": row.transcription,
                                "speaker_id": None,
                                "metadata": None,
                            }
                        key += 1
