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
@misc{Bud500,
    author = {Anh Pham, Khanh Linh Tran, Linh Nguyen, Thanh Duy Cao, Phuc Phan, Duong A. Nguyen},
    title = {Bud500: A Comprehensive Vietnamese ASR Dataset},
    url = {https://github.com/quocanh34/Bud500},
    year = {2024}
}
"""

_DATASETNAME = "bud500"

_DESCRIPTION = """\
Bud500 is a diverse Vietnamese speech corpus designed to support ASR research
community. With aprroximately 500 hours of audio, it covers a broad spectrum of
topics including podcast, travel, book, food, and so on, while spanning accents
from Vietnam's North, South, and Central regions. Derived from free public audio
resources, this publicly accessible dataset is designed to significantly enhance
the work of developers and researchers in the field of speech recognition.
Before using this dataloader, please accept the acknowledgement at
https://huggingface.co/datasets/linhtran92/viet_bud500 and use huggingface-cli
login for authentication.
"""

_HOMEPAGE = "https://huggingface.co/datasets/linhtran92/viet_bud500"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.APACHE_2_0.value

_LOCAL = False

_BASE_URL = "https://huggingface.co/datasets/linhtran92/viet_bud500/resolve/main/data/{filename}"

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # sptext

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class Bud500Dataset(datasets.GeneratorBasedBuilder):
    """A diverse Vietnamese speech corpus with aprroximately 500 hours of audio."""

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
        file_list = HfFileSystem().ls("datasets/linhtran92/viet_bud500/data", detail=False)
        train_urls, test_urls, val_urls = [], [], []

        for filename in file_list:
            if filename.endswith(".parquet"):
                filename = filename.split("/")[-1]
                split = filename.split("-")[0]
                url = _BASE_URL.format(filename=filename)

                if split == "train":
                    train_urls.append(url)
                elif split == "test":
                    test_urls.append(url)
                elif split == "validation":
                    val_urls.append(url)

        train_paths = list(map(Path, dl_manager.download(sorted(train_urls))))
        test_paths = list(map(Path, dl_manager.download(sorted(test_urls))))
        val_paths = list(map(Path, dl_manager.download(sorted(val_urls))))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_paths": train_paths},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_paths": test_paths},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_paths": val_paths},
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
