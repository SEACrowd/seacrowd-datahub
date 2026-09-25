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
IndoACD (Indonesian Adult Content Detection) is a collection of Indonesian tweets labeled as
either containing adult content (pornography and adult-oriented material) or non-adult content.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{hidayatullah2023adult,
  author    = {Hidayatullah, Ahmad Fathan and Apong, Rosyzie Anna and Ching Lai, Daphne Teck and Qazi, Atika},
  title     = {Adult Content Detection on Indonesian Tweets by Fine-tuning Transformer-based Models},
  booktitle = {2023 6th International Conference on Applied Computational Intelligence in Information Systems (ACIIS)},
  address   = {Bandar Seri Begawan, Brunei Darussalam},
  pages     = {1-6},
  year      = {2023},
  doi       = {10.1109/ACIIS59385.2023.10367283}
}
"""

_DATASETNAME = "indoacd"

_DESCRIPTION = """\
IndoACD is a collection of Indonesian tweets that can be classified as adult or non-adult content.
Adult content refers to tweets about pornography and adult-oriented material, while non-adult content
includes tweets with normal and safe content.
"""

_HOMEPAGE = "https://github.com/fathanick/Adult-content-dataset-twitter"

_LANGUAGES = ["ind"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    "train": "https://raw.githubusercontent.com/fathanick/Adult-content-dataset-twitter/main/train_set.xlsx",
    "validation": "https://raw.githubusercontent.com/fathanick/Adult-content-dataset-twitter/main/validation_set.xlsx",
    "test": "https://raw.githubusercontent.com/fathanick/Adult-content-dataset-twitter/main/test_set.xlsx",
}

_SUPPORTED_TASKS = [Tasks.DOMAIN_KNOWLEDGE_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_LABELS = ["Non_adult_content", "Adult_content"]


class IndoACDDataset(datasets.GeneratorBasedBuilder):
    """IndoACD is a binary classification dataset of Indonesian tweets labeled as adult or non-adult content."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="indoacd_source",
            version=SOURCE_VERSION,
            description="IndoACD source schema",
            schema="source",
            subset_id="indoacd",
        ),
        SEACrowdConfig(
            name="indoacd_seacrowd_text",
            version=SEACROWD_VERSION,
            description="IndoACD SEACrowd schema",
            schema="seacrowd_text",
            subset_id="indoacd",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indoacd_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "tweet": datasets.Value("string"),
                    "label": datasets.Value("int64"),
                    "category": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(_LABELS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data_paths = {split: Path(dl_manager.download(url)) for split, url in _URLS.items()}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_paths["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_paths["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_paths["test"]},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        df = pd.read_excel(filepath)

        if self.config.schema == "source":
            for i, row in df.iterrows():
                yield i, {
                    "tweet": row["tweet"],
                    "label": row["label"],
                    "category": row["category"],
                }
        elif self.config.schema == "seacrowd_text":
            for i, row in df.iterrows():
                yield i, {
                    "id": str(i),
                    "text": row["tweet"],
                    "label": row["category"],
                }


if __name__ == "__main__":
    datasets.load_dataset(__file__)
