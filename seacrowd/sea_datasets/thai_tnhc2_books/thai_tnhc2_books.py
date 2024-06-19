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
This dataset collects all 353 books from the Thai National Historical Corpus 2 (TNHC2) corpus. The dataset has been cleaned to use text for pretraining models and NLP tasks. The TNHC2 corpus is a Thai old books corpus and all books are copyright expired according to Thai law (50 years after the author's death). More information on this corpus can be found here: https://www.arts.chula.ac.th/chulaseal/tnhc2/.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@dataset{phatthiyaphaibun_2024_10783421,
  author       = {Phatthiyaphaibun, Wannaphong},
  title        = {Thai TNHC2 Books},
  month        = mar,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10783421},
  url          = {https://doi.org/10.5281/zenodo.10783421}
}
"""

_DATASETNAME = "thai_tnhc2_books"

_DESCRIPTION = """\
This dataset collects all 353 books from the Thai National Historical Corpus 2 (TNHC2) corpus. The dataset has been cleaned to use text for pretraining models and NLP tasks. The TNHC2 corpus is a Thai old books corpus and all books are copyright expired according to Thai law (50 years after the author's death). More information on this corpus can be found here: https://www.arts.chula.ac.th/chulaseal/tnhc2/.
"""

_HOMEPAGE = "https://www.arts.chula.ac.th/chulaseal/tnhc2/"

_LANGUAGES = ["tha"]

_LICENSE = Licenses.CC0_1_0.value

_LOCAL = False

_URLS = "https://huggingface.co/datasets/pythainlp/thai-tnhc2-books/resolve/main/data/train-00000-of-00001.parquet?download=true"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

class ThaiTnhc2BooksDataset(datasets.GeneratorBasedBuilder):
    """This dataset collects all 353 books from the Thai National Historical Corpus 2 (TNHC2) corpus. The dataset has been cleaned to use text for pretraining models and NLP tasks. The TNHC2 corpus is a Thai old books corpus and all books are copyright expired according to Thai law (50 years after the author's death). More information on this corpus can be found here: https://www.arts.chula.ac.th/chulaseal/tnhc2/."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_ssp",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({
                "id": datasets.Value("string"),
                "book": datasets.Value("string"),
                "author": datasets.Value("string"),
                "text": datasets.Value("string"),
            })

        elif self.config.schema == "seacrowd_ssp":
            features = schemas.ssp_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_parquet(filepath)

        # Handle multiple books with the same id
        df["id"] = df["id"] + "_" + df.groupby("id").cumcount().astype(str)

        if self.config.schema == "source":
            for i, row in df.iterrows():
                yield i, {
                    "id": row["id"],
                    "book": row["book"],
                    "author": row["author"],
                    "text": row["text"],
                }

        elif self.config.schema == "seacrowd_ssp":
            for i, row in df.iterrows():
                yield i, {
                    "id": row["id"],
                    "text": row["text"],
                }
