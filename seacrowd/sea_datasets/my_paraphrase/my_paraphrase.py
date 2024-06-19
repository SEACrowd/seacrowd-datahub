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

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{htay2022deep,
  title={Deep Siamese Neural Network Vs Random Forest for Myanmar Language Paraphrase Classification},
  author={Htay, Myint Myint and Thu, Ye Kyaw and Thant, Hnin Aye and Supnithi, Thepchai},
  journal={Journal of Intelligent Informatics and Smart Technology},
  year={2022}
}
"""

_DATASETNAME = "my_paraphrase"

_DESCRIPTION = """\
The myParaphrase corpus is intended for the task of assessing whether pairs of Burmese sentences exhibit similar meanings \
or are paraphrases. It encompasses 40461 pairs for training, along with 1000 pairs for an open test and an additional 1000 pairs \
for a closed test. If a pair of sentences in Burmese is considered a paraphrase, it is labeled with "1"; if not, they receive a label of "0."
"""

_HOMEPAGE = "https://github.com/ye-kyaw-thu/myParaphrase"

_LANGUAGES = ["mya"]

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value
_LOCAL = False

_URLS = {
    _DATASETNAME: [
        "https://github.com/ye-kyaw-thu/myParaphrase/raw/main/corpus/ver1.0/csv-qqp/train.csv",
        "https://github.com/ye-kyaw-thu/myParaphrase/raw/main/corpus/ver1.0/csv-qqp/open-test.final.manual.csv",
        "https://github.com/ye-kyaw-thu/myParaphrase/raw/main/corpus/ver1.0/csv-qqp/closed-test.csv",
    ],
}

_SUPPORTED_TASKS = [Tasks.PARAPHRASING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"
_TAGS = [0, 1]


class MyParaphraseDataset(datasets.GeneratorBasedBuilder):
    """The "myParaphrase" corpus is a Burmese dataset used for paraphrase identification. \
    It includes 40,461 training pairs and 2,000 test pairs. Pairs are labeled "1" for paraphrases and "0" otherwise."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "t2t"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",  # source
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_paraphrase",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",  # schema
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_paraphrase",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_non_paraphrase_source",  # source
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_non_paraphrase",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_non_paraphrase_seacrowd_{SEACROWD_SCHEMA_NAME}",  # schema
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_non_paraphrase",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_all_source",  # source
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_all",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_all_seacrowd_{SEACROWD_SCHEMA_NAME}",  # schema
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_all",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema.endswith("_source"):
            features = datasets.Features({"id": datasets.Value("int32"), "paraphrase1": datasets.Value("string"), "paraphrase2": datasets.Value("string"), "is_paraphrase": datasets.Value("int32")})

        elif self.config.schema.endswith(self.SEACROWD_SCHEMA_NAME):
            features = schemas.text2text_features
        
        else:
            raise ValueError

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        train = dl_manager.download(urls[0])
        open_test = dl_manager.download(urls[1])
        closed_test = dl_manager.download(urls[2])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": train,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": closed_test,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": open_test,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        columns = ["id", "paraphrase1", "paraphrase2", "is_paraphrase"]
        dataset = pd.read_csv(filepath, header=None)
        dataset.columns = columns
        dataset = dataset.dropna()

        dataset["is_paraphrase"] = dataset["is_paraphrase"].astype(int)

        if self.config.schema in [
            "paraphrase_source",
            "non_paraphrase_source",
            "all_source",
            # "source"
        ]:
            for i, row in dataset.iterrows():
                yield i, {"id": i, "paraphrase1": row["paraphrase1"], "paraphrase2": row["paraphrase2"], "is_paraphrase": row["is_paraphrase"]}

        elif self.config.schema == f"seacrowd_paraphrase_{self.SEACROWD_SCHEMA_NAME}":
            for i, row in dataset[dataset["is_paraphrase"] == 1].iterrows():
                yield i, {"id": i, "text_1": row["paraphrase1"], "text_2": row["paraphrase2"], "text_1_name": "anchor_text", "text_2_name": "paraphrased_text"}

        elif self.config.schema == f"seacrowd_non_paraphrase_{self.SEACROWD_SCHEMA_NAME}":
            for i, row in dataset[dataset["is_paraphrase"] == 0].iterrows():
                yield i, {"id": i, "text_1": row["paraphrase1"], "text_2": row["paraphrase2"], "text_1_name": "anchor_text", "text_2_name": "non_paraphrased_text"}

        elif self.config.schema == f"seacrowd_all_{self.SEACROWD_SCHEMA_NAME}":
            for i, row in dataset.iterrows():
                yield i, {"id": i, "text_1": row["paraphrase1"], "text_2": row["paraphrase2"], "text_1_name": "anchor_text", "text_2_name": "paraphrased_text" if row["is_paraphrase"] else "non_paraphrased_text"}

        else:
            raise ValueError