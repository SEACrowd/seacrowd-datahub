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

_CITATION = """
@techreport{dulaynag2021filaoa,
  author    = {Dulay, Katrina May and Nag, Somali},
  title     = {TalkTogether Age-of-Acquisition Word Lists for 885 Kannada and Filipino Words},
  institution = {TalkTogether},
  year      = {2021},
  type      = {Technical Report},
  url       = {https://osf.io/gnjmr},
  doi       = {10.17605/OSF.IO/3ZDFN},
}
"""

_LOCAL = False
_LANGUAGES = ["fil", "eng"]
_DATASETNAME = "filipino_words_aoa"
_DESCRIPTION = """\
The dataset contains 885 Filipino words derived from an age-of-acquisition participant study. The words are derived child-directed corpora
using pre-specified linguistic criteria. Each word in the corpora contains information about its meaning, part-of-speech (POS), age band,
morpheme count, syllable length, phoneme length, and the level of book it was derived from. The dataset can be used for lexical complexity
prediction, lexical simplification, and readability assessment research.
"""

_HOMEPAGE = "https://osf.io/3zdfn/"
_LICENSE = Licenses.CC_BY_SA_4_0.value
_URL = "https://osf.io/download/j42g7/"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class FilipinoWordsAOADataset(datasets.GeneratorBasedBuilder):
    """
    Dataset of Filipino words, their English meanings, and their part-of-speech tag
    obtained from an age-of-acquisition study.
    """

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
            name=f"{_DATASETNAME}_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SeaCrowd text-to-text schema",
            schema="seacrowd_t2t",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "word": datasets.Value("string"),
                    "meaning": datasets.Value("string"),
                    "POS_tag": datasets.Value("string"),
                    "mean_AoA": datasets.Value("float64"),
                    "mean_AoA_ageband": datasets.Value("string"),
                    "morpheme_count": datasets.Value("int64"),
                    "syllable_length": datasets.Value("int64"),
                    "phoneme_length": datasets.Value("int64"),
                    "book_ageband": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        filepath = dl_manager.download(_URL)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": filepath})]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_excel(filepath, index_col=None)
        for index, row in df.iterrows():
            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == "seacrowd_t2t":
                example = {
                    "id": str(index),
                    "text_1": row["word"],
                    "text_2": row["meaning"],
                    "text_1_name": "fil",
                    "text_2_name": "eng",
                }
            yield index, example