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

import pandas as pd

import datasets

from seacrowds.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@techreport{dulay2021talktogether,
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
_LANGUAGES = ["fil"]
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

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION, Tasks.POS_TAGGING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

class FilipinoWordsAOADataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    pos_labels = ["adjective", "adverb", "noun", "pronoun", "verb"]
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "seq_label"
    
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
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
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.seq_label_features(self.pos_labels)
            
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        filepath = Path(dl_manager.download_and_extract(_URL))
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN),
            gen_kwargs={"filepath": filepath}
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(filepath, index_col=None)
        if self.config.schema == "source":
            for index, row in enumerate(df):
                example = {
                    "index": str(index),
                    "tokens": row["word"],
                    "pos_tags": row["POS_tag"],
                }
                yield index, example

        elif self.config.schema == "seacrowd_[seacrowd_schema_name]":
            for index, row in enumerate(df):
                example = {
                    "id": str(index),
                    "tokens": row["word"],
                    "labels": row["POS_tag"],
                }
                
                yield index, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
