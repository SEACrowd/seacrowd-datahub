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
AlloVera is a multilingual database mapping phones (allophones) to the phonemes they
realize, for 14 languages. This loader covers the three SEACrowd-relevant subsets:
Javanese, Tagalog, and Vietnamese.

NOTE: the SEACrowd datasheet lists the task for this dataset as Automatic Speech
Recognition, but AlloVera contains no audio -- only static phone-to-phoneme mapping
tables per language. We map it to Transliteration (seacrowd_t2t) instead, since that
is the schema that actually fits the data (phone -> phoneme string pairs).
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{mortensen-etal-2020-allovera,
    title = "{A}llo{V}era: A Multilingual Allophone Database",
    author = "Mortensen, David R.  and
      Li, Xinjian  and
      Littell, Patrick  and
      Michaud, Alexis  and
      Rijhwani, Shruti  and
      Anastasopoulos, Antonios  and
      Black, Alan W  and
      Metze, Florian  and
      Neubig, Graham",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.656/",
    pages = "5329--5336",
}
"""

_DATASETNAME = "allovera"

_DESCRIPTION = """\
AlloVera provides mappings from 218 allophones to phonemes for 14 languages. Phonemes
are contrastive phonological units, and allophones are their various concrete
realizations, which are predictable from phonological context. This loader covers the
Javanese, Tagalog, and Vietnamese subsets.
"""

_HOMEPAGE = "https://github.com/dmort27/allovera"

_LANGUAGES = ["jav", "tgl", "vie"]

_LICENSE = Licenses.MIT.value

_LOCAL = False

_SUBSETS = ["jav", "tgl", "vie"]

_URLS = {subset: f"https://raw.githubusercontent.com/dmort27/allovera/master/data/json/{subset}.json" for subset in _SUBSETS}

_SUPPORTED_TASKS = [Tasks.TRANSLITERATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class AlloVeraDataset(datasets.GeneratorBasedBuilder):
    """AlloVera is a multilingual database of phone-to-phoneme (allophone) mappings."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in _SUBSETS:
        BUILDER_CONFIGS += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} {subset} source schema",
                schema="source",
                subset_id=subset,
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset}_seacrowd_t2t",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {subset} SEACrowd schema",
                schema="seacrowd_t2t",
                subset_id=subset,
            ),
        ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_jav_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "iso": datasets.Value("string"),
                    "glottocodes": datasets.Sequence(datasets.Value("string")),
                    "primary_src": datasets.Value("string"),
                    "secondary_srcs": datasets.Sequence(datasets.Value("string")),
                    "epitran": datasets.Value("string"),
                    "mappings": [
                        {
                            "phone": datasets.Value("string"),
                            "phoneme": datasets.Value("string"),
                            "environment": datasets.Value("string"),
                        }
                    ],
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
        subset = self.config.subset_id
        filepath = Path(dl_manager.download(_URLS[subset]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": filepath},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if self.config.schema == "source":
            yield 0, {
                "iso": data["iso"],
                "glottocodes": data["glottocodes"],
                "primary_src": data["primary src"],
                "secondary_srcs": data["secondary srcs"],
                "epitran": data["epitran"],
                "mappings": [
                    {
                        "phone": mapping["phone"],
                        "phoneme": mapping["phoneme"],
                        "environment": mapping.get("environment", ""),
                    }
                    for mapping in data["mappings"]
                ],
            }
        elif self.config.schema == "seacrowd_t2t":
            for i, mapping in enumerate(data["mappings"]):
                yield i, {
                    "id": str(i),
                    "text_1": mapping["phone"],
                    "text_2": mapping["phoneme"],
                    "text_1_name": "phone",
                    "text_2_name": "phoneme",
                }


if __name__ == "__main__":
    datasets.load_dataset(__file__)
