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

import os
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@article{gonzales_corpus_2021,
    title = {The {Corpus} of {Singapore} {English} {Messages} ({CoSEM})},
    issn = {0883-2919, 1467-971X},
    url = {https://onlinelibrary.wiley.com/doi/10.1111/weng.12534},
    doi = {10.1111/weng.12534},
    language = {en},
    urldate = {2022-02-19},
    journal = {World Englishes},
    author = {Gonzales, Wilkinson Daniel Wong and Hiramoto, Mie and R. E. Leimgruber, Jakob and Lim, Jun Jie},
    month = feb,
    year = {2021},
}

@article{leimgruber_ethnic_2020,
    title = {Ethnic and gender variation in the use of {Colloquial} {Singapore} {English} discourse particles},
    copyright = {CC0 1.0 Universal Public Domain Dedication},
    journal = {English Language and Linguistics},
    author = {Leimgruber, Jakob and Lim, Jun Jie and Gonzales, Wilkinson Daniel Wong and Hiramoto, Mie},
    year = {2020},
}

@incollection{hiramoto_malay_2022,
    series = {{LINCOM} {Studies} in {English} {Linguistics} 24},
    title = {From {Malay} to {Colloquial} {Singapore} {English}: {A} case study of sentence-final particle sia},
    booktitle = {World {Englishes} and creole languages today existing paradigms and current trends in action},
    publisher = {Lincom Europa},
    author = {Hiramoto, Mie and Gonzales, Wilkinson Daniel Wong and Leimgruber, Jakob and Lim, Jun Jie and Choo, Jessica Xue Ming},
    editor = {Ngefac, Aloysius and Wolf, Hans-Georg and Hoffman, Thomas},
    year = {2022},
    pages = {117--130},
    file = {Hiramoto et al. - 2022 - From Malay to Colloquial Singapore English A case.pdf:/Users/wdwg/Zotero/storage/753EKVDX/Hiramoto et al. - 2022 - From Malay to Colloquial Singapore English A case.pdf:application/pdf},
}

@article{gonzales_is_2022,
    title = {\textit{{Is} it} in {Colloquial} {Singapore} {English}: {What} variation can tell us about its conventions and
    development},
    copyright = {CC0 1.0 Universal Public Domain Dedication},
    issn = {0266-0784, 1474-0567},
    shorttitle = {\textit{{Is} it} in {Colloquial} {Singapore} {English}},
    url = {https://www.cambridge.org/core/product/identifier/S0266078422000141/type/journal_article},
    doi = {10.1017/S0266078422000141},
    abstract = {Colloquial Singapore English (CSE, commonly known as Singlish) is a linguistic variety used in Singapore, a Southeast
    Asian nation home to three major ethnic groups: the Chinese (74.35% of the citizen and permanent resident population), the
    Malays (13.43%), and the Indians (9%) (Singapore Department of Statistics, 2019). It is one of the best known post-colonial
    varieties of English and has been documented since the emergence of the field of world Englishes (e.g., Greenbaum, 1988; Richards
    & Tay, 1977). Linguistically, the grammar and lexicon of CSE are systematically imported from other non-English languages used
    in the island nation (Leimgruber, 2011). From a creolist perspective, it can be viewed as an English-lexifier creole that
    contains influences from Sinitic languages such as Hokkien, Cantonese and Mandarin, as well as Malay, Tamil and other varieties
    in the Singapore language ecology (McWhorter, 2007; Platt, 1975). Several distinct features across various levels of language
    have been investigated in CSE, including phonetics (Starr & Balasubramaniam, 2019), morphosyntax (Bao, 2010; Bao & Wee, 1999
    semantics (Hiramoto & Sato, 2012), and pragmatics (Hiramoto, 2012; Leimgruber, 2016; Lim, 2007).},
    language = {en},
    urldate = {2022-08-08},
    journal = {English Today},
    author = {Gonzales, Wilkinson Daniel Wong and Hiramoto, Mie and Leimgruber, Jakob R. E. and Lim, Jun Jie},
    month = jun,
    year = {2022},
    pages = {1--14},
    file = {Gonzales et al. - 2022 - Is it in Colloquial Singapore English What.pdf:/Users/wdwg/Zotero/storage/59E8FXJT/Gonzales et al. - 2022 - Is it in Colloquial Singapore English What.pdf:application/pdf},
}
"""

_DATASETNAME = "cosem"

_DESCRIPTION = """\
The CoSEM dataset consists of over 900,000 lines of online messages from the messaging platform WhatsApp collected from personal chat
logs of students enrolled in an advanced sociolinguistics class from the National University of Singapore. Messages collected were
from 2016 to 2019. The dataset is in .txt format, where each line of utterance is tagged with a unique identifier that includes its
metadata such as line number, year message was sent, and age and nationality of sender.
"""

_HOMEPAGE = "https://github.com/wdwgonzales/CoSEM/blob/main/Corpus/COSEM_v4_publicrelease_SEP172023.zip"

_LANGUAGES = ["eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC0_1_0.value

_LOCAL = False

_URLS = {_DATASETNAME: "https://github.com/wdwgonzales/CoSEM/raw/main/Corpus/COSEM_v4_publicrelease_SEP172023.zip"}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class MyWsl2023(datasets.GeneratorBasedBuilder):
    """The CoSEM dataset consists of over 900,000 lines of online messages from the messaging platform WhatsApp collected from
    personal chat logs of students enrolled in an advanced sociolinguistics class from the National University of Singapore."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    subset_id = _DATASETNAME

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{subset_id}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=subset_id,
        )
    ]

    seacrowd_schema_config: list[SEACrowdConfig] = []

    for seacrowd_schema in _SUPPORTED_SCHEMA_STRINGS:

        seacrowd_schema_config.append(
            SEACrowdConfig(
                name=f"{subset_id}_{seacrowd_schema}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {seacrowd_schema} schema",
                schema=f"{seacrowd_schema}",
                subset_id=subset_id,
            )
        )

    BUILDER_CONFIGS.extend(seacrowd_schema_config)

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SELF_SUPERVISED_PRETRAINING]).lower()}":
            features = schemas.ssp_features

        else:
            raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        split_generators = []

        path = dl_manager.download_and_extract(_URLS[_DATASETNAME])

        split_generators.append(
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(path, "COSEM_v4_publicrelease_SEP172023"),
                },
            )
        )

        return split_generators

    def _generate_examples(self, path: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        idx = 0
        files = os.listdir(path)
        file_paths = [os.path.join(path, file) for file in files]

        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    blocks = line.split("\t")

                    if len(blocks) < 2:
                        continue

                    id = idx
                    text = blocks[1]

                    if self.config.schema == "source" or self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.SELF_SUPERVISED_PRETRAINING]).lower()}":
                        yield idx, {
                            "id": id,
                            "text": text,
                        }
                        idx += 1

                    else:
                        raise ValueError(f"Invalid config: {self.config.name}")
