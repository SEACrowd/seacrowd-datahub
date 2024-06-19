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

from seacrowd.utils import schemas
from seacrowd.utils.common_parser import load_ud_data, load_ud_data_as_seacrowd_kb
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@unpublished{Alfina2023,
    author = {Alfina, Ika and Yuliawati, Arlisa and Tanaya, Dipta and Dinakaramani, Arawinda and Zeman, Daniel},
    title = {{A Gold Standard Dataset for Javanese Tokenization, POS Tagging, Morphological Feature Tagging, and Dependency Parsing}},
    year = {2023}
}
"""

_DATASETNAME = "ud_jv_csui"

_DESCRIPTION = """\
UD Javanese-CSUI is a dependency treebank in Javanese, a regional language in Indonesia with more than 68 million users.
It was developed by Alfina et al. from the Faculty of Computer Science, Universitas Indonesia.
The newest version has 1000 sentences and 14K words with manual annotation.

The sentences use the Latin script and do not use the original writing system of Javanese (Hanacaraka).

The original sentences were taken from several resources:
1. Javanese reference grammar books (125 sents)
2. OPUS, especially from the Javanese section of the WikiMatrix v1 corpus (150 sents)
3. Online news (Solopos) (725 sents)

Javanese has several language levels (register), such as Ngoko, Krama, Krama Inggil, and Krama Andhap.
In this treebank, the sentences predominantly use Ngoko words, some of which use Krama words.
"""

_HOMEPAGE = "https://github.com/UniversalDependencies/UD_Javanese-CSUI"

_LANGUAGES = ["jav"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/UniversalDependencies/UD_Javanese-CSUI/master/jv_csui-ud-test.conllu",
}

_SUPPORTED_TASKS = [Tasks.DEPENDENCY_PARSING, Tasks.MACHINE_TRANSLATION, Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


def _resolve_misannotation_(dataset):
    """Resolving mis-annotation in the raw data. In-place."""
    for d in dataset:
        # Metadata's typos
        if d["sent_id"] == "opus-wiki-5":  # From the raw file. Thrown-away during parsing due to no field name.
            d.setdefault("text_en", "Prior to World War II, 14 commercial and 12 public radios could be operated in France.")
        if d["sent_id"] == "wedhawati-2001-66":  # empty string
            d.setdefault("text_en", "Reading can expand knowledge.")
        if d["sent_id"] == "opus-wiki-72":
            d["text_en"] = d.pop("text-en")  # metadata mis-titled
        if d["sent_id"] == "opus-wiki-27":
            d["text_id"] = d.pop("tex_id")  # metadata mis-titled

        # Problems on the annotation itself
        if d["sent_id"] == "solopos-2022-42":  # POS tag is also wrong. Proceed with caution.
            d["form"][1] = d["form"][1].replace("tresnane", "tresna")  # tresna + e
        if d["sent_id"] == "solopos-2022-93":  # wrong annot
            d["form"][10] = d["form"][10].replace("tengene", "tengen")  # tengen + e
        if d["sent_id"] == "solopos-2022-506":  # annotation inconsistency on occurrences of word "sedina"
            d["form"][3] = d["form"][3].replace("siji", "se")
        if d["sent_id"] == "solopos-2022-711":  # annotation inconsistency on the word "rasah" from "ra" and "usah"
            d["form"][11] = d["form"][11].replace("usah", "sah")

    return dataset


class UdJvCsuiDataset(datasets.GeneratorBasedBuilder):
    """Treebank of Javanese comprises 1030 sentences from 14K words with manual annotation"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    # source: https://universaldependencies.org/u/pos/
    UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_kb",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd KB schema",
            schema="seacrowd_kb",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd Text-to-Text schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd Seq Label schema",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    # metadata
                    "sent_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_id": datasets.Value("string"),
                    "text_en": datasets.Value("string"),
                    # tokens
                    "id": [datasets.Value("string")],
                    "form": [datasets.Value("string")],
                    "lemma": [datasets.Value("string")],
                    "upos": [datasets.Value("string")],
                    "xpos": [datasets.Value("string")],
                    "feats": [datasets.Value("string")],
                    "head": [datasets.Value("string")],
                    "deprel": [datasets.Value("string")],
                    "deps": [datasets.Value("string")],
                    "misc": [datasets.Value("string")],
                }
            )
        elif self.config.schema == "seacrowd_kb":
            features = schemas.kb_features

        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(self.UPOS_TAGS)

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_path = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,  # https://github.com/UniversalDependencies/UD_Javanese-CSUI#split
                gen_kwargs={"filepath": data_path},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        # Note from hudi_f:
        #   Other than 3 sentences with multi-span of length 3, the data format seems fine.
        #   Thus, it is safe to ignore the assertion. (as of 2024/02/14)
        dataset = list(
            load_ud_data(
                filepath,
                filter_kwargs={"id": lambda i: isinstance(i, int)},
                # assert_fn=assert_multispan_range_is_one
            )
        )
        _resolve_misannotation_(dataset)

        for d in dataset:
            if "text_id" not in d or "text_en" not in d:
                print(d)

        if self.config.schema == "source":
            pass

        elif self.config.schema == "seacrowd_kb":
            dataset = load_ud_data_as_seacrowd_kb(
                filepath,
                dataset,
                morph_exceptions=[
                    # Exceptions due to inconsistencies in the raw data annotation
                    ("ne", "e"),
                    ("nipun", "ipun"),
                    ("me", "e"),  # occurrence word: "Esemme" = "Esem" + "e". original text has double 'm'.
                ],
            )

        elif self.config.schema == "seacrowd_t2t":
            dataset = list(
                map(
                    lambda d: {
                        "id": d["sent_id"],
                        "text_1": d["text"],
                        "text_2": d["text_id"],
                        "text_1_name": "jav",
                        "text_2_name": "ind",
                    },
                    dataset,
                )
            )

        elif self.config.schema == "seacrowd_seq_label":
            dataset = list(
                map(
                    lambda d: {
                        "id": d["sent_id"],
                        "tokens": d["form"],
                        "labels": d["upos"],
                    },
                    dataset,
                )
            )

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        for key, example in enumerate(dataset):
            yield key, example
