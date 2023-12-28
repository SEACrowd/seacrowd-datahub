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
from conllu import TokenList

from seacrowd.utils import schemas
from seacrowd.utils.common_parser import load_ud_data, load_ud_data_as_seacrowd_kb
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks


_CITATION = ""

_LANGUAGES = ["ind", "vie"]
_LOCAL = False

_DATASETNAME = "ud-v2.13"

_SUBSETS = {"id_gsd" : "UD_Indonesian-GSD", 
            "id_csui": "UD_Indonesian-CSUI", 
            "id_pud" : "UD_Indonesian-PUD", 
            "vi_vtb": "UD_Vietnamese-VTB"}

_DESCRIPTION = """\
Universal Dependencies (UD) is a project that is developing cross-linguistically consistent treebank annotation
 for many languages, with the goal of facilitating multilingual parser development, cross-lingual learning, and 
 parsing research from a language typology perspective. The annotation scheme is based on an evolution of (universal)
   Stanford dependencies (de Marneffe et al., 2006, 2008, 2014), Google universal part-of-speech tags 
   (Petrov et al., 2012), and the Interset interlingua for morphosyntactic tagsets (Zeman, 2008). 
   The general philosophy is to provide a universal inventory of categories and guidelines to facilitate consistent
     annotation of similar constructions across languages, while allowing language-specific extensions when necessary.
"""

_HOMEPAGE = "https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5287"

_LICENSE = "Apache license 2.0 (apache-2.0)"

_URLS = {
    "ud-v2.12": "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5150/ud-treebanks-v2.12.tgz?sequence=1&isAllowed=y",
    "ud-v2.13": "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5287/ud-treebanks-v2.13.tgz?sequence=1&isAllowed=y"
}

_SUPPORTED_TASKS = [Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"





class UDDataset(datasets.GeneratorBasedBuilder):

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
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd Seq Label schema",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

    def _info(self) -> datasets.DatasetInfo:
        self.config.schema = "seacrowd_seq_label"
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    # metadata
                    "sent_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
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

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        # data_path = dl_manager.download(urls)

        return []
    

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """instance tuple generated in the form (key, labels)"""

        dataset = list(
            load_ud_data(
                filepath,
                filter_kwargs={"id": lambda i: isinstance(i, int)},
            )
        )

        if self.config.schema == "source":
            pass

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


if __name__ == "__main__":
    data = datasets.load_dataset(__file__)
    print("xx")
