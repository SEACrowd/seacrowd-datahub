# coding=utf-8


from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{kratochvil-morgado-da-costa-2022-abui,
title = "{A}bui {W}ordnet: Using a Toolbox Dictionary to develop a wordnet for a low-resource language",
author = "Kratochvil, Frantisek  and
  Morgado da Costa, Lu{\'}s",
editor = "Serikov, Oleg  and
  Voloshina, Ekaterina  and
  Postnikova, Anna  and
  Klyachko, Elena  and
  Neminova, Ekaterina  and
  Vylomova, Ekaterina  and
  Shavrina, Tatiana  and
  Ferrand, Eric Le  and
  Malykh, Valentin  and
  Tyers, Francis  and
  Arkhangelskiy, Timofey  and
  Mikhailov, Vladislav  and
  Fenogenova, Alena",
booktitle = "Proceedings of the first workshop on NLP applications to field linguistics",
month = oct,
year = "2022",
address = "Gyeongju, Republic of Korea",
publisher = "International Conference on Computational Linguistics",
url = "https://aclanthology.org/2022.fieldmatters-1.7",
pages = "54--63",
abstract = "This paper describes a procedure to link a Toolbox dictionary of a low-resource language to correct
synsets, generating a new wordnet. We introduce a bootstrapping technique utilising the information in the gloss
fields (English, national, and regional) to generate sense candidates using a naive algorithm based on
multilingual sense intersection. We show that this technique is quite effective when glosses are available in
more than one language. Our technique complements the previous work by Rosman et al. (2014) which linked the
SIL Semantic Domains to wordnet senses. Through this work we have created a small, fully hand-checked wordnet
for Abui, containing over 1,400 concepts and 3,600 senses.",
}
"""
_DATASETNAME = "abui_wordnet"
_DESCRIPTION = """\
A small fully hand-checked wordnet for Abui, containing over 1,400 concepts and 3,600 senses, is created. A
bootstrapping technique is introduced to utilise the information in the gloss fields (English, national, and regional)
to generate sense candidates using a naive algorithm based on multilingual sense intersection.
"""

_HOMEPAGE = "https://github.com/fanacek/abuiwn"
_LANGUAGES = ["abz"]
_LICENSE = Licenses.CC_BY_4_0.value
_LOCAL = False
_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/fanacek/abuiwn/main/abwn_lmf.tsv",
}

_SUPPORTED_TASKS = [Tasks.WORD_ANALOGY]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class AbuiwordnetDataset(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=_DESCRIPTION,
            schema="source",
            subset_id="abui_wordnet",
        ),
        # SEACrowdConfig(
        #     name="abui_wordnet_seacrowd_ww",
        #     version=SEACROWD_VERSION,
        #     description="abuiw SEACrowd schema",
        #     schema="seacrowd_a",
        #     subset_id="abui_wordnet",
        # ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        features = None
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sense": datasets.Value("string"),
                    "pos": datasets.Value("string"),
                    "lang": datasets.Value("string"),
                    "lemma": datasets.Value("string"),
                    "form": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_pair":
            features = schemas.pairs_features
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name="senses",
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        with open(filepath, "r") as filein:
            data_instances = [inst.strip("\n").split("\t") for inst in filein.readlines()]
        if self.config.schema == "source":
            for idx, example in enumerate(data_instances):
                sense = example[0]
                pos = example[0][-1]
                lang = example[1]
                lemma = example[2]
                form = "" if len(example) == 3 else example[3]
                yield idx, {
                    "sense": sense,
                    "pos": pos,
                    "lang": lang,
                    "lemma": lemma,
                    "form": form,
                }
        # elif self.config.schema == "seacrowd_pair":
        #
