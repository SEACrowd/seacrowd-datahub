from pathlib import Path
from typing import List, Tuple

import datasets

from seacrowd.sea_datasets.lio_and_central_flores import processing
from seacrowd.sea_datasets.lio_and_central_flores.path_url import _URLS_DICT
from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{alexthesis2018,
  author    = {Alexander Elias},
  title     = {Lio and the Central Flores languages},
  year      = {2018},
  month     = {November},
  address   = {Rapenburg 70, 2311 EZ Leiden},
  school    = {Universiteit Leiden},
  url       = {https://studenttheses.universiteitleiden.nl/handle/1887/69452},
  note      = {Research Master's thesis},
}
"""

_DATASETNAME = "lio_and_central_flores"
_DESCRIPTION = """This dataset is a collection of language resources of Li'o, Ende, Nage, and
So'a which are collected in Ende, Flores, Eastern Nusa Tenggara. This dataset
is the dataset from the research MA thesis by Alexander Elias. Title: Lio and the Central Flores languages
"""
_HOMEPAGE = "https://archive.mpi.nl/tla/islandora/search/alexander%20elias?type=dismax&islandora_solr_search_navigation=0&f%5B0%5D=cmd.Contributor%3A%22Alexander%5C%20Elias%22"
_LICENSE = Licenses.UNKNOWN.value
_LANGUAGES = ["end", "ljl", "nxe", "eng"]
LANGUAGES_TO_FILENAME_MAP = {
    "end": "ENDE",
    "nxe": "NAGE",
    "ljl": "LIO",
}
_LOCAL = False

_URLS = _URLS_DICT

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class LioAndCentralFloresDataset(datasets.GeneratorBasedBuilder):
    """This dataset is a collection of language resources of Li'o, Ende, Nage, and So'a"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "t2t"

    BUILDER_CONFIGS = [
        # We only use source schema here for nage ("nxe") and eng because nage dataset only contain wordlist
        # For "nxe" , include a separate configuration to handle word lists. It will be return nage only word list
        SEACrowdConfig(name=f"{_DATASETNAME}_nxe_wordlist_source", version=SOURCE_VERSION, description=f"{_DATASETNAME} source schema", schema="source", subset_id=f"{_DATASETNAME}_nxe"),
        # Additionally, include a configuration for English word lists in "nxe" datasets. It will be return eng only word corresponding to nage wordlist
        SEACrowdConfig(name=f"{_DATASETNAME}_eng_wordlist_source", version=SOURCE_VERSION, description=f"{_DATASETNAME} source schema", schema="source", subset_id=f"{_DATASETNAME}_eng"),
    ]

    # For other languages, except "nxe", use a standard source & seacrowd schema configuration
    subset_names = sorted([f"{_DATASETNAME}_{lang}_eng" for lang in _LANGUAGES[:-2]]) + sorted([f"{_DATASETNAME}_eng_{lang}" for lang in _LANGUAGES[:-2]])

    for name in subset_names:
        # source schema
        source_config = SEACrowdConfig(name=f"{name}_source", version=SOURCE_VERSION, description=f"{_DATASETNAME} source schema", schema="source", subset_id=name)
        BUILDER_CONFIGS.append(source_config)

        # seacrowd_t2t schema
        seacrowd_config = SEACrowdConfig(name=f"{name}_seacrowd_{SEACROWD_SCHEMA_NAME}", version=SEACROWD_VERSION, description=f"{_DATASETNAME} SEACrowd schema", schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}", subset_id=name)
        BUILDER_CONFIGS.append(seacrowd_config)


    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            if "wordlist" in self.config.name:
                features = datasets.Features({"id": datasets.Value("string"), "word": datasets.Value("string")})
            else:
                features = datasets.Features({"source_sentence": datasets.Value("string"), "target_sentence": datasets.Value("string"), "source_lang": datasets.Value("string"), "target_lang": datasets.Value("string")})
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            if "nxe" not in self.config.name:
                features = schemas.text2text_features
        else:
            raise ValueError("Invalid config schema")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        dset_lang = None
        for lang in _LANGUAGES[:-1]: # except eng because it exists in all subset names
            if lang in self.config.name:
                dset_lang = lang
                break
        if dset_lang is None:
            raise ValueError("Invalid language name")

        filepath = {k: v["text_path"] for k, v in _URLS[LANGUAGES_TO_FILENAME_MAP[dset_lang]].items()}
        paths = dl_manager.download(filepath)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": paths,
                    "lang_1": self.config.name.split("_")[4],
                    "lang_2": self.config.name.split("_")[5]}
            )
        ]

    def _generate_examples(self, filepath: Path, lang_1: str, lang_2: str):
        """Yields examples as (key, example) tuples."""

        if "wordlist" in self.config.name:
            if "nxe" in self.config.name: # only nxe
                _, words = self._get_word_(filepath)
            else: # only eng
                words, _ = self._get_word_(filepath)

            for item in words:
                for idx, word in enumerate(item):
                    row = {"id": str(idx), "word": word}
                    yield idx, row
        else:
            source_data, target_data = self._get_sentence_(filepath)
            for idx, (eng_text, other_text) in enumerate(zip(source_data, target_data)):
                if self.config.schema == "source":
                    if lang_1 == "eng":
                        example = {
                            "source_sentence": eng_text,
                            "target_sentence": other_text,
                            "source_lang": lang_1,
                            "target_lang": lang_2,
                        }
                    else:
                        example = {
                            "source_sentence": other_text,
                            "target_sentence": eng_text,
                            "source_lang": lang_1,
                            "target_lang": lang_2,
                        }

                elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                    if lang_1 == "eng":
                        example = {
                            "id": str(idx),
                            "text_1": eng_text,
                            "text_2": other_text,
                            "text_1_name": lang_1,
                            "text_2_name": lang_2,
                        }
                    else:
                        example = {
                            "id": str(idx),
                            "text_1": other_text,
                            "text_2": eng_text,
                            "text_1_name": lang_1,
                            "text_2_name": lang_2,
                        }
                yield idx, example

    def _get_sentence_(self, path_dict) -> Tuple[List, List]:
        source_data = []
        target_data = []
        for _, v in path_dict.items():
            with open(v, "r", encoding="utf-8") as f:
                data = f.readlines()
            src, trg = processing.parse_text(data)
            source_data.extend(src)
            target_data.extend(trg)

        return source_data, target_data

    def _get_word_(self, path_dict) -> Tuple[List, List]:
        eng_data, ind_data, nage_data = [], [], []
        for _, v in path_dict.items():
            with open(v, "r", encoding="utf-8") as f:
                data = f.readlines()
            eng_word, ind_word, nage_word = processing.parse_wordlist(data)
            eng_data.append(eng_word)
            ind_data.append(ind_word)
            nage_data.append(nage_word)

        return eng_data, nage_data
