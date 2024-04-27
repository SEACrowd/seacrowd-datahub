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
_LANGUAGES = ["end", "nxe", "ljl"]
LANGUAGES_TO_FILENAME_MAP = {
    "end": "ENDE",
    "nxe": "NAGE",
    "ljl": "LIO",
}
_LOCAL = False

_URLS = _URLS_DICT

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class LioAndCentralFloresDataset(datasets.GeneratorBasedBuilder):
    """This dataset is a collection of language resources of Li'o, Ende, Nage, and So'a"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "t2t"

    dataset_names = sorted([f"{_DATASETNAME}_{lang}" for lang in _LANGUAGES])
    BUILDER_CONFIGS = []
    for name in dataset_names:
        if name.split("_")[4] == "nxe":
            # We only use source schema here for nage ("nxe") because nage dataset only contain wordlist
            # For "nxe" , include a separate configuration to handle word lists. It will be return nage only word list
            source_config = SEACrowdConfig(name=f"{name}_wordlist_source", version=SOURCE_VERSION, description=f"{_DATASETNAME} source schema", schema="source", subset_id=name)
            BUILDER_CONFIGS.append(source_config)
            # Additionally, include a configuration for English word lists in "nxe" datasets. It will be return eng only word corresponding to nage wordlist
            source_config = SEACrowdConfig(name=f"{name}_eng_wordlist_source", version=SOURCE_VERSION, description=f"{_DATASETNAME} source schema", schema="source", subset_id=name)
            BUILDER_CONFIGS.append(source_config)
        else:
            # For other languages, except "nxe", use a standard source schema configuration
            source_config = SEACrowdConfig(name=f"{name}_source", version=SOURCE_VERSION, description=f"{_DATASETNAME} source schema", schema="source", subset_id=name)
            BUILDER_CONFIGS.append(source_config)
            # Additionally, include a SEACrowd schema configuration for other languages
            seacrowd_config = SEACrowdConfig(name=f"{name}_seacrowd_{SEACROWD_SCHEMA_NAME}", version=SEACROWD_VERSION, description=f"{_DATASETNAME} SEACrowd schema", schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}", subset_id=name)
            BUILDER_CONFIGS.append(seacrowd_config)

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            if self.config.name.split("_")[4] == "nxe":
                features = datasets.Features({"id": datasets.Value("string"), "word": datasets.Value("string")})
            else:
                features = datasets.Features({"source_sentence": datasets.Value("string"), "target_sentence": datasets.Value("string"), "source_lang": datasets.Value("string"), "target_lang": datasets.Value("string")})
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            if self.config.name.split("_")[4] != "nxe":
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

        lang = self.config.name.split("_")[4]
        if lang in _LANGUAGES:
            filepath = {k: v["text_path"] for k, v in _URLS[LANGUAGES_TO_FILENAME_MAP[lang]].items()}
            paths = dl_manager.download(filepath)
        else:
            raise ValueError("Invalid language name")

        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": paths, "language": lang})]

    def _generate_examples(self, filepath: Path, language: str):
        """Yields examples as (key, example) tuples."""

        if language == "nxe" and self.config.schema == "source":
            schema_type = "_".join(self.config.name.split("_")[4:6])
            if schema_type == "nxe_eng":
                words, _ = self._get_word_(filepath)
            else:
                _, words = self._get_word_(filepath)

            for item in words:
                for idx, word in enumerate(item):
                    row = {"id": str(idx), "word": word}
                    yield idx, row

        else:
            if language != "nxe":
                source_data, target_data = self._get_sentence_(filepath)
                for idx, (source, target) in enumerate(zip(source_data, target_data)):
                    if self.config.schema == "source":
                        example = {"source_sentence": source, "target_sentence": target, "source_lang": "eng", "target_lang": language}
                    elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                        example = {
                            "id": str(idx),
                            "text_1": source,
                            "text_2": target,
                            "text_1_name": "eng",
                            "text_2_name": language,
                        }
                    yield idx, example
            else:
                raise ValueError("Not found 'nxe sentences")

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
