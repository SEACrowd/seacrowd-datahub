"""
SEA Crowd Data Loader for SEA Wiki.
"""

import json
from itertools import product
from typing import Dict, List, Tuple

import datasets
from datasets import load_dataset
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = r"""
@misc{kudugunta2023madlad400,
      title={MADLAD-400: A Multilingual And Document-Level Large Audited Dataset}, 
      author={Sneha Kudugunta and Isaac Caswell and Biao Zhang and Xavier Garcia and Christopher A. Choquette-Choo and Katherine Lee and Derrick Xin and Aditya Kusupati and Romi Stella and Ankur Bapna and Orhan Firat},
      year={2023},
      eprint={2309.04662},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

logger = datasets.logging.get_logger(__name__)

_LANG_CONFIG = {
    "ace": {"name": "Aceh", "source_subset": "ace"},
    "akb": {"name": "Batak Angkola", "source_subset": "akb"},
    "ban": {"name": "Bali", "source_subset": "ban"},
    "bbc": {"name": "Batak Toba", "source_subset": "bbc"},
    "bew": {"name": "Betawi", "source_subset": "bew"},
    "btx": {"name": "Batak Karo", "source_subset": "btx"},
    "ceb": {"name": "Cebuano", "source_subset": "ceb"},
    "fil": {"name": "Filipino", "source_subset": "fil"},
    "gor": {"name": "Gorontalo", "source_subset": "gor"},
    "hil": {"name": "Hiligaynon", "source_subset": "hil"},
    "iba": {"name": "Iban", "source_subset": "iba"},
    "ilo": {"name": "Ilocano", "source_subset": "ilo"},
    "ind": {"name": "Indonesian", "source_subset": "id"},
    "jav": {"name": "Javanese", "source_subset": "jv"},
    "kac": {"name": "Jingpho", "source_subset": "kac"},
    "khm": {"name": "Khmer", "source_subset": "km"},
    "kxd": {"name": "Brunei", "source_subset": "ms_Arab_BN"},
    "lao": {"name": "Lao", "source_subset": "lo"},
    "mad": {"name": "Madura", "source_subset": "mad"},
    "mak": {"name": "Makasar", "source_subset": "mak"},
    "meo": {"name": "Kedah Malay", "source_subset": "meo"},
    "min": {"name": "Minangkabau", "source_subset": "min"},
    "mkn": {"name": "Kupang Malay", "source_subset": "mkn"},
    "msi": {"name": "Sabah Malay", "source_subset": "msi"},
    "mya": {"name": "Burmese", "source_subset": "my"},
    "nij": {"name": "Ngaju", "source_subset": "nij"},
    "nut": {"name": "Nung", "source_subset": "nut"},
    "pag": {"name": "Pangasinan", "source_subset": "pag"},
    "shn": {"name": "Shan", "source_subset": "shn"},
    "sun": {"name": "Sunda", "source_subset": "su"},
    "tet": {"name": "Tetun", "source_subset": "tet"},
    "tha": {"name": "Thai", "source_subset": "th"},
    "vie": {"name": "Vietnamese", "source_subset": "vi"},
    "war": {"name": "Waray-Waray", "source_subset": "war"},
}

_LOCAL = False
_LANGUAGES = list(_LANG_CONFIG.keys())


_DATASETNAME = "sea_madlad"
_DESCRIPTION = r"""
    SEA MADLAD is a subset of MADLAD-400 (Multilingual Audited Dataset: Low-resource And Document-level), which is a document-level multilingual dataset based on Common Crawl.
    SEA MADLAD only filters the language of the "clean" subset, which covers 36 languages indigenous to SEA from 419 languages in total.
    As a result, some of SEA lang codes aren't available in this version because those belongs to the languages whose decision was to "remove from its clean version" based on MADLAD auditing process.
    MADLAD uses all snapshots of CommonCrawl available as of August 1, 2022.
    The primary advantage of this dataset over similar datasets is that it is more multilingual, it is audited and more highly filtered, and it is document-level.
    The main disadvantage is also its strength -- being more filtered, it may lack the recall needed for some applications.
"""

_HOMEPAGE = "https://huggingface.co/datasets/allenai/MADLAD-400"
_LICENSE = Licenses.CC_BY_4_0.value

# url won't be used since it will implement load_dataset method on HF URL provided
_URL = "https://huggingface.co/datasets/allenai/MADLAD-400"
_REMOTE_HF_REFERENCE = ("/".join(_URL.split("/")[-2:])).lower()

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]


def conform_init_config():
    """Assertion Function for Instantiated Configs"""
    if len(_LANGUAGES) == 0:
        raise AssertionError("No Languages detected from config!")
    if len(CONFIG_SUFFIXES_FOR_TASK) != len(_SUPPORTED_TASKS):
        raise AssertionError("Config prefixes doesn't matched in terms of `len` with `_SUPPORTED_TASKS`!")
    if len(CONFIG_SUFFIXES_FOR_TASK) == 0:
        raise AssertionError("Config prefixes and `_SUPPORTED_TASKS` have `len` of 0!")


conform_init_config()


def construct_configs_on_langs(languages: list = None) -> List[SEACrowdConfig]:
    """
    The function `construct_configs` constructs a list of SEACrowdConfig objects based on the provided
    languages or a default language, and returns the list.

    input:
        languages (list, default None): The `languages` parameter is a list that specifies the languages for which the
        configurations need to be constructed. If no languages are provided (value=None), the first value in language config
        will be used.
    output:
        a list of `SEACrowdConfig` objects based on instantiated init variables
    """

    # set output var
    config_list = []

    # construct zipped arg for config instantiation
    TASKS_AND_CONFIG_SUFFIX_PAIRS = list(zip(_SUPPORTED_TASKS, CONFIG_SUFFIXES_FOR_TASK))

    # implement source schema
    version, config_name_prefix = _SOURCE_VERSION, "source"
    config_list += [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{_LANG}_{config_name_prefix}",
            version=datasets.Version(version),
            description=f"{_DATASETNAME} {config_name_prefix} schema for language code {_LANG}",
            schema=f"{config_name_prefix}",
            subset_id=_LANG,
        )
        for _LANG in languages
    ]

    # implement SEACrowd schema
    version, config_name_prefix = _SEACROWD_VERSION, "seacrowd"
    for task_obj, config_name_suffix in TASKS_AND_CONFIG_SUFFIX_PAIRS:
        config_list += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{_LANG}_{config_name_prefix}_{config_name_suffix}",
                version=datasets.Version(version),
                description=f"{_DATASETNAME} {config_name_prefix} schema for {task_obj.name} and language code {_LANG}",
                schema=f"{config_name_prefix}_{config_name_suffix}",
                subset_id=_LANG,
            )
            for _LANG in languages
        ]
    return config_list


class SEA_MADLAD_Dataset(datasets.GeneratorBasedBuilder):
    """SEA MADLAD dataset, subsetted from https://huggingface.co/datasets/allenai/MADLAD-400"""

    # get all schema w/o lang arg + get all schema w/ lang arg
    BUILDER_CONFIGS = construct_configs_on_langs(_LANGUAGES)

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")
        # self supervised training schema
        if _config_schema_name == "source":
            features = datasets.Features({"text": datasets.Value("string")})

        elif _config_schema_name == "seacrowd_ssp":
            features = schemas.ssp_features

        else:
            raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        # args of dl_manager is a placeholder since this data loader will wrap the hf `load_dataset` from given _URL
        # directly using `_load_hf_data_from_remote`
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]

    def _load_hf_data_from_remote(self):
        # construct remote_hf_reference by the last 2 of string-spliited of "/"
        _lang_args = _LANG_CONFIG[self.config.subset_id]["source_subset"]
        _split = "clean"

        logger.info(f"Loading dataset from remote HF {_REMOTE_HF_REFERENCE} with seacrowd lang args of {self.config.subset_id} and source lang args of {_lang_args} and split args of {_split}")
        _hf_dataset_source = load_dataset(_REMOTE_HF_REFERENCE, languages=[_lang_args], split=_split)

        return _hf_dataset_source

    def _generate_examples(self) -> Tuple[int, Dict]:
        _config_schema_name = self.config.schema
        loaded_data = self._load_hf_data_from_remote()

        # iterate over datapoints and arrange hf dataset schema in source to match w/ config args:
        for id_, _data in enumerate(loaded_data):
            if _config_schema_name == "source":
                yield id_, {colname: _data[colname] for colname in self.info.features}

            # for ssp schema
            elif _config_schema_name == "seacrowd_ssp":
                yield id_, {"id": id_, "text": _data["text"]}

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")
