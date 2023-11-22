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
from seacrowd.utils.constants import Licenses, Tasks

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


with open(DownloadManager().download_and_extract("seacrowd/sea_datasets/sea_madlad/lang_config.json"), "r") as f:
    _LANG_CONFIG = json.load(f)

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

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

CONFIG_SUFFIXES_FOR_TASK = ["ssp"]


def conform_init_config():
    """Assertion Function for Instantiated Configs"""
    if len(_LANGUAGES) == 0:
        raise AssertionError("No Languages detected from config!")
    if len(CONFIG_SUFFIXES_FOR_TASK) != len(_SUPPORTED_TASKS):
        raise AssertionError("Config prefixes doesn't matched in terms of `len` with `_SUPPORTED_TASKS`!")
    if len(CONFIG_SUFFIXES_FOR_TASK) == 0:
        raise AssertionError("Config prefixes and `_SUPPORTED_TASKS` have `len` of 0!")


conform_init_config()


def construct_configs(languages: list = None) -> List[SEACrowdConfig]:
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

    # construct zipped arg for config instantiation
    CONFIG_NAME_AND_TASKS_PAIRS = list(zip(CONFIG_SUFFIXES_FOR_TASK, _SUPPORTED_TASKS))
    SCHEMA_PREFIX_AND_VERSION_PAIRS = list(zip(("source", "seacrowd"), (_SOURCE_VERSION, _SEACROWD_VERSION)))

    # set output var
    config_list = []

    # set default task for default config w/o task arg name (set to Tasks.SUMMARIZATION)
    _DEFAULT_TASK_IDX = [idx for idx, val in enumerate(_SUPPORTED_TASKS) if val == Tasks.SUMMARIZATION]

    # assert `_DEFAULT_TASK_IDX` to have len of 1
    if len(_DEFAULT_TASK_IDX) != 1:
        raise AssertionError("Unexpected `_DEFAULT_TASK` #item!")

    _DEFAULT_CONFIG_SUFFIX, _DEFAULT_TASK = list(CONFIG_NAME_AND_TASKS_PAIRS)[_DEFAULT_TASK_IDX[0]]

    # check `languages` variable and create config accordingly
    if languages is None:
        # set languages arg as list of first entry in `_LANGUAGES` if no lang arg received
        _languages = _LANGUAGES[0]

        config_list += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{config_name_prefix}",
                version=datasets.Version(version),
                description=f"{_DATASETNAME} {config_name_prefix} schema for default task arg ({_DEFAULT_TASK})",
                schema=f"{config_name_prefix}_{_DEFAULT_CONFIG_SUFFIX}",
                subset_id=_languages,
            )
            for (config_name_prefix, version) in SCHEMA_PREFIX_AND_VERSION_PAIRS
        ]
        config_list += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{config_name_prefix}_{config_name_suffix}",
                version=datasets.Version(version),
                description=f"{_DATASETNAME} {config_name_prefix} schema for {task_obj.name}",
                schema=f"{config_name_prefix}_{config_name_suffix}",
                subset_id=_languages,
            )
            for (config_name_prefix, version), (config_name_suffix, task_obj) in product(SCHEMA_PREFIX_AND_VERSION_PAIRS, CONFIG_NAME_AND_TASKS_PAIRS)
        ]

    # else, construct configs based on its lang
    else:
        for _LANG in languages:
            config_list += [
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{config_name_prefix}_{_LANG}_{config_name_suffix}",
                    version=datasets.Version(version),
                    description=f"{_DATASETNAME} {config_name_prefix} schema for {task_obj.name} and language code {_LANG}",
                    schema=f"{config_name_prefix}_{config_name_suffix}",
                    subset_id=_LANG,
                )
                for (config_name_prefix, version), (config_name_suffix, task_obj) in product(SCHEMA_PREFIX_AND_VERSION_PAIRS, CONFIG_NAME_AND_TASKS_PAIRS)
            ]

    return config_list


class SEAWikiDataset(datasets.GeneratorBasedBuilder):
    """SEA MADLAD dataset, subsetted from https://huggingface.co/datasets/allenai/MADLAD-400"""

    # get all schema w/o lang arg + get all schema w/ lang arg
    BUILDER_CONFIGS = construct_configs() + construct_configs(_LANGUAGES)

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")
        # self supervised training schema
        if CONFIG_SUFFIXES_FOR_TASK[0] in _config_schema_name:
            if "source" in _config_schema_name:
                features = datasets.Features({"url": datasets.Value("string"), "text": datasets.Value("string")})

            elif "seacrowd" in _config_schema_name:
                features = schemas.ssp_features

            else:
                raise ValueError(f"Unexpected schema received! {_config_schema_name}")
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
        _remote_hf_reference = "/".join(_URL.split("/")[-2:])
        _lang_args = _LANG_CONFIG[self.config.subset_id]["source_subset"]
        _split = "clean"

        logger.info(f"Loading dataset from remote HF {_remote_hf_reference} with seacrowd lang args of {self.config.subset_id} and source lang args of {_lang_args} and split args of {_split}")
        _hf_dataset_source = load_dataset(_remote_hf_reference, lang=_lang_args, split=_split)

        return _hf_dataset_source

    def _generate_examples(self) -> Tuple[int, Dict]:

        _config_schema_name = self.config.schema
        loaded_data = self._load_hf_data_from_remote()

        # iterate over datapoints and arrange hf dataset schema in source to match w/ config args:
        for id_, _data in enumerate(loaded_data):
            if "source" in _config_schema_name:
                yield id_, {colname: _data[colname] for colname in self.info.features}

            # for ssp schema
            elif "seacrowd" in _config_schema_name and CONFIG_SUFFIXES_FOR_TASK[0] in _config_schema_name:
                yield id_, {"id": id_, "text": _data["text"]}

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")
