"""
SEA Crowd Data Loader for Fleurs.
"""

import json
from itertools import product
from typing import Dict, List, Tuple

import datasets

from datasets import load_dataset
from datasets.download.download_manager import DownloadManager

from seacrowd.sea_datasets.fleurs.lang_config import _LANG_CONFIG
from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """
@inproceedings{conneau22_interspeech,
  author={Alexis Conneau and Ankur Bapna and Yu Zhang and Min Ma and Patrick {von Platen} and Anton Lozhkov and Colin Cherry
    and Ye Jia and Clara Rivera and Mihir Kale and Daan {van Esch} and Vera Axelrod and Simran Khanuja and Jonathan Clark
    and Orhan Firat and Michael Auli and Sebastian Ruder and Jason Riesa and Melvin Johnson},
  title={{XTREME-S: Evaluating Cross-lingual Speech Representations}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={3248--3252},
  doi={10.21437/Interspeech.2022-10007}
}
"""

logger = datasets.logging.get_logger(__name__)


_LOCAL = False

# since this fleurs source already subsets SEA langs, the names on lang group id is hard-coded
_LANG_GROUP_ID = ["south_east_asian_sea"]

_DATASETNAME = "fleurs"

_LANGUAGES = list(_LANG_CONFIG.keys())

_DESCRIPTION = """\
    Fleurs dataset is a part of XTREME-S benchmark to evaluate universal cross-lingual speech representations in many languages.
    Fleurs is used for two tasks: automatic speech recognition and speech classification.
    Fleurs covers 10 language native to Southeast Asian and other 3 major languages
    mostly spoken in few of Southeast Asia countries (Mandarin Chinese, Portuguese, and Tamil).
"""

_HOMEPAGE = "https://huggingface.co/datasets/google/xtreme_s"
_LICENSE = Licenses.CC_BY_4_0.value

# url won't be used since it will implement load_dataset method on HF URL provided
_URL = "https://huggingface.co/datasets/google/xtreme_s"

# construct remote_hf_reference by the last 2 of string-spliited of "/" (expected: "google/xtreme_s")
_HF_REMOTE_REF = "/".join(_URL.split("/")[-2:])

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION, Tasks.SPEECH_LANGUAGE_IDENTIFICATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

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
        languages (list): The `languages` parameter is a list that specifies the languages for which the
        configurations need to be constructed. If no languages are provided (value=None), the first value in language config
        will be used.
    output:
        a list of `SEACrowdConfig` objects based on instantiated init variables
    """
    # set output var
    config_list = []

    # set flag whether the task is lang-agnostic based on extended `_SUPPORTED_TASKS`
    IS_TASK_LANG_SUBSETTED = [True, False]

    TASKS_AND_CONFIG_SUFFIX_PAIRS = list(zip(_SUPPORTED_TASKS, CONFIG_SUFFIXES_FOR_TASK, IS_TASK_LANG_SUBSETTED))

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
    for (task_obj, config_name_suffix, is_lang_subsetted) in TASKS_AND_CONFIG_SUFFIX_PAIRS:
        if is_lang_subsetted:
            # construct configs based on its lang, since the task & config needs to defined per lang
            # for this dataloader, Tasks.SPEECH_RECOGNITION will enter this condition
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

        else:
            # else, its defined for all languages
            # for this dataloader, Tasks.SPEECH_LANGUAGE_IDENTIFICATION will enter this condition
            # however no "source" schema will be defined here (the source will follow this `fleurs_{lang_code}_source` config name)
            config_list.append(
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{config_name_prefix}_{config_name_suffix}",
                    version=datasets.Version(version),
                    description=f"{_DATASETNAME} {config_name_prefix} schema for {task_obj.name}",
                    schema=f"{config_name_prefix}_{config_name_suffix}",
                    subset_id="all",
                )
            )

    return config_list


class FleursDataset(datasets.GeneratorBasedBuilder):
    """Fleurs dataset from https://huggingface.co/datasets/google/xtreme_s"""

    # get all schema w/o lang arg + get all schema w/ lang arg
    BUILDER_CONFIGS = construct_configs_on_langs(_LANGUAGES)

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")

        # source schema
        if _config_schema_name == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "num_samples": datasets.Value("int32"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "transcription": datasets.Value("string"),
                    "raw_transcription": datasets.Value("string"),
                    "gender": datasets.ClassLabel(names=["male", "female", "other"]),
                    "lang_id": datasets.ClassLabel(names=_LANGUAGES),
                    "language": datasets.Value("string"),
                    "lang_group_id": datasets.ClassLabel(
                        names=_LANG_GROUP_ID)
                }
            )

        # asr transcription schema for seacrowd
        elif _config_schema_name == f"seacrowd_{CONFIG_SUFFIXES_FOR_TASK[0]}":
            features = schemas.speech_text_features

        # speech lang classification schema for seacrowd
        elif _config_schema_name == f"seacrowd_{CONFIG_SUFFIXES_FOR_TASK[1]}":
            features = schemas.speech_features(label_names=_LANGUAGES)

        else:
            raise ValueError(f"Unexpected schema received! {_config_schema_name}")

        return datasets.DatasetInfo(description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION)

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        # args of dl_manager is useless since this data loader will wrap the hf `load_dataset` from given _URL
        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={"split_name": split_name._name})
            for split_name in (
                datasets.Split.TRAIN,
                datasets.Split.VALIDATION,
                datasets.Split.TEST)
        ]

    def _load_hf_data_from_remote(self, split_name: str) -> datasets.DatasetDict:

        if self.config.subset_id == "all":
            raise ValueError("Unexpected subset_id value of `all` received in eager-load of SEACrowd fleurs loader!")
        else:
            _config_name_args = "fleurs." + _LANG_CONFIG[self.config.subset_id]["fleurs_lang_code"] + "_" + _LANG_CONFIG[self.config.subset_id]["fleurs_country_code"]

        logger.info(f"Loading dataset from remote HF {_HF_REMOTE_REF} with seacrowd lang args of {self.config.subset_id} and hf-source config args of {_config_name_args}")
        _hf_dataset_source = load_dataset(_HF_REMOTE_REF, _config_name_args, split=split_name)

        return _hf_dataset_source

    def _lazy_load_hf_data_from_remote(self, split_name: str) -> datasets.DatasetDict:

        if self.config.subset_id != "all":
            raise ValueError(f"Unexpected subset_id value of {self.config.subset_id} received in lazy-load of SEACrowd fleurs loader!")
        else:
            _config_name_args = [(f"fleurs.{fleurs_lang_info['fleurs_lang_code']}_{fleurs_lang_info['fleurs_country_code']}", lang) for lang, fleurs_lang_info in _LANG_CONFIG.items()]

        for _config, lang_name in _config_name_args:
            logger.info(f"Loading dataset from remote HF {_HF_REMOTE_REF} with seacrowd lang args of {self.config.subset_id} and hf-source config args of {_config}")
            yield load_dataset(_HF_REMOTE_REF, _config, split=split_name), lang_name

    def _generate_examples(self, split_name: str) -> Tuple[int, Dict]:

        _config_schema_name = self.config.schema

        # for source schema and asr transcription schema (the data is loaded eagerly, since it's splitted by lang)
        if _config_schema_name in ("source", f"seacrowd_{CONFIG_SUFFIXES_FOR_TASK[0]}"):
            loaded_data = self._load_hf_data_from_remote(split_name)

            # iterate over datapoints and arrange hf dataset schema in source to match w/ config args:
            for id_, _data in enumerate(loaded_data):
                if _config_schema_name == "source":

                    #re-map "language_id" and "lang_group_id"
                    _data["lang_id"] = _LANGUAGES.index(self.config.subset_id)
                    _data["lang_group_id"] = 0

                    yield id_, {
                        colname: _data[colname] for colname in self.info.features}

                # 2 notes on seacrowd schema for ASR:
                # 1. since in source data, no speakers id nor its info were provided, it will be filled by default values:
                #    ("" for any data string-typed, and -1 for age data int-typed)
                # 2. the "id" is re-created on sequential order on loaded data bcs it's original id
                #    doesn't pass unit-test of seacrowd schema

                elif "seacrowd" in _config_schema_name:
                    yield id_, {
                        "id": id_,
                        "path": _data["path"],
                        "audio": _data["audio"],
                        "text": _data["transcription"],
                        "speaker_id": "",
                        "metadata": {
                            "speaker_age": -1,
                            "speaker_gender": _data["gender"],
                        },
                    }

                else:
                    raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

                # add id_ so it will be globally unique
                id_ += 1

        # for speech lang classification schema (the data is loaded lazily per lang)
        elif _config_schema_name == f"seacrowd_{CONFIG_SUFFIXES_FOR_TASK[1]}":
            loaded_data = self._lazy_load_hf_data_from_remote(split_name)
            id_ = 0
            while True:
                _loaded_data, lang_info = next(loaded_data, (None, None))
                if _loaded_data is None:
                    break
                # iterate over datapoints and arrange hf dataset schema in source to match w/ config args:
                for _data in _loaded_data:
                    yield id_, {
                        "id": id_,
                        "path": _data["path"],
                        "audio": _data["audio"],
                        "labels": _LANGUAGES.index(lang_info),
                        "speaker_id": "",
                        "metadata": {
                            "speaker_age": -1,
                            "speaker_gender": _data["gender"],
                        },
                    }

                    # add id_ so it will be globally unique
                    id_ += 1

        else:
            raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")
