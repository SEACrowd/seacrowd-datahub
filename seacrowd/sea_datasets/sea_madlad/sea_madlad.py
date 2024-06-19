"""
SEA Crowd Data Loader for SEA MADLAD.
"""

import gzip
import json
from typing import Dict, List, Tuple

import datasets
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

# this config is created for SEACrowd Dataloader
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
    "msa": {"name": "Malay", "source_subset": "ms"},
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

# this config is copied and added from source dataloader
# only using the `clean` values
_N_SHARDS_PER_SPLIT = {
    "ace": 1,
    "akb": 1,
    "ban": 1,
    "bbc": 1,
    "bew": 1,
    "btx": 1,
    "ceb": 1,
    "fil": 1,
    "gor": 1,
    "hil": 1,
    "iba": 1,
    "id": 18,
    "ilo": 1,
    "jv": 1,
    "kac": 1,
    "km": 1,
    "lo": 1,
    "mad": 1,
    "mak": 1,
    "meo": 1,
    "min": 1,
    "mkn": 1,
    "ms": 2,
    "ms_Arab_BN": 1,
    "msi": 1,
    "my": 1,
    "nij": 1,
    "nut": 1,
    "pag": 1,
    "shn": 1,
    "su": 1,
    "tet": 1,
    "th": 21,
    "vi": 32,
    "war": 1,
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

_URL = "https://huggingface.co/datasets/allenai/MADLAD-400/resolve/ecd71297d60c1eb996cd3d7c44c60ad5b55adfc6/data/{language}/{language}_{split}_{index:04d}.jsonl.gz"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]


def conform_init_config():
    """Assertion Function for Instantiated Configs"""
    if len(_LANGUAGES) == 0:
        raise AssertionError("No Languages detected from config!")
    if len(CONFIG_SUFFIXES_FOR_TASK) != len(_SUPPORTED_TASKS):
        raise AssertionError("Config prefixes don't matched in terms of `len` with `_SUPPORTED_TASKS`!")
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


class SEAMADLADDataset(datasets.GeneratorBasedBuilder):
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
        # construct URL from "lang", "split" -> "clean" split, and "index" based on `_N_SHARDS_PER_SPLIT`
        _lang = _LANG_CONFIG[self.config.subset_id]["source_subset"]
        _split = "clean"
        _data_list = [_URL.format(language=_lang, split=_split, index=idx) for idx in range(_N_SHARDS_PER_SPLIT[_lang])]

        filepaths = dl_manager.download(_data_list)

        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": filepaths})]

    def _generate_examples(self, filepaths) -> Tuple[int, Dict]:
        _config_schema_name = self.config.schema

        # the id_ constructions follows the source Dataloader
        id_ = 0
        for filepath in filepaths:
            logger.info("generating examples from = %s", filepath)
            with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for line in f:
                    if line:
                        example = json.loads(line)

                        # for source_schema
                        if _config_schema_name == "source":
                            yield id_, {colname: example[colname] for colname in self.info.features}

                        # for ssp schema
                        elif _config_schema_name == "seacrowd_ssp":
                            yield id_, {"id": id_, "text": example["text"]}

                        else:
                            raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

                        id_ += 1
