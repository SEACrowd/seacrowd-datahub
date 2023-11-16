"""
SEA Crowd Data Loader for Fleurs.
"""

import json
from functools import partial
from itertools import product
from collections import OrderedDict
from collections.abc import Iterable
from typing import Dict, List, Tuple

import datasets
from datasets import load_dataset
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks, TASK_TO_SCHEMA

_CITATION = """
@inproceedings{conneau22_interspeech,
  author={Alexis Conneau and Ankur Bapna and Yu Zhang and Min Ma and Patrick {von Platen} and Anton Lozhkov and Colin Cherry and Ye Jia and Clara Rivera and Mihir Kale and Daan {van Esch} and Vera Axelrod and Simran Khanuja and Jonathan Clark and Orhan Firat and Michael Auli and Sebastian Ruder and Jason Riesa and Melvin Johnson},
  title={{XTREME-S: Evaluating Cross-lingual Speech Representations}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={3248--3252},
  doi={10.21437/Interspeech.2022-10007}
}
"""

logger = datasets.logging.get_logger(__name__)


with open(DownloadManager().download_and_extract("seacrowd/sea_datasets/fleurs/lang_config.json"), "r") as f:
    _LANG_CONFIG = json.load(f)

_LOCAL = False
_LANGUAGES = list(_LANG_CONFIG.keys())

_DATASETNAME = "fleurs"
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

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION, Tasks.SPEECH_LANGUAGE_IDENTIFICATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

CONFIG_SUFFIXES_FOR_TASK = ["_"+TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS if task != task.FACT_CHECKING]

# # fleurs-specific original config:
# _FLEURS_LANG_TO_ID = OrderedDict([("Afrikaans", "af"), ("Amharic", "am"), ("Arabic", "ar"), ("Armenian", "hy"), ("Assamese", "as"), ("Asturian", "ast"), ("Azerbaijani", "az"), ("Belarusian", "be"), ("Bengali", "bn"), ("Bosnian", "bs"), ("Bulgarian", "bg"), ("Burmese", "my"), ("Catalan", "ca"), ("Cebuano", "ceb"), ("Mandarin Chinese", "cmn_hans"), ("Cantonese Chinese", "yue_hant"), ("Croatian", "hr"), ("Czech", "cs"), ("Danish", "da"), ("Dutch", "nl"), ("English", "en"), ("Estonian", "et"), ("Filipino", "fil"), ("Finnish", "fi"), ("French", "fr"), ("Fula", "ff"), ("Galician", "gl"), ("Ganda", "lg"), ("Georgian", "ka"), ("German", "de"), ("Greek", "el"), ("Gujarati", "gu"), ("Hausa", "ha"), ("Hebrew", "he"), ("Hindi", "hi"), ("Hungarian", "hu"), ("Icelandic", "is"), ("Igbo", "ig"), ("Indonesian", "id"), ("Irish", "ga"), ("Italian", "it"), ("Japanese", "ja"), ("Javanese", "jv"), ("Kabuverdianu", "kea"), ("Kamba", "kam"), ("Kannada", "kn"), ("Kazakh", "kk"), ("Khmer", "km"), ("Korean", "ko"), ("Kyrgyz", "ky"), ("Lao", "lo"), ("Latvian", "lv"), ("Lingala", "ln"), ("Lithuanian", "lt"), ("Luo", "luo"), ("Luxembourgish", "lb"), ("Macedonian", "mk"), ("Malay", "ms"), ("Malayalam", "ml"), ("Maltese", "mt"), ("Maori", "mi"), ("Marathi", "mr"), ("Mongolian", "mn"), ("Nepali", "ne"), ("Northern-Sotho", "nso"), ("Norwegian", "nb"), ("Nyanja", "ny"), ("Occitan", "oc"), ("Oriya", "or"), ("Oromo", "om"), ("Pashto", "ps"), ("Persian", "fa"), ("Polish", "pl"), ("Portuguese", "pt"), ("Punjabi", "pa"), ("Romanian", "ro"), ("Russian", "ru"), ("Serbian", "sr"), ("Shona", "sn"), ("Sindhi", "sd"), ("Slovak", "sk"), ("Slovenian", "sl"), ("Somali", "so"), ("Sorani-Kurdish", "ckb"), ("Spanish", "es"), ("Swahili", "sw"), ("Swedish", "sv"), ("Tajik", "tg"), ("Tamil", "ta"), ("Telugu", "te"), ("Thai", "th"), ("Turkish", "tr"), ("Ukrainian", "uk"), ("Umbundu", "umb"), ("Urdu", "ur"), ("Uzbek", "uz"), ("Vietnamese", "vi"), ("Welsh", "cy"), ("Wolof", "wo"), ("Xhosa", "xh"), ("Yoruba", "yo"), ("Zulu", "zu")])
# _FLEURS_LANG_SHORT_TO_LONG = {v: k for k, v in _FLEURS_LANG_TO_ID.items()}

# _FLEURS_LANG = sorted(["af_za", "am_et", "ar_eg", "as_in", "ast_es", "az_az", "be_by", "bn_in", "bs_ba", "ca_es", "ceb_ph", "cmn_hans_cn", "yue_hant_hk", "cs_cz", "cy_gb", "da_dk", "de_de", "el_gr", "en_us", "es_419", "et_ee", "fa_ir", "ff_sn", "fi_fi", "fil_ph", "fr_fr", "ga_ie", "gl_es", "gu_in", "ha_ng", "he_il", "hi_in", "hr_hr", "hu_hu", "hy_am", "id_id", "ig_ng", "is_is", "it_it", "ja_jp", "jv_id", "ka_ge", "kam_ke", "kea_cv", "kk_kz", "km_kh", "kn_in", "ko_kr", "ckb_iq", "ky_kg", "lb_lu", "lg_ug", "ln_cd", "lo_la", "lt_lt", "luo_ke", "lv_lv", "mi_nz", "mk_mk", "ml_in", "mn_mn", "mr_in", "ms_my", "mt_mt", "my_mm", "nb_no", "ne_np", "nl_nl", "nso_za", "ny_mw", "oc_fr", "om_et", "or_in", "pa_in", "pl_pl", "ps_af", "pt_br", "ro_ro", "ru_ru", "bg_bg", "sd_in", "sk_sk", "sl_si", "sn_zw", "so_so", "sr_rs", "sv_se", "sw_ke", "ta_in", "te_in", "tg_tj", "th_th", "tr_tr", "uk_ua", "umb_ao", "ur_pk", "uz_uz", "vi_vn", "wo_sn", "xh_za", "yo_ng", "zu_za"])
# _FLEURS_LONG_TO_LANG = {_FLEURS_LANG_SHORT_TO_LONG["_".join(k.split("_")[:-1]) or k]: k for k in _FLEURS_LANG}
# _FLEURS_LANG_TO_LONG = {v: k for k, v in _FLEURS_LONG_TO_LANG.items()}

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
    # set output var
    config_list = []

    # set flag whether the task is lang-agnostic based on extended `_SUPPORTED_TASKS`
    IS_TASK_LANG_SUBSETTED = [True, False]

    # set default task for default config w/o task arg name
    _DEFAULT_TASK = Tasks.SPEECH_RECOGNITION

    __SUPPORTED_TASKS = [_DEFAULT_TASK] + _SUPPORTED_TASKS
    __CONFIG_SUFFIXES_FOR_TASK = [""] + CONFIG_SUFFIXES_FOR_TASK

    # set flag whether the task is lang-agnostic based on extended `__SUPPORTED_TASKS`
    __IS_TASK_LANG_SUBSETTED = [True]+IS_TASK_LANG_SUBSETTED

    TASKS_AND_CONFIG_SUFFIX_PAIRS = list(zip(_SUPPORTED_TASKS, CONFIG_SUFFIXES_FOR_TASK, IS_TASK_LANG_SUBSETTED))
    EXT_TASKS_AND_CONFIG_SUFFIX_PAIRS = list(zip(__SUPPORTED_TASKS, __CONFIG_SUFFIXES_FOR_TASK, __IS_TASK_LANG_SUBSETTED))
    VERSION_AND_CONFIG_PREFIX_PAIRS = list(zip((_SOURCE_VERSION, _SEACROWD_VERSION), ("source", "seacrowd")))

    ## 
    # check `languages` variable and create config accordingly
    if languages is None:
        # set languages arg as list of first entry in `_LANGUAGES` if no lang arg received
        _languages = _LANGUAGES[0]

        config_list += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{config_name_prefix}{config_name_suffix}",
                version=datasets.Version(version),
                description=f"{_DATASETNAME} {config_name_prefix} schema for {task_obj.name}",
                schema=f"{config_name_prefix}{config_name_suffix if config_name_suffix != '' else CONFIG_SUFFIXES_FOR_TASK[_SUPPORTED_TASKS.index(_DEFAULT_TASK)]}",
                subset_id=_languages if is_lang_subsetted else "all",
            )
            for (version, config_name_prefix), (task_obj, config_name_suffix, is_lang_subsetted) in product(VERSION_AND_CONFIG_PREFIX_PAIRS, EXT_TASKS_AND_CONFIG_SUFFIX_PAIRS)
        ]

    # else, construct configs based on its lang
    else:
        for _LANG in languages:
            config_list += [
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{config_name_prefix}_{_LANG}{config_name_suffix}",
                    version=datasets.Version(version),
                    description=f"{_DATASETNAME} {config_name_prefix} schema for {task_obj.name} and language code {_LANG}",
                    schema=f"{config_name_prefix}{config_name_suffix}",
                    subset_id=_LANG,
                )
                for (version, config_name_prefix), (task_obj, config_name_suffix, is_lang_subsetted) in product(VERSION_AND_CONFIG_PREFIX_PAIRS, TASKS_AND_CONFIG_SUFFIX_PAIRS)
                if is_lang_subsetted
            ]

    return config_list


class SEAWikiDataset(datasets.GeneratorBasedBuilder):
    """Fleurs dataset from https://huggingface.co/datasets/google/xtreme_s"""

    # get all schema w/o lang arg + get all schema w/ lang arg
    BUILDER_CONFIGS = construct_configs() + construct_configs(_LANGUAGES)

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")
        # asr transcription schema
        if CONFIG_SUFFIXES_FOR_TASK[0] in _config_schema_name:
            if "source" in _config_schema_name:
                features = datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "transcription": datasets.Value("string"),
                    "raw_transcription": datasets.Value("string"),
                    "gender": datasets.ClassLabel(names=["male", "female", "other"])
                }
            )

            elif "seacrowd" in _config_schema_name:
                features = schemas.speech_text_features

            else:
                raise ValueError(f"Unexpected schema received! {_config_schema_name}")

        # speech lang classification schema
        elif CONFIG_SUFFIXES_FOR_TASK[1] in _config_schema_name:
            if "source" in _config_schema_name:
                features = datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "gender": datasets.ClassLabel(names=["male", "female", "other"]),
                    "language": datasets.Value("string"),
                }
            )

            elif "seacrowd" in _config_schema_name:
                features = schemas.speech_features(label_names=_LANGUAGES)

            else:
                raise ValueError(f"Unexpected schema received! {_config_schema_name}")
        
        else:
            raise ValueError(f"Unexpected schema received! {_config_schema_name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        # args of dl_manager is useless since this data loader will wrap the hf `load_dataset` from given _URL
        return [datasets.SplitGenerator(name=split_name, gen_kwargs={"split_name": split_name._name}) for split_name in (datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST)]

    def _load_hf_data_from_remote(self, split_name: str) -> datasets.DatasetDict:
        # construct remote_hf_reference by the last 2 of string-spliited of "/"
        _remote_hf_reference = "/".join(_URL.split("/")[-2:])
        if self.config.subset_id == "all":
            raise ValueError("Unexpected subset_id value of `all` received in eager-load of SEACrowd fleurs loader!")
        else:
            _config_name_args = "fleurs."+_LANG_CONFIG[self.config.subset_id]["fleurs_lang_code"]+"_"+_LANG_CONFIG[self.config.subset_id]["fleurs_country_code"]

        logger.info(f"Loading dataset from remote HF {_remote_hf_reference} with seacrowd lang args of {self.config.subset_id} and hf-source config args of {_config_name_args}")
        _hf_dataset_source = load_dataset(_remote_hf_reference, _config_name_args, split=split_name)

        return _hf_dataset_source

    def _lazy_load_hf_data_from_remote(self, split_name: str) -> datasets.DatasetDict:
        _remote_hf_reference = "/".join(_URL.split("/")[-2:])
        if self.config.subset_id != "all":
            raise ValueError(f"Unexpected subset_id value of {self.config.subset_id} received in lazy-load of SEACrowd fleurs loader!")
        else:
            _config_name_args = [(f"fleurs.{fleurs_lang_info['fleurs_lang_code']}_{fleurs_lang_info['fleurs_country_code']}", lang) for lang, fleurs_lang_info in _LANG_CONFIG.items()]

        for _config, lang_name in _config_name_args:
            logger.info(f"Loading dataset from remote HF {_remote_hf_reference} with seacrowd lang args of {self.config.subset_id} and hf-source config args of {_config}")
            yield load_dataset(_remote_hf_reference, _config, split=split_name), lang_name


    def _generate_examples(self, split_name: str) -> Tuple[int, Dict]:

        _config_schema_name = self.config.schema

        # for asr transcription schema (the data is loaded eagerly)
        if CONFIG_SUFFIXES_FOR_TASK[0] in _config_schema_name:
            loaded_data = self._load_hf_data_from_remote(split_name)

            # iterate over datapoints and arrange hf dataset schema in source to match w/ config args:
            for id_, _data in enumerate(loaded_data):
                if "source" in _config_schema_name:
                    yield id_, {colname: _data[colname] for colname in self.info.features}

                # 2 notes on seacrowd schema:
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
                        }
                    }

                else:
                    raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

        # for speech lang classification schema (the data is loaded lazily per lang)
        if CONFIG_SUFFIXES_FOR_TASK[1] in _config_schema_name:
            loaded_data = self._lazy_load_hf_data_from_remote(split_name)
            id_ = 0
            while True:
                _loaded_data, lang_info = next(loaded_data, (None, None))
                if _loaded_data is None:
                    break
                # iterate over datapoints and arrange hf dataset schema in source to match w/ config args:
                for _data in _loaded_data:
                    if "source" in _config_schema_name:
                        yield id_, {colname: _data[colname] for colname in self.info.features}

                    elif "seacrowd" in _config_schema_name:
                        yield id_, {
                            "id": id_,
                            "path": _data["path"],
                            "audio": _data["audio"],
                            "labels": _LANGUAGES.index(lang_info),
                            "speaker_id": "",
                            "metadata": {
                                "speaker_age": -1,
                                "speaker_gender": _data["gender"],
                            }
                        }

                    else:
                        raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

                    # add id_ so it will be globally unique
                    id_ += 1


if __name__ == "__main__":
    datasets.load_dataset(__file__)