"""
SEA Crowd Data Loader for Bloom Speech.
"""
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = r"""
@inproceedings{leong-etal-2022-bloom,
    title = "Bloom Library: Multimodal Datasets in 300+ Languages for a Variety of Downstream Tasks",
    author = "Leong, Colin  and
      Nemecek, Joshua  and
      Mansdorfer, Jacob  and
      Filighera, Anna  and
      Owodunni, Abraham  and
      Whitenack, Daniel",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.590",
    doi = "10.18653/v1/2022.emnlp-main.590",
    pages = "8608--8621",
}
"""

logger = datasets.logging.get_logger(__name__)

# this config is created for SEACrowd Dataloader
_LANG_CONFIG = {"bjn": "Banjar", "bzi": "Bisu", "ceb": "Cebuano", "ind": "Indonesian", "jra": "Jarai", "kqr": "Kimaragang", "mya": "Burmese", "tgl": "Tagalog"}

_LOCAL = False
_LANGUAGES = list(_LANG_CONFIG.keys())


_DATASETNAME = "bloom_speech"
_DESCRIPTION = r"""
This version of the Bloom Library data is developed specifically for the automatic speech recognition and speech-to-text tasks.
It includes data from 56 languages across 18 language families. 8 languages are spoken in Southeast Asia.
Before using this dataloader, please accept the acknowledgement at https://huggingface.co/datasets/sil-ai/bloom-speech and use huggingface-cli login for authentication.
"""

_HOMEPAGE = "https://huggingface.co/datasets/sil-ai/bloom-speech"
_LICENSE = Licenses.CC.value

_URL = "https://huggingface.co/datasets/sil-ai/bloom-speech"
_HF_REMOTE_REF = "/".join(_URL.split("/")[-2:])

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SOURCE_VERSION = "0.0.1"
_SEACROWD_VERSION = "2024.06.20"

CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]


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


class BloomSpeechDataset(datasets.GeneratorBasedBuilder):
    """Bloom Speech dataset, subsetted from https://huggingface.co/datasets/sil-ai/bloom-speech"""

    # get all schema w/o lang arg + get all schema w/ lang arg
    BUILDER_CONFIGS = construct_configs_on_langs(_LANGUAGES)

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")
        # source schema
        if _config_schema_name == "source":
            features = datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "book": datasets.Value("string"),
                    "instance": datasets.Value("string"),
                    "license": datasets.Value("string"),
                    "credits": datasets.Value("string"),
                    "original_lang_tag": datasets.Value("string"),
                }
            )

        # speech-text schema
        elif _config_schema_name == "seacrowd_sptext":
            features = schemas.speech_text_features

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
        hf_dset_dict = datasets.load_dataset(_HF_REMOTE_REF, self.config.subset_id)

        return [datasets.SplitGenerator(name=datasets.Split(dset_key), gen_kwargs={"hf_dset": dset}) for dset_key, dset in hf_dset_dict.items() if dset.num_rows > 0]

    def _generate_examples(self, hf_dset) -> Tuple[int, Dict]:
        _config_schema_name = self.config.schema

        _idx = 0
        for datapoints in hf_dset:
            # since no _idx is available to be used, we're creating it manually for both schema
            if _config_schema_name == "source":
                yield _idx, {colname: datapoints[colname] for colname in self.info.features}

            elif _config_schema_name == "seacrowd_sptext":
                yield _idx, {"id": _idx, "path": datapoints["file"], "audio": datapoints["audio"], "text": datapoints["text"], "speaker_id": None, "metadata": {"speaker_age": None, "speaker_gender": None}}

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

            _idx += 1
