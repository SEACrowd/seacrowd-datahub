"""
This new update refers to the this HF dataloader script
https://huggingface.co/datasets/csebuetnlp/xlsum/blob/main/xlsum.py
while conforming to SEACrowd schema
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@inproceedings{valk2021slt,
  title={{VoxLingua107}: a Dataset for Spoken Language Recognition},
  author={J{\"o}rgen Valk and Tanel Alum{\"a}e},
  booktitle={Proc. IEEE SLT Workshop},
  year={2021},
}
"""

_LOCAL = False
_LANGUAGES = ["ceb", "ind", "jav", "khm", "lao", "zlm", "mya", "sun", "tha", "tgl", "vie", "war"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LANG_TO_DATASOURCE_LANG = {
    "ceb": "ceb",
    "ind": "id",
    "jav": "jw",
    "khm": "km",
    "lao": "lo",
    "zlm": "ms",
    "mya": "my",
    "sun": "su",
    "tha": "th",
    "tgl": "tl",
    "vie": "vi",
    "war": "war"}

_DATASETNAME = "voxlingua"

_DESCRIPTION = """\
VoxLingua107 is a comprehensive speech dataset designed for training spoken language identification models.
It comprises short speech segments sourced from YouTube videos, labeled based on the language indicated in the video
title and description. The dataset covers 107 languages and contains a total of 6628 hours of speech data,
averaging 62 hours per language. However, the actual amount of data per language varies significantly.
Additionally, there is a separate development set consisting of 1609 speech segments from 33 languages,
validated by at least two volunteers to ensure the accuracy of language representation.
"""

_HOMEPAGE = "https://bark.phon.ioc.ee/voxlingua107/"

_LICENSE = Licenses.CC_BY_4_0.value

_URLS = "https://bark.phon.ioc.ee/voxlingua107/{identifier}.zip"

_SUPPORTED_TASKS = [Tasks.SPEECH_LANGUAGE_IDENTIFICATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


def construct_configs() -> List[SEACrowdConfig]:
    """
    The function `construct_configs` constructs a list of SEACrowdConfig objects, and returns the list.

    output:
        a list of `SEACrowdConfig` objects.
    """

    # set output var
    config_list = []

    # construct zipped arg for config instantiation
    CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]
    TASKS_AND_CONFIG_SUFFIX_PAIRS = list(zip(_SUPPORTED_TASKS, CONFIG_SUFFIXES_FOR_TASK))

    # implement source schema
    version, config_name_prefix = _SOURCE_VERSION, "source"
    config_list += [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{config_name_prefix}",
            version=datasets.Version(version),
            description=f"{_DATASETNAME} {config_name_prefix} schema",
            schema=f"{config_name_prefix}",
            subset_id=f"{config_name_prefix}",
        )
    ]

    # implement SEACrowd schema
    version, config_name_prefix = _SEACROWD_VERSION, "seacrowd"
    for task_obj, config_name_suffix in TASKS_AND_CONFIG_SUFFIX_PAIRS:
        config_list += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{config_name_prefix}_{config_name_suffix}",
                version=datasets.Version(version),
                description=f"{_DATASETNAME} {config_name_prefix} schema for {task_obj.name}",
                schema=f"{config_name_prefix}_{config_name_suffix}",
                subset_id=f"{config_name_prefix}_{config_name_suffix}",
            )
        ]
    return config_list


class VoxLinguaDataset(datasets.GeneratorBasedBuilder):
    """Speech Lang ID on dataset VoxLingua."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = construct_configs()

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            # since the source only contains audio folder structure,
            # we will define it using simplified ver of SEACrowd speech_features schema
            features = datasets.Features({
                "id": datasets.Value("string"),
                "path": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "labels": datasets.ClassLabel(names=_LANGUAGES)})

        elif self.config.schema == "seacrowd_speech":
            features = schemas.speech_features(label_names=_LANGUAGES)

        else:
            raise ValueError(f"Unexpected self.config.schema of {self.config.schema}!")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # since this is a Speech LID, all languages must be downloaded in a single lists
        # for train data, the identifier is a lang_code defined in `_LANG_TO_DATASOURCE_LANG`
        train_url_list = [_URLS.format(identifier=_LANG_TO_DATASOURCE_LANG[lang_val]) for lang_val in _LANGUAGES]
        train_data_dir = dl_manager.download_and_extract(train_url_list)

        # for val data, the `dev.zip` doesn't contain any data indicated in _LANGUAGES

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_data_dir,
                },
            )
        ]

    def _generate_examples(self, filepath: List[Path]) -> Tuple[int, Dict]:

        # this is defined as -1 so that in the first loop it will have value of 0
        example_idx = -1

        for idx, child_path in enumerate(filepath):
            # check for 2 things:

            # 1. for every filepath list element (which contain 1 lang data), it will contain only 1 subdir and named its lang code in source
            first_level_rel_dir = os.listdir(child_path)
            expected_lang_label = _LANG_TO_DATASOURCE_LANG[_LANGUAGES[idx]]
            assert first_level_rel_dir == [expected_lang_label], f"The structure of path is unexpected! Expected {[expected_lang_label]} got: {first_level_rel_dir}"

            # 2. within the first_level_dir, all of them are file (no directory)
            first_level_dir = os.path.join(child_path, first_level_rel_dir[0])
            second_level_dir = os.listdir(first_level_dir)
            assert not all(os.path.isdir(expected_file) for expected_file in second_level_dir), f"Found directory within folder {first_level_dir}!"

            # extract sound data with format ".wav"
            wav_files = [os.path.join(first_level_dir, file) for file in second_level_dir if file.endswith(".wav")]

            if self.config.schema == "source":
                for _fp in wav_files:
                    example_idx += 1
                    ex = {"id": example_idx, "path": _fp, "audio": _fp, "labels": _LANGUAGES.index(expected_lang_label)}
                    yield example_idx, ex

            elif self.config.schema == "seacrowd_speech":
                for _fp in wav_files:
                    example_idx += 1
                    # audio = {"path": file, "bytes": file.read()}
                    ex = {
                        "id": example_idx,
                        "path": _fp,
                        "audio": _fp,
                        "speaker_id": "",
                        "labels": _LANGUAGES.index(expected_lang_label),
                        "metadata": {
                            # unavailable, filled with default val
                            "speaker_age": -1,
                            "speaker_gender": "",
                        },
                    }
                    yield example_idx, ex

            else:
                raise ValueError(f"Invalid config schema of {self.config.schema}!")
