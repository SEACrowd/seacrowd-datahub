"""
SEA Crowd Data Loader for Bloom Speech.
"""
import os
import json

from typing import Dict, Generator, List, Tuple, Union

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = r"""
@article{
    msl4emergency,
    title={Statistical Machine Translation between Myanmar Sign Language and Myanmar Written Text},
    url={https://core.ac.uk/outputs/489828410?source=oai},
    journal={CORE},
    author={Moe, Swe Zin and Thu, Ye Kyaw and Hlaing, Hnin Wai Wai and Nwe, Hlaing Myat and Aung, Ni Htwe and Thant, Hnin Aye and Min, Nandar Win},
    year={2018},
    month={Mar}
}
"""

logger = datasets.logging.get_logger(__name__)

_LOCAL = False
_LANGUAGES = ["ysm", "mya"]

_DATASETNAME = "msl4emergency"
_DESCRIPTION = r"""
The MSL4Emergency corpus is part of a larger Myanmar sign language (MSL) corpus that specifically contains sign language videos for the emergency domain.
Each signing video is annotated with both its transcription and its Burmese written translation, which may differ from each other due to grammar, syntax and vocabulary differences between MSL and Burmese.
Signing videos were made by sign language trainers and deaf trainees.
"""

_HOMEPAGE = "https://github.com/ye-kyaw-thu/MSL4Emergency/tree/master"
_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_URL = "https://github.com/ye-kyaw-thu/MSL4Emergency/archive/refs/heads/master.zip"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION, Tasks.SIGN_LANGUAGE_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

_CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]


class MSL4Emergency(datasets.GeneratorBasedBuilder):
    """MSL4Emergency dataset"""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME)
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{cfg_sufix}",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} seacrowd schema for {task.name}",
            schema=f"seacrowd_{cfg_sufix}",
            subset_id=_DATASETNAME)
        for task, cfg_sufix in zip(_SUPPORTED_TASKS, _CONFIG_SUFFIXES_FOR_TASK)
    ]

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")

        if _config_schema_name == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "mya_text": datasets.Value("string"),
                    "ysm_text": datasets.Value("string"),
                    "video_url": datasets.Value("string")
                }
            )

        # speech-text schema
        elif _config_schema_name == f"seacrowd_sptext":
            features = schemas.text2text_features

        elif _config_schema_name == f"seacrowd_imtext":
            features = schemas.image_text_features()

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
        local_path = dl_manager.download_and_extract(_URL)
        local_path = os.path.join(local_path.title(), "msl4emergency-ver-1.0")

        base_video_dir = os.path.join(local_path, "video")
        video_dir_list = []
        for _child_dir in base_video_dir:
            _full_child_dir = os.path.join(base_video_dir, _child_dir)
            if os.path.isdir(_full_child_dir):
                video_dir_list.extend([os.path.join(_full_child_dir, video_fp) for video_fp in os.listdir(_full_child_dir) if video_fp.endswith(".mp4")])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "article_data_dir": video_dir_list,
                    "text_translation": os.path.join(local_path, "my-sl")
                })
        ]

    def _generate_examples(self, video_dir_list: str, text_translation: str) -> Generator[Tuple[int, Dict], None, None]:
        _config_schema_name = self.config.schema

        with open(text_translation, "r") as f:
            trans_data = [data.split("\t") for data in f.readlines()]
        idx = 1
        for video_path in video_dir_list:
            if _config_schema_name == "source":
                yield idx, {
                    "id": idx,
                    "mya_text": trans_data[idx][0],
                    "ysm_text": trans_data[idx][1],
                    "video_url": video_path
                }

            elif _config_schema_name == "seacrowd_sptext":
                yield idx, {
                    "id": idx,
                    "text_1": trans_data[idx][0],
                    "text_2": trans_data[idx][1],
                    "text_1_name": "target_mya",
                    "text_2_name": "source_ysm"
                }

            elif _config_schema_name == "seacrowd_imtext":
                yield idx, {
                    "id": idx,
                    "image_paths": [video_path],
                    "texts": trans_data[idx][1],
                    "metadata": {
                        "context": "myanmar sign language transcribed",
                        "labels": None,
                    },
                }

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

            idx += 1
