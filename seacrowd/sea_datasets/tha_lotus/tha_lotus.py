"""
SEA Crowd Data Loader for Bloom Speech.
"""
import os
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = r"""
@INPROCEEDINGS{thaiLOTUSBN,
  author={Chotimongkol, Ananlada and Saykhum, Kwanchiva and Chootrakool, Patcharika and Thatphithakkul, Nattanun and Wutiwiwatchai, Chai},
  booktitle={2009 Oriental COCOSDA International Conference on Speech Database and Assessments}, 
  title={LOTUS-BN: A Thai broadcast news corpus and its research applications}, 
  year={2009},
  volume={},
  number={},
  pages={44-50},
  doi={10.1109/ICSDA.2009.5278377}}
"""

logger = datasets.logging.get_logger(__name__)

_LOCAL = False
_LANGUAGES = ["tha"]


_DATASETNAME = "tha_lotus"
_DESCRIPTION = r"""
The Large vOcabualry Thai continUous Speech recognition (LOTUS) corpus was designed for developing large vocabulary
continuous speech recognition (LVCSR), spoken dialogue system, speech dictation, broadcast news transcriber.
It contains two datasets, one for training acoustic model, another for training a language model.
"""

_HOMEPAGE = "https://github.com/korakot/corpus/tree/main/LOTUS"
_LICENSE = Licenses.CC.value

_URL = "https://github.com/korakot/corpus/releases/download/v1.0/AIFORTHAI-LotusCorpus.zip"


_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]
assert len(CONFIG_SUFFIXES_FOR_TASK) == 1

config_choices_folder_path = {
    "unidrection_clean": os.path.join("U", "Clean"),
    "unidrection_office": os.path.join("U", "Office"),
    "closetalk_clean": os.path.join("C", "Clean"),
    "closetalk_office": os.path.join("C", "Office")}

class ThaiLOTUS(datasets.GeneratorBasedBuilder):
    """Bloom Speech dataset, subsetted from https://huggingface.co/datasets/sil-ai/bloom-speech"""

    # get all schema w/o lang arg + get all schema w/ lang arg
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{config_name}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema for config {config_name}",
            schema=f"source",
            subset_id=config_name
        ) for config_name in config_choices_folder_path.keys()
    ] + [
        SEACrowdConfig(
                name=f"{_DATASETNAME}_{config_name}_seacrowd_{CONFIG_SUFFIXES_FOR_TASK[0]}",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} seacrowd schema for {_SUPPORTED_TASKS[0].name} and config {config_name}",
                schema=f"seacrowd_{CONFIG_SUFFIXES_FOR_TASK[0]}",
                subset_id=config_name
            ) for config_name in config_choices_folder_path.keys()
    ]

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

                }
            )

        # speech-text schema
        elif _config_schema_name == f"seacrowd_{CONFIG_SUFFIXES_FOR_TASK[0]}":
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
        # since the folder are zipped, the zipped URL containing whole resource of this dataset must be downloaded
        _all_folder_local = dl_manager.download_and_extract(_URL)

        # supplement files is used regardless of the config
        # it contains the mapper and individual text files
        supplement_folder = os.path.join(_all_folder_local, "Supplement")
        wav_folder = os.path.join(_all_folder_local, config_choices_folder_path[self.config.subset_id], "Wav")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "wav_folder": wav_folder,
                    "supplement_folder": supplement_folder})]

    def _generate_examples(self, wav_folder, supplement_folder) -> Tuple[int, Dict]:
        _config_schema_name = self.config.schema
        with open("{supplement_folder}/PDsen.txt", "r") as f:
            text_file = f.readlines()

        with open("{supplement_folder}/index.txt", "r") as f:
            index_text_file = f.readlines()

        # this record list will contain short .wav files contain of Thai graphemes which 
        wav_record_list = [os.path.join(wav_folder, wav_file) for wav_file in os.listdir(wav_folder)]


        # for datapoints in index_text_file:
        #     _idx = 
        #     if _config_schema_name == "source":
        #         yield _idx, {colname: datapoints[colname] for colname in self.info.features}

        #     elif _config_schema_name == "seacrowd_sptext":
        #         yield _idx, {"id": _idx, "path": datapoints["file"], "audio": datapoints["audio"], "text": datapoints["text"], "speaker_id": None, "metadata": {"speaker_age": None, "speaker_gender": None}}

        #     else:
        #         raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

        #     _idx += 1
