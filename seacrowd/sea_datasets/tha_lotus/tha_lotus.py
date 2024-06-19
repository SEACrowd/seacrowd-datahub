"""
SEA Crowd Data Loader for Thai LOTUS.
"""
import os
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

import pandas as pd
from collections import Counter
from collections.abc import KeysView, Iterable

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
_LICENSE = Licenses.CC_BY_NC_SA_3_0.value

_URL = "https://github.com/korakot/corpus/releases/download/v1.0/AIFORTHAI-LotusCorpus.zip"


_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]
assert len(CONFIG_SUFFIXES_FOR_TASK) == 1

config_choices_folder_structure = {
    "unidrection_clean": ("PD", "U", "Clean"),
    "unidrection_office": ("PD", "U", "Office"),
    "closetalk_clean": ("PD", "C", "Clean"),
    "closetalk_office": ("PD", "C", "Office")}


class ThaiLOTUS(datasets.GeneratorBasedBuilder):
    """Thai Lotus free-version dataset, re-implemented for SEACrowd from https://github.com/korakot/corpus/blob/main/LOTUS"""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{config_name}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema for config {config_name}",
            schema=f"source",
            subset_id=config_name
        ) for config_name in config_choices_folder_structure.keys()
    ] + [
        SEACrowdConfig(
                name=f"{_DATASETNAME}_{config_name}_seacrowd_{CONFIG_SUFFIXES_FOR_TASK[0]}",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} seacrowd schema for {_SUPPORTED_TASKS[0].name} and config {config_name}",
                schema=f"seacrowd_{CONFIG_SUFFIXES_FOR_TASK[0]}",
                subset_id=config_name
            ) for config_name in config_choices_folder_structure.keys()
    ]

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")
        # source schema
        if _config_schema_name == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "audio_id": datasets.Value("string"),
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "thai_text": datasets.Value("string"),
                    "audio_arr_pos_start": datasets.Sequence(datasets.Value("float")),
                    "audio_arr_pos_end": datasets.Sequence(datasets.Value("float")),
                    "phonemes": datasets.Sequence(datasets.Value("string"))
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

    @staticmethod
    def __strip_text_iterables(input: Iterable):
        if not isinstance(input, str):
            return list(map(str.strip, input))
        else:
            return input.strip()

    @classmethod
    def __read_text_files(cls, path: str, init_lines_to_skip:int=0, remove_empty_line: bool=True, strip_trailing_whitespace: bool=True):
        with open(path, "r") as f:
            data = cls.__strip_text_iterables(f.readlines())

        # pre-processing steps based on args
        if init_lines_to_skip>0:
            data = data[init_lines_to_skip:]
        if remove_empty_line:
            data = [_data for _data in data if len(_data.strip()) != 0]
        if strip_trailing_whitespace:
            data = [_data.strip() for _data in data]

        return data

    @classmethod
    def __preprocess_cc_lab_file(cls, cc_lab_file: str):
        if not cc_lab_file.endswith(".lab"):
            raise ValueError("The file isn't a .lab!")

        meta = ["audio_arr_pos_start", "audio_arr_pos_end", "phonemes"]
        raw_data = cls.__read_text_files(cc_lab_file)

        data = pd.DataFrame([dict(zip(meta, cls.__strip_text_iterables(_data.split(" ")))) for _data in raw_data])

        # since the ratio of end time and audio array length around (624.5, 625.5) is 97.50074382624219%
        # we can divide the array ratio by 625
        len_ratio = 625
        data["audio_arr_pos_start"] = data["audio_arr_pos_start"].astype("int")/len_ratio
        data["audio_arr_pos_end"] = data["audio_arr_pos_end"].astype("int")/len_ratio

        return data.to_dict(orient="list")

    @classmethod
    def __folder_walk_file_grabber(cls, folder_dir: str, ext: str=""):
        all_files = []
        for child_dir in os.listdir(folder_dir):
            _full_path = os.path.join(folder_dir, child_dir)
            if os.path.isdir(_full_path):
                all_files.extend(cls.__folder_walk_file_grabber(_full_path, ext))
            elif _full_path.endswith(ext):
                all_files.append(_full_path)
        
        return all_files

    @classmethod
    def __lotus_index_generator(cls, root_folder: str):
        index_raw_data = cls.__read_text_files(f"{root_folder}/index.txt", init_lines_to_skip=5)

        # since in the index file we have many-to-one audio recording to the same identifier of sentence values in PDsen.txt
        # except for PD data (phonetically distributed -- one sentence, multiple audios) we will filter such occurrences (for now)
        _index_candidates = [data.split("\t")[2] for data in index_raw_data]
        valid_idx = [idx for idx, val in Counter(_index_candidates).items() if val == 1 or "pd" in idx]

        # contains triplets of ("dataset number", "sequence number", "text identifier")
        metadata = ("dataset_number", "sequence_number")
        text_index_data = {
            data.split("\t")[2].strip():
                dict(zip(metadata, cls.__strip_text_iterables(data.split("\t")[:2])))
            for data in index_raw_data if data.split("\t")[2] in valid_idx}

        audio_index_data = {
            "_".join(values.values()): key for key, values in text_index_data.items()
        }

        return text_index_data, audio_index_data

    @classmethod
    def __lotus_pd_sen_generator(cls, root_folder: str, valid_idx_key: KeysView):
        text_data = [text for text in cls.__read_text_files(f"{root_folder}/PDsen.txt")]

        metadata = ("thai_text", "phonemes")
        captioned_text_data = {
            text.split("\t")[0].strip():
                dict(zip(metadata, cls.__strip_text_iterables(text.split("\t")[1:])))
            for text in text_data if text.split("\t")[0].strip() in valid_idx_key}

        return captioned_text_data


    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        # since the folder are zipped, the zipped URL containing whole resource of this dataset must be downloaded
        _all_folder_local = os.path.join(dl_manager.download_and_extract(_URL), "LOTUS")

        # Process all suplement files
        # supplement files is used regardless of the config
        # it contains the index mapper of text & audio, word list and its Phonemes
        supplement_folder = os.path.join(_all_folder_local, "Supplement")

        text_index_data, audio_index_data = self.__lotus_index_generator(supplement_folder)
        audio_level_text_data = self.__lotus_pd_sen_generator(supplement_folder, text_index_data.keys())

        _folder_structure = config_choices_folder_structure[self.config.subset_id]
        # for lab folder, it could be UC, UO, CC, or CO, depending on the folder_structure choice based on dataset config name
        _lab_foldername = _folder_structure[1][0].upper() + _folder_structure[2][0].upper() + "lab"
        
        wav_folder = os.path.join(_all_folder_local, os.path.join(*_folder_structure), "Wav")
        cc_lab_folder = os.path.join(_all_folder_local, os.path.join(*_folder_structure), _lab_foldername)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "wav_folder": wav_folder,
                    "cc_lab_folder": cc_lab_folder,
                    "captioned_data": audio_level_text_data,
                    "audio_index_data": audio_index_data}
            )]

    def _generate_examples(self, wav_folder, cc_lab_folder, captioned_data, audio_index_data) -> Tuple[int, Dict]:
        """
        This dataset contains 2 version of texts:
        1. Transcriptions per syllables and its timestamp
        2. A Text DB (in PDsen.txt) containing the whole text in Thai Script and its Romanized Morphemes
        """
        _config_schema_name = self.config.schema
        # this record list will contain short .wav files contain of Thai short audio 
        wav_record_list = self.__folder_walk_file_grabber(wav_folder, ".wav")

        idx = 1
        for audio_path in wav_record_list:
            audio_id = audio_path.split("/")[-1][:-4]
            example_data = {"id": idx, "audio_id": audio_id, "file": audio_path, "audio": audio_path}

            # for obtaining pd_text_supplement_data, we get the audio_index from the filename
            # then chaining it to the captioned data which uses the value from audio_index_data
            default_pd_text_data = {"thai_text": "", "romanized_phonemes":""}
            
            _pd_text_key = audio_index_data.get("_".join(audio_id.split("_")[1:]))
            pd_text_supplement_data = captioned_data.get(_pd_text_key, default_pd_text_data)

            example_data.update(pd_text_supplement_data)

            if _config_schema_name == "source":
                # add sequential data from cc_lab_data
                cc_lab_data = self.__preprocess_cc_lab_file(os.path.join(cc_lab_folder, audio_id + ".lab"))
                example_data.update(cc_lab_data)

                yield idx, {colname: example_data[colname] for colname in self.info.features}

            elif _config_schema_name == "seacrowd_sptext":
                # skip if the text data not found
                if pd_text_supplement_data != default_pd_text_data:
                    yield idx, {"id": idx, "path": example_data["file"], "audio": example_data["audio"], "text": example_data["thai_text"], "speaker_id": None, "metadata": {"speaker_age": None, "speaker_gender": None}}

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

            idx += 1
