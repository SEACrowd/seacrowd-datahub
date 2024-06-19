import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks

_DATASETNAME = "medisco"

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@INPROCEEDINGS{8629259,
  author={Qorib, Muhammad Reza and Adriani, Mirna},
  booktitle={2018 International Conference on Asian Language Processing (IALP)},
  title={Building MEDISCO: Indonesian Speech Corpus for Medical Domain},
  year={2018},
  volume={},
  number={},
  pages={133-138},
  keywords={Training;Automatic speech recognition;Medical services;Writing;Buildings;Computer science;Indonesian Automatic Speech Recognition;Medical Speech Corpus;Text Corpus},
  doi={10.1109/IALP.2018.8629259}
}
"""

_DESCRIPTION = "MEDISCO is a medical Indonesian speech corpus that contains 731 medical terms and consists of 4,680 utterances with total duration 10 hours"

_HOMEPAGE = "https://mrqorib.github.io/2018/02/01/building-medisco.html"

_LICENSE = "GNU General Public License v3.0 (gpl-3.0)"

_URLs = {
    "medisco": {
        "train": {
            "audio": "https://huggingface.co/datasets/mrqorib/MEDISCO/resolve/main/MEDISCO/train/audio.tar.gz",
            "text": "https://huggingface.co/datasets/mrqorib/MEDISCO/resolve/main/MEDISCO/train/annotation/sentences.txt",
        },
        "test": {"audio": "https://huggingface.co/datasets/mrqorib/MEDISCO/resolve/main/MEDISCO/test/audio.tar.gz", "text": "https://huggingface.co/datasets/mrqorib/MEDISCO/resolve/main/MEDISCO/test/annotation/sentences.txt"},
    }
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class Medisco(datasets.GeneratorBasedBuilder):
    "MEDISCO is a medical Indonesian speech corpus that contains 731 medical terms and consists of 4,680 utterances with total duration 10 hours"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="medisco_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="MEDISCO source schema",
            schema="source",
            subset_id="medisco",
        ),
        SEACrowdConfig(
            name="medisco_seacrowd_sptext",
            version=datasets.Version(_SEACROWD_VERSION),
            description="MEDISCO seacrowd schema",
            schema="seacrowd_sptext",
            subset_id="medisco",
        ),
    ]

    DEFAULT_CONFIG_NAME = "medisco_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=44_100),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_sptext":
            features = schemas.speech_text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[datasets.AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")],
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        base_path = _URLs["medisco"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": dl_manager.download_and_extract(base_path["train"]["audio"]), "text_path": dl_manager.download_and_extract(base_path["train"]["text"]), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": dl_manager.download_and_extract(base_path["test"]["audio"]), "text_path": dl_manager.download_and_extract(base_path["test"]["text"]), "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, text_path: Path, split: str) -> Tuple[int, Dict]:

        with open(text_path, encoding="utf-8") as f:
            texts = f.readlines()  # contains trailing \n

        for speaker_id in os.listdir(filepath):
            speaker_path = os.path.join(filepath, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
            for audio_id in os.listdir(speaker_path):
                audio_idx = int(audio_id.split(".", 1)[0]) - 1  # get 0-based index
                audio_path = os.path.join(speaker_path, audio_id)
                key = "{}_{}_{}".format(split, speaker_id, audio_idx)
                example = {
                    "id": key,
                    "speaker_id": speaker_id,
                    "path": audio_path,
                    "audio": audio_path,
                    "text": texts[audio_idx].strip(),
                }
                if self.config.schema == "seacrowd_sptext":
                    gender = speaker_id.split("-", 1)[0]
                    example["metadata"] = {
                        "speaker_gender": gender,
                        "speaker_age": None,
                    }
                yield key, example
