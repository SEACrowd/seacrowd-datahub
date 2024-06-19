import os
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{widiaputri-etal-5641,
  author = {Widiaputri, Ruhiyah Faradishi and Purwarianti, Ayu and Lestari, Dessi Puji and Azizah, Kurniawati and Tanaya, Dipta and Sakti, Sakriani},
  title = {Speech Recognition and Meaning Interpretation: Towards Disambiguation of Structurally Ambiguous Spoken Utterances in Indonesian},
  booktitle = {Proceedings of the EMNLP 2023},
  year = {2023}
}
"""

_DATASETNAME = "struct_amb_ind"

_DESCRIPTION = """
This dataset contains the first Indonesian speech dataset for structurally ambiguous utterances and each of transcription and two disambiguation texts.
The structurally ambiguous sentences were adapted from Types 4,5,6, and 10 of Types Of Syntactic Ambiguity in English by [Taha et al., 1983].
For each chosen type, 100 structurally ambiguous sentences in Indonesian were made by crowdsourcing.
Each Indonesian ambiguous sentence has two possible interpretations, resulting in two disambiguation text outputs for each ambiguous sentence.
Each disambiguation text is made up of two sentences. All of the sentences have been checked by linguists.
"""

_HOMEPAGE = "https://github.com/ha3ci-lab/struct_amb_ind"

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = True  # get the audio data externally from https://drive.google.com/drive/folders/1QeaptstBgwGYO6THGkZHHViExrogCMUj
_LANGUAGES = ["ind"]

_URL_TEMPLATES = {
    "keys": "https://raw.githubusercontent.com/ha3ci-lab/struct_amb_ind/main/keys/train_dev_test_spk_keys/",
    "text": "https://raw.githubusercontent.com/ha3ci-lab/struct_amb_ind/main/text/",
}

_URLS = {
    "split_train": _URL_TEMPLATES["keys"] + "train_spk",
    "split_dev": _URL_TEMPLATES["keys"] + "dev_spk",
    "split_test": _URL_TEMPLATES["keys"] + "test_spk",
    "text_transcript": _URL_TEMPLATES["text"] + "ID_amb_disam_transcript.txt",
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class StructAmbInd(datasets.GeneratorBasedBuilder):
    """
    This dataset contains the first Indonesian speech dataset for structurally ambiguous utterances and each of transcription and two disambiguation texts.
    This dataloader does NOT contain the additional training data for as mentioned in the _HOMEPAGE, as it is already implemented in the dataloader "indspeech_news_lvcsr".
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_sptext",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_sptext",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "amb_transcript": datasets.Value("string"),
                    "disam_text": datasets.Value("string"),
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
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # The data_dir configuration is required ONLY for the audio_urls.
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        # load the local audio folders
        audio_urls = [data_dir + "/" + f"{gender}{_id:02}.zip" for gender in ["F", "M"] for _id in range(1, 12, 1)]
        audio_files_dir = [Path(dl_manager.extract(audio_url)) / audio_url.split("/")[-1][:-4] for audio_url in audio_urls]
        # load the speaker splits and transcript
        split_train = Path(dl_manager.download(_URLS["split_train"]))
        split_dev = Path(dl_manager.download(_URLS["split_dev"]))
        split_test = Path(dl_manager.download(_URLS["split_test"]))
        text_transcript = Path(dl_manager.download(_URLS["text_transcript"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": split_train, "transcript": text_transcript, "audio_files_dir": audio_files_dir},
            ),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"split": split_dev, "transcript": text_transcript, "audio_files_dir": audio_files_dir}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": split_test, "transcript": text_transcript, "audio_files_dir": audio_files_dir},
            ),
        ]

    def _generate_examples(self, split: Path, transcript: Path, audio_files_dir: List[Path]) -> Tuple[int, Dict]:
        speaker_ids = open(split, "r").readlines()
        speaker_ids = [id.replace("\n", "") for id in speaker_ids]
        speech_folders = [audio_folder for audio_folder in audio_files_dir if audio_folder.name.split("/")[-1] in speaker_ids]
        speech_files = list(chain(*[list(map((str(speech_folder) + "/").__add__, os.listdir(speech_folder))) for speech_folder in speech_folders]))

        transcript = open(transcript, "r").readlines()
        transcript = [sent.replace("\n", "").split("|") for sent in transcript]
        transcript_dict = {sent[0]: {"amb_transcript": sent[1], "disam_text": sent[2]} for sent in transcript}

        for key, aud_file in enumerate(speech_files):
            aud_id = aud_file.split("/")[-1][:-4]
            aud_info = aud_id.split("_")
            if self.config.schema == "source":
                row = {
                    "id": aud_id,
                    "speaker_id": aud_info[1],
                    "path": aud_file,
                    "audio": aud_file,
                    "amb_transcript": transcript_dict[aud_id]["amb_transcript"],
                    "disam_text": transcript_dict[aud_id]["disam_text"],
                }
                yield key, row
            elif self.config.schema == "seacrowd_sptext":
                row = {
                    "id": aud_id,
                    "path": aud_file,
                    "audio": aud_file,
                    "text": transcript_dict[aud_id]["amb_transcript"],
                    "speaker_id": aud_info[1],
                    "metadata": {
                        "speaker_age": None,
                        "speaker_gender": aud_info[1][0],
                    },
                }
                yield key, row
            else:
                raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")
