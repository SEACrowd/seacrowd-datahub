import os
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

import datasets
import yaml

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{PhoST,
    title     = {{A High-Quality and Large-Scale Dataset for English-Vietnamese Speech Translation}},
    author    = {Linh The Nguyen and Nguyen Luong Tran and Long Doan and Manh Luong and Dat Quoc Nguyen},
    booktitle = {Proceedings of the 23rd Annual Conference of the International Speech Communication Association (INTERSPEECH)},
    year      = {2022}
}
"""

_DATASETNAME = "phost"

_DESCRIPTION = """\
PhoST is a high-quality and large-scale benchmark dataset for English-Vietnamese speech translation
with 508 audio hours, consisting of 331K triplets of (sentence-lengthed audio, English source
transcript sentence, Vietnamese target subtitle sentence).
"""

_HOMEPAGE = "https://github.com/VinAIResearch/PhoST"

_LICENSE = Licenses.CC_BY_NC_ND_4_0.value

_LOCAL = True

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION, Tasks.SPEECH_TO_TEXT_TRANSLATION, Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_LANGUAGES = ["eng", "vie"]


def seacrowd_config_constructor(src_lang, tgt_lang, schema, version):
    if src_lang == "" or tgt_lang == "":
        raise ValueError(f"Invalid src_lang {src_lang} or tgt_lang {tgt_lang}")

    if schema not in ["source", "seacrowd_sptext", "seacrowd_t2t"]:
        raise ValueError(f"Invalid schema: {schema}")

    return SEACrowdConfig(
        name="phost_{src}_{tgt}_{schema}".format(src=src_lang, tgt=tgt_lang, schema=schema),
        version=datasets.Version(version),
        description="phost schema for {schema} from {src} to {tgt}".format(schema=schema, src=src_lang, tgt=tgt_lang),
        schema=schema,
        subset_id="phost_{src}_{tgt}".format(src=src_lang, tgt=tgt_lang),
    )


class Phost(datasets.GeneratorBasedBuilder):
    """
    PhoST is a high-quality and large-scale benchmark dataset for English-Vietnamese speech translation
    with 508 audio hours, consisting of 331K triplets of (sentence-lengthed audio, English source
    transcript sentence, Vietnamese target subtitle sentence).
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        seacrowd_config_constructor("en", "vi", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("en", "vi", "seacrowd_sptext", _SEACROWD_VERSION),
        seacrowd_config_constructor("en", "vi", "seacrowd_t2t", _SEACROWD_VERSION),
    ]

    DEFAULT_CONFIG_NAME = "phost_en_vi_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "en_text": datasets.Value("string"),
                    "vi_text": datasets.Value("string"),
                    "timing": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == "seacrowd_sptext":
            features = schemas.speech_text_features
        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

            aud_path = os.path.join(data_dir, "audio_data")
            if not os.path.exists(aud_path):
                os.makedirs(aud_path)

            # loading the temp.zip and creating a zip object
            with ZipFile(os.path.join(data_dir, "train_audio.zip"), "r") as zObject:
                for member in zObject.namelist():
                    if not os.path.exists(os.path.join(aud_path, "train", member)) or not os.path.isfile(os.path.join(aud_path, "train", member)):
                        zObject.extract(member, os.path.join(aud_path, "train"))

            # dev audio files
            with ZipFile(os.path.join(data_dir, "dev_audio.zip"), "r") as zObject:
                for member in zObject.namelist():
                    if not os.path.exists(os.path.join(aud_path, "dev", member)) or not os.path.isfile(os.path.join(aud_path, "dev", member)):
                        zObject.extract(member, aud_path)
            # test audio files
            with ZipFile(os.path.join(data_dir, "test_audio.zip"), "r") as zObject:
                for member in zObject.namelist():
                    if not os.path.exists(os.path.join(aud_path, "test", member)) or not os.path.isfile(os.path.join(aud_path, "test", member)):
                        zObject.extract(member, aud_path)
            # text data
            with ZipFile(os.path.join(data_dir, "text_data.zip"), "r") as zObject:
                for member in zObject.namelist():
                    if not os.path.exists(os.path.join(data_dir, member)) or not os.path.isfile(os.path.join(data_dir, member)):
                        zObject.extract(member, data_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {"audio": os.path.join(aud_path, "train", "wav"), "text": os.path.join(data_dir, "text_data", "train")},
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {"audio": os.path.join(aud_path, "test", "wav"), "text": os.path.join(data_dir, "text_data", "test")},
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": {"audio": os.path.join(aud_path, "dev", "wav"), "text": os.path.join(data_dir, "text_data", "dev")},
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        config_names_split = self.config.name.split("_")
        src_lang = config_names_split[1]
        tgt_lang = config_names_split[2]
        track_ids = os.listdir(filepath["text"])
        timing = []
        en_sub = []
        vi_sub = []
        counter = 0
        for key, track_id in enumerate(track_ids):
            with open(os.path.join(filepath["text"], track_id, track_id + ".yaml")) as timing_file:
                timing = yaml.safe_load(timing_file)
            with open(os.path.join(filepath["text"], track_id, track_id + ".en")) as en_text:
                en_sub = [line.strip() for line in en_text]
            with open(
                os.path.join(filepath["text"], track_id, track_id + ".vi"),
            ) as vi_text:
                vi_sub = [line.strip() for line in vi_text]

            if self.config.schema == "source":
                yield key, {"file": os.path.join(filepath["audio"], track_id + ".wav"), "audio": os.path.join(filepath["audio"], track_id + ".wav"), "en_text": " ".join(en_sub), "vi_text": " ".join(vi_sub), "timing": timing}

            elif self.config.schema == "seacrowd_sptext":
                if tgt_lang not in ["en", "vi"]:
                    raise NotImplementedError(f"Target language '{tgt_lang}' is not defined.")

                yield key, {
                    "id": track_id,
                    "path": os.path.join(filepath["audio"], track_id + ".wav"),
                    "audio": os.path.join(filepath["audio"], track_id + ".wav"),
                    "text": " ".join(en_sub) if tgt_lang == "en" else " ".join(vi_sub),
                    "speaker_id": None,
                    "metadata": {
                        "speaker_age": None,
                        "speaker_gender": None,
                    },
                }

            elif self.config.schema == "seacrowd_t2t":
                if src_lang not in ["en", "vi"]:
                    raise NotImplementedError(f"Source language '{src_lang}' is not defined.")
                if tgt_lang not in ["en", "vi"]:
                    raise NotImplementedError(f"Target language '{tgt_lang}' is not defined.")
                for en_line, vi_line in zip(en_sub, vi_sub):
                    yield counter, {
                        "id": f"{track_id}_{str(counter)}",
                        "text_1": en_line if src_lang == "en" else vi_line,
                        "text_2": en_line if tgt_lang == "en" else vi_line,
                        "text_1_name": src_lang,
                        "text_2_name": tgt_lang,
                    }
                    counter += 1
            else:
                raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")
