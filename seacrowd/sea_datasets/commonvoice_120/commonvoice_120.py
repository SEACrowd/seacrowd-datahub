# coding=utf-8
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{commonvoice:2020,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
  pages = {4211--4215},
  year = 2020
}
"""

_DATASETNAME = "commonvoice_120"

_DESCRIPTION = """\
The Common Mozilla Voice dataset consists of a unique MP3 and corresponding text file.
Many of the 26119 recorded hours in the dataset also include demographic metadata like age, sex, and accent that can help improve the accuracy of speech recognition engines.
The dataset currently consists of 17127 validated hours in 104 languages, but more voices and languages are always added.

Before using this dataloader, please accept the acknowledgement at https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0 and use huggingface-cli login for authentication
"""

_HOMEPAGE = "https://commonvoice.mozilla.org/en/datasets"

_LANGUAGES = ["cnh", "ind", "tha", "vie"]
_LANG_TO_CVLANG = {"cnh": "cnh", "ind": "id", "tha": "th", "vie": "vi"}

_AGE_TO_INT = {"": None, "teens": 10, "twenties": 20, "thirties": 30, "fourties": 40, "fifties": 50, "sixties": 60, "seventies": 70, "eighties": 80}

_LICENSE = Licenses.CC0_1_0.value

# Note: the dataset is gated in HuggingFace. It's public after providing access token
_LOCAL = False

_COMMONVOICE_URL_TEMPLATE = "https://huggingface.co/datasets/mozilla-foundation/common_voice_12_0/resolve/main/"
_URLS = {"audio": _COMMONVOICE_URL_TEMPLATE + "audio/{lang}/{split}/{lang}_{split}_{shard_idx}.tar", "transcript": _COMMONVOICE_URL_TEMPLATE + "transcript/{lang}/{split}.tsv", "n_shards": _COMMONVOICE_URL_TEMPLATE + "n_shards.json"}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION, Tasks.TEXT_TO_SPEECH]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class Commonvoice120(datasets.GeneratorBasedBuilder):
    """This is the dataloader for CommonVoice 12.0 Mozilla"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = (
        *[
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang}{'_' if lang else ''}source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} source schema for {lang}",
                schema="source",
                subset_id=f"{_DATASETNAME}{'_' if lang else ''}{lang}",
            )
            for lang in ["", *_LANGUAGES]
        ],
        *[
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang}{'_' if lang else ''}seacrowd_sptext",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} SEACrowd schema for {lang}",
                schema="seacrowd_sptext",
                subset_id=f"{_DATASETNAME}{'_' if lang else ''}{lang}",
            )
            for lang in ["", *_LANGUAGES]
        ],
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "client_id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.features.Audio(sampling_rate=48_000),
                    "sentence": datasets.Value("string"),
                    "up_votes": datasets.Value("int64"),
                    "down_votes": datasets.Value("int64"),
                    "age": datasets.Value("string"),
                    "gender": datasets.Value("string"),
                    "accent": datasets.Value("string"),
                    "locale": datasets.Value("string"),
                    "segment": datasets.Value("string"),
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
        lang_code = self.config.subset_id.split("_")[-1]
        languages = [_LANG_TO_CVLANG.get(lang, lang) for lang in (_LANGUAGES if lang_code == "120" else [lang_code])]
        n_shards_path = dl_manager.download_and_extract(_URLS["n_shards"])
        with open(n_shards_path, encoding="utf-8") as f:
            n_shards = json.load(f)

        audio_urls = {}
        meta_urls = {}
        splits = ("train", "dev", "test")
        for split in splits:
            audio_urls[split] = [_URLS["audio"].format(lang=lang, split=split, shard_idx=i) for lang in languages for i in range(n_shards[lang][split])]
            meta_urls[split] = [_URLS["transcript"].format(lang=lang, split=split) for lang in languages]
        archive_paths = dl_manager.download(audio_urls)
        local_extracted_archive_paths = dl_manager.extract(archive_paths)
        meta_paths = dl_manager.download_and_extract(meta_urls)

        split_names = {
            "train": datasets.Split.TRAIN,
            "dev": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }
        return [
            datasets.SplitGenerator(
                name=split_names.get(split, split),
                gen_kwargs={
                    "local_extracted_archive_paths": local_extracted_archive_paths.get(split),
                    "audio_archives": [dl_manager.iter_archive(path) for path in archive_paths.get(split)],
                    "meta_paths": meta_paths[split],
                    "split": "train",
                },
            )
            for split in splits
        ]

    def _generate_examples(self, local_extracted_archive_paths: [Path], audio_archives: [Path], meta_paths: [Path], split: str) -> Tuple[int, Dict]:
        data_fields = list(self._info().features.keys())
        metadata = {}
        for meta_path in meta_paths:
            with open(meta_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row in reader:
                    if not row["path"].endswith(".mp3"):
                        row["path"] += ".mp3"
                    if "accents" in row:
                        row["accent"] = row["accents"]
                        del row["accents"]
                    for field in data_fields:
                        if field not in row:
                            row[field] = ""
                    metadata[row["path"]] = row

        if self.config.schema == "source":
            for i, audio_archive in enumerate(audio_archives):
                for path, file in audio_archive:
                    _, filename = os.path.split(path)
                    if filename in metadata:
                        src_result = dict(metadata[filename])
                        path = os.path.join(local_extracted_archive_paths[i], path)
                        result = {
                            "client_id": src_result["client_id"],
                            "path": path,
                            "audio": {"path": path, "bytes": file.read()},
                            "sentence": src_result["sentence"],
                            "up_votes": src_result["up_votes"],
                            "down_votes": src_result["down_votes"],
                            "age": src_result["age"],
                            "gender": src_result["gender"],
                            "accent": src_result["accent"],
                            "locale": src_result["locale"],
                            "segment": src_result["segment"],
                        }
                        yield path, result

        elif self.config.schema == "seacrowd_sptext":
            for i, audio_archive in enumerate(audio_archives):
                for path, file in audio_archive:
                    _, filename = os.path.split(path)
                    if filename in metadata:
                        src_result = dict(metadata[filename])
                        # set the audio feature and the path to the extracted file
                        path = os.path.join(local_extracted_archive_paths[i], path)
                        result = {
                            "id": src_result["path"].replace(".mp3", ""),
                            "path": path,
                            "audio": {"path": path, "bytes": file.read()},
                            "text": src_result["sentence"],
                            "speaker_id": src_result["client_id"],
                            "metadata": {
                                "speaker_age": _AGE_TO_INT[src_result["age"]],
                                "speaker_gender": src_result["gender"],
                            },
                        }
                        yield path, result
