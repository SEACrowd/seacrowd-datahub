# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

# no paper citation
_CITATION = """\
"""
_DATASETNAME = "thai_ser"
_DESCRIPTION = """\
THAI SER dataset consists of 5 main emotions assigned to actors: Neutral,
Anger, Happiness, Sadness, and Frustration. The recordings were 41 hours,
36 minutes long (27,854 utterances), and were performed by 200 professional
actors (112 female, 88 male) and directed by students, former alumni, and
professors from the Faculty of Arts, Chulalongkorn University. The THAI SER
contains 100 recordings and is separated into two main categories: Studio and
Zoom. Studio recordings also consist of two studio environments: Studio A, a
controlled studio room with soundproof walls, and Studio B, a normal room
without soundproof or noise control.
"""
_HOMEPAGE = "https://github.com/vistec-AI/dataset-releases/releases/tag/v1"
_LANGUAGES = ["tha"]
_LICENSE = Licenses.CC_BY_SA_4_0.value
_LOCAL = False

_URLS = {
    "actor_demography": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/actor_demography.json",
    "emotion_label": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/emotion_label.json",
    "studio": {
        "studio1-10": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/studio1-10.zip",
        "studio11-20": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/studio11-20.zip",
        "studio21-30": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/studio21-30.zip",
        "studio31-40": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/studio31-40.zip",
        "studio41-50": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/studio41-50.zip",
        "studio51-60": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/studio51-60.zip",
        "studio61-70": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/studio61-70.zip",
        "studio71-80": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/studio71-80.zip",
    },
    "zoom": {"zoom1-10": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/zoom1-10.zip", "zoom11-20": "https://github.com/vistec-AI/dataset-releases/releases/download/v1/zoom11-20.zip"},
}
_URLS["studio_zoom"] = {**_URLS["studio"], **_URLS["zoom"]}

_SUPPORTED_TASKS = [Tasks.SPEECH_EMOTION_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ThaiSER(datasets.GeneratorBasedBuilder):
    """Thai speech emotion recognition dataset THAI SER contains 100 recordings (80 studios and 20 zooms)."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "speech"
    _LABELS = ["Neutral", "Angry", "Happy", "Sad", "Frustrated"]

    BUILDER_CONFIGS = [
        # studio
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}",
        ),
        # studio and zoom
        SEACrowdConfig(
            name=f"{_DATASETNAME}_include_zoom_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_include_zoom",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_include_zoom_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_include_zoom",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=44_100),
                    "speaker_id": datasets.Value("string"),
                    "labels": datasets.ClassLabel(names=self._LABELS),
                    "majority_emo": datasets.Value("string"),  # 'None' when no single majority
                    "annotated": datasets.Value("string"),
                    "agreement": datasets.Value("float32"),
                    "metadata": {
                        "speaker_age": datasets.Value("int64"),
                        "speaker_gender": datasets.Value("string"),
                    },
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            # same as schemas.speech_features(self._LABELS) except for sampling_rate
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=44_100),
                    "speaker_id": datasets.Value("string"),
                    "labels": datasets.ClassLabel(names=self._LABELS),
                    "metadata": {
                        "speaker_age": datasets.Value("int64"),
                        "speaker_gender": datasets.Value("string"),
                    },
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        setting = "studio_zoom" if "zoom" in self.config.name else "studio"

        data_paths = {"actor_demography": Path(dl_manager.download_and_extract(_URLS["actor_demography"])), "emotion_label": Path(dl_manager.download_and_extract(_URLS["emotion_label"])), setting: {}}
        for url_name, url_path in _URLS[setting].items():
            data_paths[setting][url_name] = Path(dl_manager.download_and_extract(url_path))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "actor_demography_filepath": data_paths["actor_demography"],
                    "emotion_label_filepath": data_paths["emotion_label"],
                    "data_filepath": data_paths[setting],
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, actor_demography_filepath: Path, emotion_label_filepath: Path, data_filepath: Dict[str, Union[Path, Dict]], split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # read actor_demography file
        with open(actor_demography_filepath, "r", encoding="utf-8") as actor_demography_file:
            actor_demography = json.load(actor_demography_file)
        actor_demography_dict = {actor["Actor's ID"]: {"speaker_age": actor["Age"], "speaker_gender": actor["Sex"].lower()} for actor in actor_demography["data"]}

        # read emotion_label file
        with open(emotion_label_filepath, "r", encoding="utf-8") as emotion_label_file:
            emotion_label = json.load(emotion_label_file)

        # iterate through data folders
        for folder_path in data_filepath.values():
            flac_files = glob.glob(os.path.join(folder_path, "**/*.flac"), recursive=True)
            # iterate through recordings
            for audio_path in flac_files:
                id = audio_path.split("/")[-1]
                speaker_id = id.split("_")[2].strip("actor")
                # labels in emotion_label are incomplete, labels only provided for microphone types: mic, con
                # otherwise, obtain label from id for scripted utterances and skip sample for the improvised utterances
                if id in emotion_label.keys():
                    assigned_emo = emotion_label[id][0]["assigned_emo"]
                    majority_emo = emotion_label[id][0]["majority_emo"]
                    agreement = emotion_label[id][0]["agreement"]
                    annotated = emotion_label[id][0]["annotated"]
                else:
                    if "script" in id:
                        label = id.split("_")[-1][0]  # Emotion (1 = Neutral, 2 = Angry, 3 = Happy, 4 = Sad, 5 = Frustrated)
                        assigned_emo = self._LABELS[int(label) - 1]
                        majority_emo = agreement = annotated = None
                    else:
                        continue

                if self.config.schema == "source":
                    example = {
                        "id": id.strip(".flac"),
                        "path": audio_path,
                        "audio": audio_path,
                        "speaker_id": speaker_id,
                        "labels": assigned_emo,
                        "majority_emo": majority_emo,
                        "agreement": agreement,
                        "annotated": annotated,
                        "metadata": {"speaker_age": actor_demography_dict[speaker_id]["speaker_age"], "speaker_gender": actor_demography_dict[speaker_id]["speaker_gender"]},
                    }
                elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                    example = {
                        "id": id.strip(".flac"),
                        "path": audio_path,
                        "audio": audio_path,
                        "speaker_id": speaker_id,
                        "labels": assigned_emo,
                        "metadata": {"speaker_age": actor_demography_dict[speaker_id]["speaker_age"], "speaker_gender": actor_demography_dict[speaker_id]["speaker_gender"]},
                    }

                yield id.strip(".flac"), example
