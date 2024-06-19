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

"""
Alorese Corpus is a collection of language data in a couple of Alorese variation (Alor and Pantar Alorese). The collection is available in video, audio, and text formats with genres
ranging from Experiment or task, Stimuli, Discourse, and Written materials.
"""
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.sea_datasets.alorese.alorese_url import _URLS_DICT
from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{Moro2018-ms,
  title     = "The plural word hire in alorese: Contact-induced change from
               neighboring Alor-pantar languages",
  author    = "Moro, Francesca R",
  journal   = "Oceanic Linguistics",
  publisher = "University of Hawai'i Press",
  volume    =  57,
  number    =  1,
  pages     = "177--198",
  year      =  2018,
  language  = "en"
}
"""

_DATASETNAME = "alorese"
_DESCRIPTION = """\
 Alorese Corpus is a collection of language data in a couple of Alorese variation (Alor and Pantar Alorese). The collection is available in video, audio, and text formats with genres
 ranging from Experiment or task, Stimuli, Discourse, and Written materials.
"""
_HOMEPAGE = "https://hdl.handle.net/1839/e10d7de5-0a6d-4926-967b-0a8cc6d21fb1"
_LANGUAGES = ["aol", "ind"]
_LICENSE = Licenses.UNKNOWN.value
_LOCAL = False

_URLS = _URLS_DICT

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION, Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class AloreseDataset(datasets.GeneratorBasedBuilder):
    """Alorese Corpus is a collection of language data in a couple of Alorese variation (Alor and Pantar Alorese). The collection is available in video, audio, and text formats with genres ranging
    from Experiment or task, Stimuli, Discourse, and Written materials."""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}"
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd for text2text schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_sptext",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd for sptext schema",
            schema="seacrowd_sptext",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "nr": datasets.Value("int64"),
                    "media_id": datasets.Value("string"),
                    "speaker_id": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16000),
                    "annotation_aol": datasets.Value("string"),
                    "annotation_ind": datasets.Value("string"),
                    "begin_time": datasets.Value("int64"),
                    "end_time": datasets.Value("int64"),
                }
            )

        elif self.config.schema == "seacrowd_sptext":
            features = schemas.speech_text_features

        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        else:
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:

        if self.config.schema == "seacrowd_t2t":
            filepath = {k: v["text_path"] for k, v in _URLS.items()}
            paths = dl_manager.download(filepath)
        else:
            paths = dl_manager.download(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": paths,
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:

        if self.config.schema == "source":
            source_df = self._get_source_df(filepath)

            for k, row in source_df.iterrows():
                yield k, {
                    "nr": k + 1,
                    "media_id": row["media_id"],
                    "speaker_id": row["speaker_id"],
                    "audio": row["audio_path"],
                    "annotation_aol": row["annotation_aol"],
                    "annotation_ind": row["annotation_ind"],
                    "begin_time": row["begin_time"],
                    "end_time": row["end_time"],
                }

        elif self.config.schema == "seacrowd_t2t":
            caption_df = self._merge_text_dfs(filepath)

            for k, row in caption_df.iterrows():
                yield k, {
                    "id": k + 1,
                    "text_1": row["annotation_aol"],
                    "text_2": row["annotation_ind"],
                    "text_1_name": _LANGUAGES[0],
                    "text_2_name": _LANGUAGES[1],
                }

        elif self.config.schema == "seacrowd_sptext":
            sptext_df = self._get_sptext_df(filepath)

            for k, row in sptext_df.iterrows():
                yield k, {
                    "id": k + 1,
                    "path": row["audio_path"],
                    "audio": row["audio_path"],
                    "text": row["annotation_aol"],
                    "speaker_id": row["speaker_id"],
                    "metadata": {
                        "speaker_age": None,
                        "speaker_gender": None
                }}

    def _get_time_df(self, xml_tree) -> pd.DataFrame:
        time_slot_values = [(time_slot.attrib["TIME_SLOT_ID"], int(time_slot.attrib["TIME_VALUE"])) for time_slot in xml_tree.iter(tag="TIME_SLOT")]

        return pd.DataFrame({"time_slot_id": [v[0] for v in time_slot_values], "time_value": [v[1] for v in time_slot_values]})

    def _get_aol_annotations(self, xml_tree) -> pd.DataFrame:
        aol_annotations = [(annotation.attrib["ANNOTATION_ID"], annotation.attrib["TIME_SLOT_REF1"], annotation.attrib["TIME_SLOT_REF2"], annotation.find("ANNOTATION_VALUE").text) for annotation in xml_tree.iter(tag="ALIGNABLE_ANNOTATION")]

        return pd.DataFrame({"annotation_id": [v[0] for v in aol_annotations], "time_slot_ref1": [v[1] for v in aol_annotations], "time_slot_ref2": [v[2] for v in aol_annotations], "annotation_value": [v[3] for v in aol_annotations]})

    def _get_ind_annotations(self, xml_tree) -> pd.DataFrame:
        ind_annotations = [(annotation.attrib["ANNOTATION_ID"], annotation.attrib["ANNOTATION_REF"], annotation.find("ANNOTATION_VALUE").text) for annotation in xml_tree.iter(tag="REF_ANNOTATION")]

        return pd.DataFrame({"annotation_id": [v[0] for v in ind_annotations], "annotation_ref_id": [v[1] for v in ind_annotations], "annotation_value": [v[2] for v in ind_annotations]})

    def _get_text_df(self, xml_tree) -> pd.DataFrame:
        time_df = self._get_time_df(xml_tree)
        aol_df = self._get_aol_annotations(xml_tree)
        ind_df = self._get_ind_annotations(xml_tree)

        df1 = aol_df.merge(time_df, left_on="time_slot_ref1", right_on="time_slot_id", how="left").rename(columns={"time_value": "begin_time", "annotation_value": "annotation_aol"}).drop(columns=["time_slot_ref1", "time_slot_id"])
        df2 = df1.merge(time_df, left_on="time_slot_ref2", right_on="time_slot_id", how="left").rename(columns={"time_value": "end_time"}).drop(columns=["time_slot_ref2", "time_slot_id"])
        final_df = df2.merge(ind_df, left_on="annotation_id", right_on="annotation_ref_id", how="left").rename(columns={"annotation_value": "annotation_ind"}).drop(columns=["annotation_ref_id", "annotation_id_y", "annotation_id_x"])

        return final_df[["annotation_aol", "annotation_ind", "begin_time", "end_time"]]

    def _merge_text_dfs(self, xml_dict) -> pd.DataFrame:
        final_df = pd.DataFrame()
        len_tracker = []
        media_ids = []

        xml_trees = [ET.parse(xml_path) for xml_path in xml_dict.values()]
        for xml_tree in xml_trees:
            cur_df = self._get_text_df(xml_tree)
            final_df = pd.concat([final_df, cur_df], axis=0)
            len_tracker.append(len(cur_df))

        media_id_list = list(xml_dict.keys())
        for i in range(len(len_tracker)):
            media_ids.extend([media_id_list[i]] * len_tracker[i])

        final_df["media_id"] = media_ids

        return final_df.reset_index()

    def _groupby_caption_by_media_ids(self, caption_df: pd.DataFrame) -> pd.DataFrame:
        caption_df = (
            caption_df.groupby("media_id")
            .agg({"annotation_aol": lambda x: " ".join([str(value) if value is not None else "<NONE>" for value in x]), "annotation_ind": lambda x: " ".join([str(value) if value is not None else "<NONE>" for value in x])})
            .reset_index()
        )
        return caption_df

    def _get_sptext_df(self, complete_dict) -> pd.DataFrame:
        xml_dict = {k: v["text_path"] for k, v in complete_dict.items()}

        audio_df = pd.DataFrame({"media_id": [k for k in complete_dict.keys()], "speaker_id": [k.split("_")[-1] for k in complete_dict.keys()], "audio_path": [v["audio_path"] for v in complete_dict.values()]})
        caption_df = self._groupby_caption_by_media_ids(self._merge_text_dfs(xml_dict))

        df = caption_df.merge(audio_df, on="media_id", how="inner")

        return df[["media_id", "speaker_id", "audio_path", "annotation_aol", "annotation_ind"]]

    def _get_source_df(self, complete_dict) -> pd.DataFrame:
        xml_dict = {k: v["text_path"] for k, v in complete_dict.items()}

        audio_df = pd.DataFrame({"media_id": [k for k in complete_dict.keys()], "speaker_id": [k.split("_")[-1] for k in complete_dict.keys()], "audio_path": [v["audio_path"] for v in complete_dict.values()]})
        text_df = self._merge_text_dfs(xml_dict)

        df = text_df.merge(audio_df, on="media_id", how="inner")

        return df[["media_id", "speaker_id", "audio_path", "annotation_aol", "annotation_ind", "begin_time", "end_time"]]
