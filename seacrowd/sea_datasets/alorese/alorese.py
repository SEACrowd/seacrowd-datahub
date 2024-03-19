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
Alorese Corpus is a collection of language data in a couple of Alorese variation (Alor and Pantar Alorese). The collection is available in video, audio, and text formats with genres ranging from Experiment or task, Stimuli, Discourse, and Written materials.
"""
from typing import Dict, List, Tuple
import pandas as pd
import xml.etree.ElementTree as ET

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses, TASK_TO_SCHEMA, SCHEMA_TO_FEATURES
from .alorese_url import _URLS_DICT

_CITATION = """\
@article{Moro2018-ms,
  title     = "The plural word hire in alorese: Contact-induced change from
               neighboring Alor-pantar languages",
  author    = "Moro, Francesca R",
  journal   = "Ocean. Linguist.",
  publisher = "Project MUSE",
  volume    =  57,
  number    =  1,
  pages     = "177--198",
  year      =  2018,
  language  = "en"
}
"""

_DATASETNAME = "alorese"
_DESCRIPTION = """\
 Alorese Corpus is a collection of language data in a couple of Alorese variation (Alor and Pantar Alorese). The collection is available in video, audio, and text formats with genres ranging from Experiment or task, Stimuli, Discourse, and Written materials.
"""
_HOMEPAGE = "	https://hdl.handle.net/1839/e10d7de5-0a6d-4926-967b-0a8cc6d21fb1"
_LANGUAGES = ["aol", "ind"]
_LICENSE = Licenses.UNKNOWN.value
_LOCAL = False

_URLS = {}
for k,v in _URLS_DICT.items():
    _URLS[k] = v["text_path"]

_SUPPORTED_TASKS = [
    Tasks.SPEECH_RECOGNITION,
    Tasks.MACHINE_TRANSLATION
] 

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

class AloreseDataset(datasets.GeneratorBasedBuilder):
    """ Alorese Corpus is a collection of language data in a couple of Alorese variation (Alor and Pantar Alorese). The collection is available in video, audio, and text formats with genres ranging from Experiment or task, Stimuli, Discourse, and Written materials."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()

    BUILDER_CONFIGS = [
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
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "nr": datasets.Value("int64"),
                    "media_id": datasets.Value("string"),
                    "annotation_aol": datasets.Value("string"),
                    "annotation_ind": datasets.Value("string"),
                    "begin_time": datasets.Value("int64"),
                    "end_time": datasets.Value("int64"),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        paths = dl_manager.download(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": paths,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        df = self._merge_all_dfs(filepath)

        if self.config.schema == "source":
            for k, row in df.iterrows():
                yield k, {
                    "nr": k + 1,
                    "media_id": row["media_id"],
                    "annotation_aol": row["annotation_aol"],
                    "annotation_ind": row["annotation_ind"],
                    "begin_time": row["begin_time"],
                    "end_time": row["end_time"],
                }

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for k, row in df.iterrows():
                yield k, {
                    "id": k + 1,
                    "text_1": row["annotation_aol"],
                    "text_2": row["annotation_ind"],
                    "text_1_name": _LANGUAGES[0],
                    "text_2_name": _LANGUAGES[1],
                }

    def _get_time_df(self, xml_tree) -> pd.DataFrame:
        time_slot_values = [
            (
                time_slot.attrib["TIME_SLOT_ID"], 
                int(time_slot.attrib["TIME_VALUE"])
            ) for time_slot in xml_tree.iter(tag = "TIME_SLOT")
        ]

        return pd.DataFrame({
            "time_slot_id": [v[0] for v in time_slot_values], 
            "time_value": [v[1] for v in time_slot_values]})

    def _get_aol_annotations(self, xml_tree) -> pd.DataFrame:
        aol_annotations = [
            (
                annotation.attrib["ANNOTATION_ID"],
                annotation.attrib["TIME_SLOT_REF1"],
                annotation.attrib["TIME_SLOT_REF2"],
                annotation.find("ANNOTATION_VALUE").text
            ) for annotation in xml_tree.iter(tag = "ALIGNABLE_ANNOTATION")
        ]

        return pd.DataFrame({
            "annotation_id": [v[0] for v in aol_annotations], 
            "time_slot_ref1": [v[1] for v in aol_annotations],
            "time_slot_ref2": [v[2] for v in aol_annotations], 
            "annotation_value": [v[3] for v in aol_annotations]})
    
    def _get_ind_annotations(self, xml_tree) -> pd.DataFrame:
        ind_annotations = [
            (
                annotation.attrib["ANNOTATION_ID"], 
                annotation.attrib["ANNOTATION_REF"],
                annotation.find("ANNOTATION_VALUE").text
            ) for annotation in xml_tree.iter(tag = "REF_ANNOTATION")
        ]

        return pd.DataFrame({
            "annotation_id": [v[0] for v in ind_annotations], 
            "annotation_ref_id": [v[1] for v in ind_annotations], 
            "annotation_value": [v[2] for v in ind_annotations]})
    
    def _get_final_df(self, xml_tree) -> pd.DataFrame:
        time_df = self._get_time_df(xml_tree)
        aol_df = self._get_aol_annotations(xml_tree)
        ind_df = self._get_ind_annotations(xml_tree)

        df1 = aol_df.merge(time_df, left_on="time_slot_ref1", right_on="time_slot_id", how="left").rename(columns={"time_value": "begin_time", "annotation_value": "annotation_aol"}).drop(columns=["time_slot_ref1", "time_slot_id"])
        df2 = df1.merge(time_df, left_on="time_slot_ref2", right_on="time_slot_id", how="left").rename(columns={"time_value": "end_time"}).drop(columns=["time_slot_ref2", "time_slot_id"])
        final_df = df2.merge(ind_df, left_on="annotation_id", right_on="annotation_ref_id", how="left").rename(columns={"annotation_value": "annotation_ind"}).drop(columns=["annotation_ref_id", "annotation_id_y","annotation_id_x"])
        
        return final_df[['annotation_aol', 'annotation_ind', 'begin_time', 'end_time']]
    
    def _merge_all_dfs(self, xml_dict) -> pd.DataFrame:
        final_df = pd.DataFrame()
        len_tracker = []
        media_ids = []

        xml_trees = [ET.parse(xml_path) for xml_path in xml_dict.values()]
        for xml_tree in xml_trees:
            cur_df = self._get_final_df(xml_tree)
            final_df = pd.concat([final_df, cur_df], axis=0)
            len_tracker.append(len(cur_df))

        media_id_list = list(xml_dict.keys())
        for i in range(len(len_tracker)):
            media_ids.extend([media_id_list[i]]*len_tracker[i])
        
        final_df["media_id"] = media_ids

        return final_df.reset_index()

# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
