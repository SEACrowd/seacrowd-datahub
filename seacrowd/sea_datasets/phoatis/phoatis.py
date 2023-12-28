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

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

# TODO: Add BibTeX citation
_CITATION = """\
@article{dao2021intent,
      title={Intent Detection and Slot Filling for Vietnamese}, 
      author={Mai Hoang Dao and Thinh Hung Truong and Dat Quoc Nguyen},
      year={2021},
      eprint={2104.02021},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "phoatis"

_DESCRIPTION = """\
This is first public intent detection and slot filling dataset for Vietnamese. The data contains 5871 English utterances from ATIS that are manually translated by professional translators into Vietnamese.
"""

_HOMEPAGE = "https://github.com/VinAIResearch/JointIDSF/"

_LICENSE = "Licenses.UNKNOWN.value"

_URLS = {
    _DATASETNAME: {
        "syllable_train": ["https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/train/seq.in", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/train/seq.out", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/train/label"],
        "syllable_dev": ["https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/dev/seq.in", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/dev/seq.out", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/dev/label"],
        "syllable_test": ["https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/test/seq.in", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/test/seq.out", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/test/label"],
        "word_train": ["https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/train/seq.in", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/train/seq.out", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/train/label"],
        "word_dev": ["https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/dev/seq.in", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/dev/seq.out", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/dev/label"],
        "word_test": ["https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/test/seq.in", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/test/seq.out", "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/test/label"],
    },
}

_SUPPORTED_TASKS = [Tasks.INTENT_CLASSIFICATION, Tasks.SLOT_FILLING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class PhoATIS(datasets.GeneratorBasedBuilder):
    """This is first public intent detection and slot filling dataset for Vietnamese. The data contains 5871 English utterances from ATIS that are manually translated by professional translators into Vietnamese."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="phoatis_source",
            version=SOURCE_VERSION,
            description="PhoATIS source schema",
            schema="source",
            subset_id="phoatis",
        ),
        SEACrowdConfig(
            name="phoatis_seacrowd_text_seq_label",
            version=SEACROWD_VERSION,
            description="PhoATIS SEACrowd schema",
            schema="seacrowd_text_seq_label",
            subset_id="phoatis",
        ),
    ]

    DEFAULT_CONFIG_NAME = "phoatis_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "intent_label": datasets.Value("string"),
                    "slot_label": datasets.Sequence(datasets.Value("string")),
                }
            )

        elif self.config.schema == "seacrowd_text_seq_label":
            # e.g. features = schemas.kb_features
            # TODO: Choose your seacrowd schema here
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["syllable_train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["syllable_test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["syllable_dev"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        with open(filepath[0], "r+", encoding="utf8") as fw:
            data_input = fw.read()
            data_input = data_input.split('\n')
        with open(filepath[1], "r+", encoding="utf8") as fw:
            data_slot = fw.read()
            data_slot = data_slot.split('\n')
        with open(filepath[2], "r+", encoding="utf8") as fw:
            data_intent = fw.read()
            data_intent = data_intent.split('\n')

        # {
        #     "id": datasets.Value("string"),
        #     "text": datasets.Value("string"),
        #     "intent_label": datasets.ClassLabel(names=intent_label_names),
        #     "slot_label": datasets.Sequence(datasets.ClassLabel(names=slot_label_names)),
        # }
        if self.config.schema == "source":
            for idx, text in enumerate(data_input):
                example = {}
                example["id"] = str(idx)
                example["text"] = text
                example["intent_label"] = data_intent[idx]
                data_slot[idx] = data_slot[idx].split()
                example["slot_label"] = data_slot[idx]
                yield example["id"], example

        elif self.config.schema == "seacrowd_[seacrowd_schema_name]":
            # TODO: yield (key, example) tuples in the seacrowd schema
            for key, example in thing:
                yield key, example


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
