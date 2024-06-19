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
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{lyu-etal-2021-goal,
    title = "Goal-Oriented Script Construction",
    author = "Lyu, Qing  and
      Zhang, Li  and
      Callison-Burch, Chris",
    editor = "Belz, Anya  and
      Fan, Angela  and
      Reiter, Ehud  and
      Sripada, Yaji",
    booktitle = "Proceedings of the 14th International Conference on Natural Language Generation",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.inlg-1.19",
    doi = "10.18653/v1/2021.inlg-1.19",
    pages = "184--200",
}
"""
_LOCAL = False
_LANGUAGES = {"ind": "id", "tha": "th", "vie": "vn"}
_DATASETNAME = "wikihow_gosc"
_DESCRIPTION = """
This dataset consists of wikiHow goal-oriented scripts. For each goal or task, sections with steps to achieve this task are
generated. Both the sections and steps within them are classified as either ordered or unordered.
"""

_HOMEPAGE = "https://github.com/veronica320/wikihow-GOSC/tree/main?tab=readme-ov-file"
_LICENSE = Licenses.MIT.value
_URL = "https://drive.google.com/uc?id=1AqAocrNFEPhBAfa5ATCj-3xMWbq659ME"

_SUPPORTED_TASKS = [Tasks.INSTRUCTION_TUNING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class WikiHowGOSCDataset(datasets.GeneratorBasedBuilder):
    """Dataset of WikiHow tasks/goals with generated steps to perform them."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_source",
            version=_SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema for {lang} language",
            schema="source",
            subset_id=f"{_DATASETNAME}_{lang}",
        )
        for lang in _LANGUAGES
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_ind_source"

    def _info(self) -> datasets.DatasetInfo:

        features = datasets.Features(
            {
                "title": datasets.Value("string"),
                "category": datasets.Value("string"),
                "sections": datasets.Sequence({"section": datasets.Value("string"), "steps": datasets.Sequence(datasets.Value("string")), "ordered": datasets.Value("int32")}),
                "ordered": datasets.Value("int32"),
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
        try:
            import gdown
        except ImportError:
            raise ImportError("Please install `gdown` to enable downloading data from google drive.")

        # Download from Google drive
        output_dir = Path.cwd() / "data" / "wikihow_gosc"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "wikihow_multilingual_scripts.zip"
        if not output_file.exists():
            gdown.download(_URL, str(output_file), fuzzy=True)
        else:
            print(f"File already downloaded: {str(output_file)}")

        data_dir = Path(dl_manager.extract(output_file))
        lang = _LANGUAGES[self.config.subset_id.split("_")[-1]]

        return [  # Train and test are in same file
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"script_{lang}.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"script_{lang}.json"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            for key, example in enumerate(data[split]):
                if "sections" not in example: # Single-section example
                    yield key, {
                        "title": example["title"],
                        "category": example["category"],
                        "sections": [{
                            "section": "",
                            "steps": example["steps"],
                            "ordered": example["ordered"],
                        }],
                        "ordered": 1
                    }
                else:
                    yield key, example