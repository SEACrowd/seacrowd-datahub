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

'''
This is a dataset containing pdfs scraped from 735 gov.my websites.
'''

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{malaysua_ai_government,
  author    = {{Malaysia-AI}},
  title     = {Crawl Malaysian Government},
  year      = {2023},  % Change to the relevant year if known
  url       = {https://huggingface.co/datasets/malaysia-ai/crawl-my-website
}
"""

_DATASETNAME = "malaysia_ai_government"

_DESCRIPTION = """\
This is a dataset containing pdfs scraped from 735 gov.my websites.
It consists of thousands of the unedited text, a link to the URL where the website was retrieved, and the name of the pdf.
"""

_HOMEPAGE = "https://huggingface.co/datasets/malaysia-ai/crawl-my-website"

_LANGUAGES = ["zlm"]

_LICENSE = Licenses.APACHE_2_0.value

_LOCAL = False


_URLS = {
    _DATASETNAME: "https://huggingface.co/datasets/malaysia-ai/crawl-my-website",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


_SUBSETS = ["gov_my", "govdocs", "muftiwp_gov_my", "myjms_mohe_gov_my"]


class MalaysiaAIGovernmentDataset(datasets.GeneratorBasedBuilder):
    """Thousands of the unedited text for Malay."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"malaysia_ai_government_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"malaysia_ai_government_{subset} source schema",
            schema="source",
            subset_id=f"{subset}",
        )
        for subset in _SUBSETS
    ] + [
        SEACrowdConfig(
            name=f"malaysia_ai_government_{subset}_seacrowd_ssp",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"malaysia_ai_government_{subset} SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id=f"{subset}",
        )
        for subset in _SUBSETS
    ]

    DEFAULT_CONFIG_NAME = "malaysia_ai_government_gov_my_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "url": datasets.Value("string"),
                }
            )

        # For example seacrowd_kb, seacrowd_t2t
        elif self.config.schema == "seacrowd_ssp":
            features = schemas.ssp_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        subset = self.config.subset_id
        # data_dir = dl_manager.download_and_extract(urls)
        # dl_manager not used since dataloader uses HF 'load_dataset'

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": urls,
                    "split": "train",
                    "subset": subset,
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str, subset: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        subset_file = subset.replace("_", ".") + ".jsonl"

        if "myjms_mohe_gov_my" in subset:
            # the last subset has jsonl in the wrong format, actually it's a txt file
            data = datasets.load_dataset("text", data_files=filepath + "/resolve/main/" + subset_file, split="train")
        else:
            data = datasets.load_dataset("/".join(filepath.split("/")[-2:]), split="train", data_files={"train": subset_file})

        for key, sample in enumerate(data):
            if self.config.schema == "source":
                yield key, {
                    "file": sample["file"] if "file" in sample else None,
                    "text": sample["text"] if "text" in sample else sample["body"],
                    "url": sample["url"] if "url" in sample else None,
                }

            elif self.config.schema == "seacrowd_ssp":
                yield key, {
                    "id": key,
                    "text": sample["text"] if "text" in sample else sample["body"],
                }
