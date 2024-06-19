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
The Malaysia AI Hansard Scrape dataset contains 142,766 PDFs from the Malaysian Parliament website.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{malaysua_ai_hansard,
  author    = {{Malaysia-AI}},
  title     = {Crawl Malaysian Hansard},
  year      = {2023},  % Change to the relevant year if known
  url       = {https://huggingface.co/datasets/malaysia-ai/crawl-malaysian-hansard}

}
"""

_DATASETNAME = "malaysia_ai_hansard"

_DESCRIPTION = """\
The Malaysia AI Hansard Scrape dataset contains 142,766 PDFs from the Malaysian Parliament website.
(https://www.parlimen.gov.my/hansard-dewan-rakyat.html?uweb=dr).
It includes a JSON file for each document with the text labeled "original", page numbers "no_page" and "actual_no_page", the document's "date", and the "url" of the original PDF.
"""

_HOMEPAGE = "https://huggingface.co/datasets/malaysia-ai/crawl-malaysian-hansard"

_LANGUAGES = ["zlm"]

_LICENSE = Licenses.APACHE_2_0.value

_LOCAL = False


_URLS = {
    _DATASETNAME: "https://huggingface.co/datasets/malaysia-ai/crawl-malaysian-hansard",
}


_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MalaysiaAIHansardDataset(datasets.GeneratorBasedBuilder):
    """Malaysia AI Hansard Scrape dataset contains 142,766 PDFs from the Malaysian Parliament website."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="malaysia_ai_hansard_source",
            version=SOURCE_VERSION,
            description="malaysia_ai_hansard source schema",
            schema="source",
            subset_id="malaysia_ai_hansard",
        ),
        SEACrowdConfig(
            name="malaysia_ai_hansard_seacrowd_ssp",
            version=SEACROWD_VERSION,
            description="malaysia_ai_hansard SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id="malaysia_ai_hansard",
        ),
    ]

    DEFAULT_CONFIG_NAME = "malaysia_ai_hansard_source"

    def _info(self) -> datasets.DatasetInfo:
        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "original": datasets.Value("string"),
                    "cleaned": datasets.Value("string"),
                    "no_page": datasets.Value("string"),
                    "actual_no_page": datasets.Value("string"),
                    "date": datasets.Value("string"),
                    "url": datasets.Value("string"),
                }
            )

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
        # data_dir = dl_manager.download_and_extract(urls)
        # dl_manager not used since dataloader uses HF 'load_dataset'

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": urls,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        data = datasets.load_dataset("/".join(filepath.split("/")[-2:]), split="train")

        for key, sample in enumerate(data):
            if self.config.schema == "source":
                yield key, {
                    "original": sample["original"],
                    "cleaned": sample["cleaned"],
                    "no_page": sample["no_page"],
                    "actual_no_page": sample["actual_no_page"],
                    "date": sample["date"],
                    "url": sample["url"],
                }

            elif self.config.schema == "seacrowd_ssp":
                yield key, {
                    "id": key,
                    "text": sample["cleaned"],
                }
