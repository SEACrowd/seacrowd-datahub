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

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{10.1145/3404835.3463257,
    author = {Srinivasan, Krishna and Raman, Karthik and Chen, Jiecao and Bendersky, Michael and Najork, Marc},
    title = {WIT: Wikipedia-Based Image Text Dataset for Multimodal Multilingual Machine Learning},
    year = {2021},
    isbn = {9781450380379},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3404835.3463257},
    doi = {10.1145/3404835.3463257},
    booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {2443–2449},
    numpages = {7},
    keywords = {dataset, multimodal, machine learning, wikipedia, multilingual, image-text retrieval, neural networks},
    location = {Virtual Event, Canada},
    series = {SIGIR '21}
}
"""

_DATASETNAME = "wit"

_DESCRIPTION = """\
Wikipedia-based Image Text (WIT) Dataset is a large multimodal multilingual dataset.
WIT is composed of a curated set of 37.6 million entity rich image-text examples with
11.5 million unique images across 108 Wikipedia languages. There are more than 12k
examples in each of 108 languages, with 53 languages having 100k image-text pairs.
Nine languages are spoken in the Southeast Asian region.
Since the dataset contains multiple references, following Section 3.2 of the dataset's
paper, the `seacrowd_imtext` subsets specify which reference is used for each data
instance's texts via context in metadata.
"""

_HOMEPAGE = "https://github.com/google-research-datasets/wit"

_LANGUAGES = {"ceb": "ceb", "fil": "fil", "ind": "id", "jav": "jv", "zlm": "zlm", "mya": "my", "tha": "th", "vie": "vi", "war": "war"}

_LANGUAGE_CODES = list(_LANGUAGES.values())

_LICENSE = Licenses.CC_BY_SA_3_0.value

_LOCAL = False

_URLS = {
    "train_0": "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00000-of-00010.tsv.gz",
    "train_1": "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00001-of-00010.tsv.gz",
    "train_2": "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00002-of-00010.tsv.gz",
    "train_3": "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00003-of-00010.tsv.gz",
    "train_4": "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00004-of-00010.tsv.gz",
    "train_5": "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00005-of-00010.tsv.gz",
    "train_6": "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00006-of-00010.tsv.gz",
    "train_7": "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00007-of-00010.tsv.gz",
    "train_8": "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00008-of-00010.tsv.gz",
    "train_9": "https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00009-of-00010.tsv.gz",
    "test_0": "https://storage.googleapis.com/gresearch/wit/wit_v1.test.all-00000-of-00005.tsv.gz",
    "test_1": "https://storage.googleapis.com/gresearch/wit/wit_v1.test.all-00001-of-00005.tsv.gz",
    "test_2": "https://storage.googleapis.com/gresearch/wit/wit_v1.test.all-00002-of-00005.tsv.gz",
    "test_3": "https://storage.googleapis.com/gresearch/wit/wit_v1.test.all-00003-of-00005.tsv.gz",
    "test_4": "https://storage.googleapis.com/gresearch/wit/wit_v1.test.all-00004-of-00005.tsv.gz",
    "val_0": "https://storage.googleapis.com/gresearch/wit/wit_v1.val.all-00000-of-00005.tsv.gz",
    "val_1": "https://storage.googleapis.com/gresearch/wit/wit_v1.val.all-00001-of-00005.tsv.gz",
    "val_2": "https://storage.googleapis.com/gresearch/wit/wit_v1.val.all-00002-of-00005.tsv.gz",
    "val_3": "https://storage.googleapis.com/gresearch/wit/wit_v1.val.all-00003-of-00005.tsv.gz",
    "val_4": "https://storage.googleapis.com/gresearch/wit/wit_v1.val.all-00004-of-00005.tsv.gz",
}

_SUPPORTED_TASKS = [Tasks.IMAGE_CAPTIONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class WITDataset(datasets.GeneratorBasedBuilder):
    """
    WIT is an image-text dataset from https://huggingface.co/datasets/google/wit.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = (
        [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} source schema for all 9 languages",
                schema="source",
                subset_id=f"{_DATASETNAME}",
            )
        ]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_imtext",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} SEACrowd schema for all 9 languages",
                schema="seacrowd_imtext",
                subset_id=f"{_DATASETNAME}",
            )
        ]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME}_{lang} source schema",
                schema="source",
                subset_id=f"{_DATASETNAME}_{lang}",
            )
            for lang in _LANGUAGES
        ]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang}_seacrowd_imtext",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME}_{lang} SEACrowd schema",
                schema="seacrowd_imtext",
                subset_id=f"{_DATASETNAME}_{lang}",
            )
            for lang in _LANGUAGES
        ]
    )

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "language": datasets.Value("string"),
                    "page_url": datasets.Value("string"),
                    "image_url": datasets.Value("string"),
                    "page_title": datasets.Value("string"),
                    "section_title": datasets.Value("string"),
                    "hierarchical_section_title": datasets.Value("string"),
                    "caption_reference_description": datasets.Value("string"),
                    "caption_attribution_description": datasets.Value("string"),
                    "caption_alt_text_description": datasets.Value("string"),
                    "mime_type": datasets.Value("string"),
                    "original_height": datasets.Value("int32"),
                    "original_width": datasets.Value("int32"),
                    "is_main_image": datasets.Value("bool"),
                    "attribution_passes_lang_id": datasets.Value("bool"),
                    "page_changed_recently": datasets.Value("bool"),
                    "context_page_description": datasets.Value("string"),
                    "context_section_description": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_imtext":
            features = schemas.image_text_features()
        else:
            raise ValueError(f"Invalid schema: '{self.config.schema}'")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """
        Returns SplitGenerators.
        """

        train_paths = dl_manager.download_and_extract([v for k, v in _URLS.items() if "train" in k])
        test_paths = dl_manager.download_and_extract([v for k, v in _URLS.items() if "test" in k])
        val_paths = dl_manager.download_and_extract([v for k, v in _URLS.items() if "val" in k])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": train_paths,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": test_paths,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": val_paths,
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepaths: Path, split: str) -> Tuple[int, Dict]:
        """
        Yields examples as (key, example) tuples.
        """
        subset_id = self.config.subset_id.split("_")
        if len(subset_id) > 1:
            language_list = subset_id[1]
            if language_list in _LANGUAGES:
                language_list = [_LANGUAGES[language_list]]
        else:
            language_list = _LANGUAGE_CODES

        idx = 0
        for file in filepaths:
            with open(
                file,
                "r",
                encoding="utf-8",
                newline="",
            ) as f:
                data = csv.DictReader(
                    f,
                    delimiter="\t",
                    quoting=csv.QUOTE_ALL,
                )
                if self.config.schema == "seacrowd_imtext":
                    for d in data:
                        if d["language"] in language_list:
                            text = None
                            context = None
                            if d["caption_reference_description"] != "":
                                text = d["caption_reference_description"]
                                context = "caption_reference_description"
                            elif d["caption_attribution_description"] != "":
                                text = d["caption_attribution_description"]
                                context = "caption_attribution_description"
                            else:
                                text = d["caption_alt_text_description"]
                                context = "caption_alt_text_description"
                            x = {
                                "id": idx,
                                "image_paths": [d["image_url"]],
                                "texts": text,
                                "metadata": {
                                    "context": context,
                                    "labels": None,
                                },
                            }
                            yield idx, x
                            idx += 1

                elif self.config.schema == "source":
                    for d in data:
                        if d["language"] in language_list:
                            x = {k: v if v != "" and k in self.info.features else None for k, v in d.items()}
                            yield idx, x
                            idx += 1
                else:
                    raise ValueError(f"Invalid schema: '{self.config.schema}'")
