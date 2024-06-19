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
Southeast Asian language subsets from Universal Propositions (UP) 2.0 dataset.
Semantic role labeling (SRL) is a shallow semantic parsing task that identifies “who did what to whom when, where etc” for each predicate in a sentence.
It provides an intermediate (shallow) level of a semantic representation that helps the map from syntactic parse structures to more fully-specified representations of meaning.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.common_parser import load_ud_data
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses

_CITATION = """\
@inproceedings{jindal-etal-2022-universal,
    title = "Universal {P}roposition {B}ank 2.0",
    author = "Jindal, Ishan  and
      Rademaker, Alexandre  and
      Ulewicz, Micha{l}  and
      Linh, Ha  and
      Nguyen, Huyen  and
      Tran, Khoi-Nguyen  and
      Zhu, Huaiyu  and
      Li, Yunyao",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.181",
    pages = "1700--1711",
}}
"""

_DATASETNAME = "up2"

_DESCRIPTION = """\
Southeast Asian language subsets from Universal Propositions (UP) 2.0 dataset.
Semantic role labeling (SRL) is a shallow semantic parsing task that identifies “who did what to whom when, where etc” for each predicate in a sentence.
It provides an intermediate (shallow) level of a semantic representation that helps the map from syntactic parse structures to more fully-specified representations of meaning.
"""

_HOMEPAGE = "https://universalpropositions.github.io/"

_LANGUAGES = ["ind", "vie"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CDLA_SHARING_1_0.value

_LOCAL = False

_URLS = {
    split: {
        "ind": [
            f"https://raw.githubusercontent.com/UniversalPropositions/UP_Indonesian-GSD/main/id_gsd-up-{split}.conllup",
            f"https://raw.githubusercontent.com/UniversalDependencies/UD_Indonesian-GSD/master/id_gsd-ud-{split}.conllu",
            # f"https://raw.githubusercontent.com/indolem/indolem/main/dependency_parsing/UD_Indonesian_GSD/id_gsd-ud-{split}.conllu",   # there are missing sent_id from the IndoLEM's dataset.
        ],
        "vie": [
            f"https://raw.githubusercontent.com/UniversalPropositions/UP_Vietnamese-VTB/main/vi_vtb-up-{split}.conllup",
            # f"https://raw.githubusercontent.com/UniversalDependencies/UD_Vietnamese-VTB/master/vi_vtb-ud-{split}.conllu", # new data => mismatch.
            f"https://raw.githubusercontent.com/UniversalDependencies/UD_Vietnamese-VTB/0edef6d63df949aea0494c6d4ff4f91bb1959019/vi_vtb-ud-{split}.conllu",  # r2.8
        ],
    }
    for split in ["train", "test", "dev"]
}

_SUPPORTED_TASKS = []

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class UP2Dataset(datasets.GeneratorBasedBuilder):
    """
    Southeast Asian language subsets from Universal Propositions (UP) 2.0 dataset.
    Semantic role labeling (SRL) is a shallow semantic parsing task that identifies “who did what to whom when, where etc” for each predicate in a sentence.
    It provides an intermediate (shallow) level of a semantic representation that helps the map from syntactic parse structures to more fully-specified representations of meaning.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        *[
            SEACrowdConfig(
                name=f"{_DATASETNAME}{'_' if _LANG else ''}{_LANG}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} source schema",
                schema="source",
                subset_id=f"{_DATASETNAME}{'_' if _LANG else ''}{_LANG}",
            )
            for _LANG in ["", *_LANGUAGES]
        ],
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{_LANGUAGES[0]}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "lang": datasets.Value("string"),
                    "source_sent_id": datasets.Value("string"),
                    "sent_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "id": [datasets.Value("string")],
                    "up:pred": [datasets.Value("string")],
                    "up:argheads": [datasets.Value("string")],
                    "up:argspans": [datasets.Value("string")],
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
        _subset_id = self.config.subset_id.split("_")
        if len(_subset_id) > 1:
            _lang = _subset_id[1]
            urls = {split: {_lang: urls_up_ud[_lang]} for split, urls_up_ud in _URLS.items()}
        else:
            urls = _URLS
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": data_dir["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": data_dir["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": data_dir["dev"],
                },
            ),
        ]

    def _generate_examples(self, filepaths: Dict[str, List[Path]]) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        _subset_id = self.config.subset_id.split("_")
        _langs = [_subset_id[1]] if (len(_subset_id) > 1) else _LANGUAGES

        for _lang in _langs:
            data = list(load_ud_data(filepaths[_lang][0]))
            sentid2text = {_b["sent_id"]: _b["text"] for _b in load_ud_data(filepaths[_lang][1])}

            for cur_data in data:
                txt_src = sentid2text[cur_data["sent_id"]]
                txt_up = cur_data["text"].rsplit("..........", 1)[0].rstrip(" -")
                assert txt_up == txt_src[: len(txt_up)], f"Text mismatch. Found '{txt_up}' in conllup but source is '{txt_src[:len(txt_up)]}'"
                cur_data["text"] = txt_src
                cur_data["lang"] = _lang

            if self.config.schema == "source":
                for key, example in enumerate(data):
                    yield f"{_lang}_{key}", example
