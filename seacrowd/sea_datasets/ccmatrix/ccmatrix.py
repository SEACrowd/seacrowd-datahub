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
The CCMatrix dataset was collected from web crawls and released by Meta. The dataset is constructed based on the margin-based bitext mining which can be applied to monolingual corpora of billions of sentences to produce high quality aligned translation data.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses, TASK_TO_SCHEMA

_CITATION = """\
@inproceedings{schwenk-etal-2021-ccmatrix,
    title = "{CCM}atrix: Mining Billions of High-Quality Parallel Sentences on the Web",
    author = "Schwenk, Holger  and
      Wenzek, Guillaume  and
      Edunov, Sergey  and
      Grave, Edouard  and
      Joulin, Armand  and
      Fan, Angela",
    editor = "Zong, Chengqing  and
      Xia, Fei  and
      Li, Wenjie  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.507",
    doi = "10.18653/v1/2021.acl-long.507",
    pages = "6490--6500",
    abstract = "We show that margin-based bitext mining in a multilingual sentence space can be successfully scaled to operate on monolingual corpora of billions of sentences. We use 32 snapshots of a curated common crawl corpus (Wenzel et al, 2019) totaling 71 billion unique sentences. Using one unified approach for 90 languages, we were able to mine 10.8 billion parallel sentences, out of which only 2.9 billions are aligned with English. We illustrate the capability of our scalable mining system to create high quality training sets from one language to any other by training hundreds of different machine translation models and evaluating them on the many-to-many TED benchmark. Further, we evaluate on competitive translation benchmarks such as WMT and WAT. Using only mined bitext, we set a new state of the art for a single system on the WMT{'}19 test set for English-German/Russian/Chinese. In particular, our English/German and English/Russian systems outperform the best single ones by over 4 BLEU points and are on par with best WMT{'}19 systems, which train on the WMT training data and augment it with backtranslation. We also achieve excellent results for distant languages pairs like Russian/Japanese, outperforming the best submission at the 2020 WAT workshop. All of the mined bitext will be freely available.",
}
"""

_DATASETNAME = "ccmatrix"

_DESCRIPTION = """\
The CCMatrix dataset was collected from web crawls and released by Meta. The dataset is constructed based on the margin-based bitext mining which can be applied to monolingual corpora of billions of sentences to produce high quality aligned translation data.
"""

_HOMEPAGE = "https://opus.nlpl.eu/CCMatrix/corpus/version/CCMatrix"

_LANGUAGES = ["jav", "eng", "vie", "ind", "tgl", "mya", "zlm"]

_LICENSE = Licenses.BSD.value

_LOCAL = False

_FILE = "CCMatrix.{}.{}"  # E.g. CCMatrix.en-nl.nl

_URLS = "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/{}.txt.zip"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

class CCMatrixDataset(datasets.GeneratorBasedBuilder):
    """The CCMatrix dataset was collected from web crawls and released by Meta. The dataset is constructed based on the margin-based bitext mining which can be applied to monolingual corpora of billions of sentences to produce high quality aligned translation data."""

    SEACROWD_SCHEMA = TASK_TO_SCHEMA[Tasks.MACHINE_TRANSLATION].lower()

    LANG_PAIRS = [
        ("en", "jv"),
        ("id", "jv"),
        ("jv", "tl"), 
        ("jv", "ms"),
        ("en", "vi"),
        ("en", "id"),
        ("en", "tl"),
        ("en", "ms"),
        ("id", "vi"),
        ("tl", "vi"),
        ("ms", "vi"),
        ("id", "tl"),
        ("id", "ms"),
        ("ms", "tl")
    ]


    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang1}-{lang2}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema for translation from {lang1} to {lang2}",
            schema="source",
            subset_id=f"{_DATASETNAME}_{lang1}-{lang2}",
        )
        for lang1, lang2 in LANG_PAIRS
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang1}-{lang2}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_{lang1}-{lang2}",
        )
        for lang1, lang2 in LANG_PAIRS
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_en-jv_source"

    def _info(self) -> datasets.DatasetInfo:
        lang1, lang2 = self.config.subset_id.split("_")[-1].split("-")

        if self.config.schema == "source":
            features = datasets.Features({
                "id": datasets.Value("int32"),
                "score": datasets.Value("float"),
                "translation": datasets.Translation(
                    languages=(lang1, lang2)
                ),
            })

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA}":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        lang_pair = self.config.subset_id.split("_")[-1]
        url = _URLS.format(lang_pair)
        data_dir = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            )
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        lang_pair = self.config.subset_id.split("_")[-1]
        lang1, lang2 = lang_pair.split("-")

        l1_path = os.path.join(
            filepath, _FILE.format(lang_pair, lang1)
        )
        l2_path = os.path.join(
            filepath, _FILE.format(lang_pair, lang2)
        )
        scores_path = os.path.join(
            filepath, _FILE.format(lang_pair, "scores")
        )

        if self.config.schema == "source":
            with open(l1_path, encoding="utf-8") as f1, \
                 open(l2_path, encoding="utf-8") as f2, \
                 open(scores_path, encoding="utf-8") as f3:
                for sentence_counter, (x, y, score) in enumerate(zip(f1, f2, f3)):
                    yield sentence_counter, {
                            "id": sentence_counter,
                            "score": score,
                            "translation": {
                                lang1: x.strip(),
                                lang2: y.strip(),
                            },
                        },

        elif self.config.schema == "seacrowd_t2t":
            with open(l1_path, encoding="utf-8") as f1, \
                 open(l2_path, encoding="utf-8") as f2, \
                 open(scores_path, encoding="utf-8") as f3:
                for sentence_counter, (x, y, _) in enumerate(zip(f1, f2, f3)):
                    yield sentence_counter, {
                            "id": str(sentence_counter),
                            "text_1": x.strip(),
                            "text_2": y.strip(),
                            "text_1_name": lang1,
                            "text_2_name": lang2,
                        },
