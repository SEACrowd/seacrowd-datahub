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
QED - The QCRI Educational Domain Corpus (formerly QCRI AMARA Corpus) is an open multilingual collection of subtitles for educational videos and lectures collaboratively transcribed and translated over the AMARA web-based platform. It's developed by Qatar Computing Research Institute, Arabic Language Technologies Group. Along with English, it covers multiple SEA languages, such as vi (Vietnamese), my (Burnmese), jv (Javanese), id (Indonesia), th (Thai), tl (Tagalog), ms (Malaysia).
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = """\
@inproceedings{abdelali-etal-2014-amara,
    title = "The {AMARA} Corpus: Building Parallel Language Resources for the Educational Domain",
    author = "Abdelali, Ahmed  and
      Guzman, Francisco  and
      Sajjad, Hassan  and
      Vogel, Stephan",
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Declerck, Thierry  and
      Loftsson, Hrafn  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)",
    month = may,
    year = "2014",
    address = "Reykjavik, Iceland",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2014/pdf/877_Paper.pdf",
    pages = "1856--1862",
    abstract = "This paper presents the AMARA corpus of on-line educational content: a new parallel corpus of educational video subtitles, multilingually aligned for 20 languages, i.e. 20 monolingual corpora and 190 parallel corpora. This corpus includes both resource-rich languages such as English and Arabic, and resource-poor languages such as Hindi and Thai. In this paper, we describe the gathering, validation, and preprocessing of a large collection of parallel, community-generated subtitles. Furthermore, we describe the methodology used to prepare the data for Machine Translation tasks. Additionally, we provide a document-level, jointly aligned development and test sets for 14 language pairs, designed for tuning and testing Machine Translation systems. We provide baseline results for these tasks, and highlight some of the challenges we face when building machine translation systems for educational content.",
}
"""

_DATASETNAME = "qed"

_DESCRIPTION = """\
QED - The QCRI Educational Domain Corpus (formerly QCRI AMARA Corpus) is an open multilingual collection of subtitles for educational videos and lectures collaboratively transcribed and translated over the AMARA web-based platform. It's developed by Qatar Computing Research Institute, Arabic Language Technologies Group. Along with English, it covers multiple SEA languages, such as vi (Vietnamese), my (Burnmese), jv (Javanese), id (Indonesia), th (Thai), tl (Tagalog), ms (Malaysia).
"""

_HOMEPAGE = "https://opus.nlpl.eu/QED/corpus/version/QED"

_LANGUAGES = ["eng", "vie", "tha", "mya", "jav", "ind", "tgl", "zlm"]

_LICENSE = Licenses.OTHERS.value

_LOCAL = False

_FILE = "QED.{}.{}"  # E.g. QED.en-id.id

_URLS = "https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/{}.txt.zip"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION, Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "2.0.0"

_SEACROWD_VERSION = "1.0.0"


class QEDDataset(datasets.GeneratorBasedBuilder):
    """QED - The QCRI Educational Domain Corpus (formerly QCRI AMARA Corpus) is an open multilingual collection of subtitles for educational videos and lectures collaboratively transcribed and translated over the AMARA web-based platform. It's developed by Qatar Computing Research Institute, Arabic Language Technologies Group. Along with English, it covers multiple SEA languages, such as vi (Vietnamese), my (Burnmese), jv (Javanese), id (Indonesia), th (Thai), tl (Tagalog), ms (Malaysia)."""

    SEACROWD_SCHEMA = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()

    LANG_PAIRS = [
        ("en", "vi"),
        ("en", "th"),
        ("en", "my"),
        ("en", "jv"),
        ("en", "id"),
        ("en", "tl"),
        ("en", "ms"),
        ("th", "vi"),
        ("th", "my"),
        ("th", "jv"),
        ("th", "tl"),
        ("my", "tl"),
        ("my", "vi"),
        ("jv", "vi"),
        ("jv", "my"),
        ("jv", "tl"),
        ("jv", "ms"),
        ("id", "jv"),
        ("id", "th"),
        ("id", "vi"),
        ("id", "my"),
        ("id", "tl"),
        ("id", "ms"),
        ("tl", "vi"),
        ("ms", "tl"),
        ("ms", "th"),
        ("ms", "vi"),
        ("ms", "my"),
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
            name=f"{_DATASETNAME}_{lang1}-{lang2}_{lang1}_source",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} source schema {lang1} for translation from {lang1} to {lang2}",
            schema=f"source",
            subset_id=f"{_DATASETNAME}_{lang1}-{lang2}_{lang1}",
        )
        for lang1, lang2 in LANG_PAIRS
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang1}-{lang2}_{lang2}_source",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} source schema {lang2} for translation from {lang1} to {lang2}",
            schema=f"source",
            subset_id=f"{_DATASETNAME}_{lang1}-{lang2}_{lang2}",
        )
        for lang1, lang2 in LANG_PAIRS
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang1}-{lang2}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema for translation from {lang1} to {lang2} for Machine Translation task",
            schema=f"seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_{lang1}-{lang2}",
        )
        for lang1, lang2 in LANG_PAIRS
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang1}-{lang2}_{lang1}_seacrowd_ssp",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema {lang1} for translation from {lang1} to {lang2} for Self-supervised Pretraining task",
            schema=f"seacrowd_ssp",
            subset_id=f"{_DATASETNAME}_{lang1}-{lang2}_{lang1}",
        )
        for lang1, lang2 in LANG_PAIRS
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang1}-{lang2}_{lang2}_seacrowd_ssp",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema {lang2} for translation from {lang1} to {lang2} for Self-supervised Pretraining task",
            schema=f"seacrowd_ssp",
            subset_id=f"{_DATASETNAME}_{lang1}-{lang2}_{lang2}",
        )
        for lang1, lang2 in LANG_PAIRS
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_en-id_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            if len(self.config.subset_id.split("_")) == 2: # MT TASK
                lang1, lang2 = self.config.subset_id.split("_")[-1].split("-")
                features = datasets.Features(
                    {
                        "id": datasets.Value("int32"),
                        "translation": datasets.Translation(languages=(lang1, lang2)),
                    }
                )
            elif len(self.config.subset_id.split("_")) == 3: # ssp task
                lang = self.config.subset_id.split("_")[-1]
                features = datasets.Features(
                    {
                        "id": datasets.Value("int32"),
                        "text": datasets.Value("string"),
                    }
                )

        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features
        
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
        
        if len(self.config.subset_id.split("_")) == 2:
            lang_pair = self.config.subset_id.split("_")[-1]
        elif len(self.config.subset_id.split("_")) == 3:
            lang_pair = self.config.subset_id.split("_")[-2]

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

        if len(self.config.subset_id.split("_")) == 2: # MT Task

            lang_pair = self.config.subset_id.split("_")[-1]
            lang1, lang2 = lang_pair.split("-")

            l1_path = os.path.join(filepath, _FILE.format(lang_pair, lang1))
            l2_path = os.path.join(filepath, _FILE.format(lang_pair, lang2))

            if self.config.schema == "source":
                with open(l1_path, encoding="utf-8") as f1, open(l2_path, encoding="utf-8") as f2:
                    for i, (x, y) in enumerate(zip(f1, f2)):
                        yield i, {
                            "id": i,
                            "translation": {
                                lang1: x.strip(),
                                lang2: y.strip(),
                            },
                        }

            elif self.config.schema == "seacrowd_t2t":
                with open(l1_path, encoding="utf-8") as f1, open(l2_path, encoding="utf-8") as f2:
                    for i, (x, y) in enumerate(zip(f1, f2)):
                        yield i, {
                            "id": str(i),
                            "text_1": x.strip(),
                            "text_2": y.strip(),
                            "text_1_name": lang1,
                            "text_2_name": lang2,
                        },
            

        elif len(self.config.subset_id.split("_")) == 3: # SSP Task

            lang_pair = self.config.subset_id.split("_")[-2]
            lang = self.config.subset_id.split("_")[-1]

            l_path = os.path.join(filepath, _FILE.format(lang_pair, lang))

            if self.config.schema == "source":            
                with open(l_path, encoding="utf-8") as f:
                    for i, x in enumerate(f.readlines()):
                        yield i, {
                            "id": i,
                            "text": x.strip(), 
                        }


            elif self.config.schema == "seacrowd_ssp":
                with open(l_path, encoding="utf-8") as f:
                    for i, x in enumerate(f.readlines()):
                        yield i, {
                            "id": str(i),
                            "text": x.strip(),
                        }
