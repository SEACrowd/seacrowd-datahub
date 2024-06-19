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
Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. 
This dataset enables the evaluation of mono- and multi-lingual models in high-, medium-, and low-resource languages. 
Each question has four multiple-choice answers and is linked to a short passage from the FLORES-200 dataset. 
The human annotation procedure was carefully curated to create questions that discriminate between different
 levels of generalizable language comprehension and is reinforced by extensive quality checks. While all 
 questions directly relate to the passage, the English dataset on its own proves difficult enough to 
 challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison 
 of model performance across all languages. Belebele opens up new avenues for evaluating and analyzing
  the multilingual abilities of language models and NLP systems.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
import datasets
import hashlib

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@article{,
  author    = {Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
  title     = {The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
  journal   = {arXiv preprint arXiv:2308.16884},
  year      = {2023},
  url       = {https://arxiv.org/abs/2308.16884},
}
"""

_DATASETNAME = "belebele"

_DESCRIPTION = """\
Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning
 122 language variants. This dataset enables the evaluation of mono- and multi-lingual 
 models in high-, medium-, and low-resource languages. 
 Each question has four multiple-choice answers and is linked to a short passage 
 from the FLORES-200 dataset. The human annotation procedure was carefully curated 
 to create questions that discriminate between different levels of generalizable 
 language comprehension and is reinforced by extensive quality checks. 
 While all questions directly relate to the passage, the English dataset on its own 
 proves difficult enough to challenge state-of-the-art language models. 
 Being fully parallel, this dataset enables direct comparison of model performance 
 across all languages. Belebele opens up new avenues for evaluating and analyzing 
 the multilingual abilities of language models and NLP systems.
"""

_HOMEPAGE = "https://github.com/facebookresearch/belebele"

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_URLS = {
    _DATASETNAME: "https://dl.fbaipublicfiles.com/belebele/Belebele.zip",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_SOURCE_NAMES = ["ceb_Latn", "ilo_Latn", "ind_Latn", "jav_Latn", "kac_Latn", "khm_Khmr", "lao_Laoo", "mya_Mymr", "shn_Mymr", "sun_Latn", "tgl_Latn", "tha_Thai", "vie_Latn", "war_Latn", "zsm_Latn"]
_LANGUAGES = [source.split("_")[0] for source in _SOURCE_NAMES]

_DEFAULT_LANG = "zsm"

_LOCAL = False

def config_constructor(belebele_subset: str, schema: str, version: str) -> SEACrowdConfig:
    lang = _LANGUAGES[_SOURCE_NAMES.index(belebele_subset)]
    return SEACrowdConfig(
        name="belebele_{belebele_subset}_{schema}".format(belebele_subset=belebele_subset.lower(), schema=schema),
        version=version,
        description="belebele {lang} {schema} schema".format(lang=lang, schema=schema),
        schema=schema,
        subset_id=lang,
    )

class BelebeleDataset(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    BUILDER_CONFIGS = [config_constructor(lang, "source", _SOURCE_VERSION) for lang in _SOURCE_NAMES]
    BUILDER_CONFIGS.extend([config_constructor(source_subset, "seacrowd_qa", _SEACROWD_VERSION) for source_subset in _SOURCE_NAMES])

    #add config of "belebele_source" and "belebele_seacrowd_qa" for defined "_DEFAULT_LANG"
    BUILDER_CONFIGS.extend([
        SEACrowdConfig(
            name="belebele_source",
            version=_SOURCE_VERSION,
            description=f"belebele default source schema (using language of {_DEFAULT_LANG})",
            schema="source",
            subset_id=_DEFAULT_LANG
        ),
        SEACrowdConfig(
            name="belebele_seacrowd_qa",
            version=_SEACROWD_VERSION,
            description=f"belebele default seacrowd schema for QA task (using language of {_DEFAULT_LANG})",
            schema="seacrowd_qa",
            subset_id=_DEFAULT_LANG
        )]
    )
    DEFAULT_CONFIG_NAME = "belebele_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "link": datasets.Value("string"),
                    "question_number": datasets.Value("int64"),
                    "flores_passage": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "mc_answer1": datasets.Value("string"), 
                    "mc_answer2": datasets.Value("string"), 
                    "mc_answer3": datasets.Value("string"), 
                    "mc_answer4": datasets.Value("string"), 
                    "correct_answer_num": datasets.Value("string"), 
                    "dialect": datasets.Value("string"), 
                    "ds": datasets.Value("string"), # timedate
                }
            ) 
        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        source_name = _SOURCE_NAMES[_LANGUAGES.index(self.config.subset_id)]
        path = dl_manager.download_and_extract(_URLS[_DATASETNAME])
        file = "{path}/Belebele/{source_name}.jsonl".format(path=path, source_name=source_name)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "file": file,
                },
            ),
        ]

    def _generate_examples(self, file: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            with open(file, "r", encoding="utf-8") as f: 
                for key, line in enumerate(f):
                    line = json.loads(line)
                    yield key, line
        elif self.config.schema == "seacrowd_qa":
            with open(file, "r", encoding="utf-8") as f: 
                for key, line in enumerate(f):
                    line = json.loads(line)
                    choices = [line['mc_answer1'], line['mc_answer2'], line['mc_answer3'], line['mc_answer4']]
                    answer = choices[int(line['correct_answer_num'])-1]
                    yield key, {
                        "id": key,
                        "question_id": str(line['question_number']),
                        "document_id": hashlib.md5(line['flores_passage'].encode('utf-8')).hexdigest(),
                        "question": line['question'],
                        "type": 'multiple_choice',
                        "choices": choices,
                        "context": line['flores_passage'],
                        "answer": [answer],
                        "meta": {}
                    }
        else:
            raise ValueError(f"Invalid config {self.config.name}")
