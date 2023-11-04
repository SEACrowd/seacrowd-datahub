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

_SEACROWD_VERSION = "1.0.0"

_NAMES = ["acm_Arab", "arz_Arab", "ceb_Latn", "fin_Latn", "hin_Deva", "ita_Latn", "khm_Khmr", "lvs_Latn", "npi_Deva", "pol_Latn", "slv_Latn", "swe_Latn", "tso_Latn", "xho_Latn", "afr_Latn", "asm_Beng", "ces_Latn", "fra_Latn", "hin_Latn", "jav_Latn", "kin_Latn", "mal_Mlym", "npi_Latn", "por_Latn", "sna_Latn", "swh_Latn", "tur_Latn", "yor_Latn", "als_Latn", "azj_Latn", "ckb_Arab", "fuv_Latn", "hrv_Latn", "jpn_Jpan", "kir_Cyrl", "mar_Deva", "nso_Latn", "snd_Arab", "tam_Taml", "ukr_Cyrl", "zho_Hans", "amh_Ethi", "bam_Latn", "dan_Latn", "gaz_Latn", "hun_Latn", "kac_Latn", "kor_Hang", "mkd_Cyrl", "nya_Latn", "ron_Latn", "som_Latn", "tel_Telu", "urd_Arab", "zho_Hant", "apc_Arab", "ben_Beng", "deu_Latn", "grn_Latn", "hye_Armn", "kan_Knda", "lao_Laoo", "mlt_Latn", "ory_Orya", "rus_Cyrl", "sot_Latn", "tgk_Cyrl", "urd_Latn", "zsm_Latn", "arb_Arab", "ben_Latn", "ell_Grek", "guj_Gujr", "ibo_Latn", "kat_Geor", "lin_Latn", "mri_Latn", "pan_Guru", "shn_Mymr", "spa_Latn", "tgl_Latn", "uzn_Latn", "zul_Latn", "arb_Latn", "bod_Tibt", "eng_Latn", "hat_Latn", "ilo_Latn", "kaz_Cyrl", "lit_Latn", "mya_Mymr", "pbt_Arab", "sin_Latn", "srp_Cyrl", "tha_Thai", "vie_Latn", "ars_Arab", "bul_Cyrl", "est_Latn", "hau_Latn", "ind_Latn", "kea_Latn", "lug_Latn", "nld_Latn", "pes_Arab", "sin_Sinh", "ssw_Latn", "tir_Ethi", "war_Latn", "ary_Arab", "cat_Latn", "eus_Latn", "heb_Hebr", "isl_Latn", "khk_Cyrl", "luo_Latn", "nob_Latn", "plt_Latn", "slk_Latn", "sun_Latn", "tsn_Latn", "wol_Latn"]

def config_constructor(lang, schema, version):
    return SEACrowdConfig(
        name="belebele_{lang}_{schema}".format(lang=lang, schema=schema),
        version=version,
        description="belebele {lang} {schema} schema".format(lang=lang, schema=schema),
        schema=schema,
        subset_id="belebele",
    )

class BelebeleDataset(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    BUILDER_CONFIGS = [config_constructor(lang, "source", _SOURCE_VERSION) for lang in _NAMES]
    BUILDER_CONFIGS.extend((config_constructor(lang, "seacrowd_qa", _SEACROWD_VERSION) for lang in _NAMES))
    DEFAULT_CONFIG_NAME = "belebele_acm_Arab_source"

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
        comps = self.config.name.split("_")
        lang = comps[1]+"_"+comps[2]
        path = dl_manager.download_and_extract(_URLS[_DATASETNAME])
        file = "{path}/Belebele/{lang}.jsonl".format(path=path, lang=lang)

        return datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "file": file,
            },
        )

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
                        "question_id": line['question_number'],
                        "document_id": hashlib.md5(line['question_number'].encode('utf-8')).hexdigest(),
                        "question": line['question'],
                        "type": 'multiple_choice',
                        "choices": choices,
                        "context": line['flores_passage'],
                        "answer": [answer],
                    }
        else:
            raise ValueError(f"Invalid config {self.config.name}")