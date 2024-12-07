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
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@article{nllb2022,
  author    = {NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe
  Kalbassi,  Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic
  Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon
  Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami,
  Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Jeff Wang},
  title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year      = {2022}
}
@inproceedings{,
  title={The FLORES-101  Evaluation Benchmark for Low-Resource and Multilingual Machine Translation},
  author={Goyal, Naman and Gao, Cynthia and Chaudhary, Vishrav and Chen, Peng-Jen and Wenzek, Guillaume and Ju, Da and Krishnan, Sanjana and Ranzato, Marc'Aurelio and Guzm\'{a}n, Francisco and Fan, Angela},
  year={2021}
}
@inproceedings{,
  title={Two New Evaluation Datasets for Low-Resource Machine Translation: Nepali-English and Sinhala-English},
  author={Guzm\'{a}n, Francisco and Chen, Peng-Jen and Ott, Myle and Pino, Juan and Lample, Guillaume and Koehn, Philipp and Chaudhary, Vishrav and Ranzato, Marc'Aurelio},
  journal={arXiv preprint arXiv:1902.01382},
  year={2019}
}
"""

_DATASETNAME = "flores200"

_DESCRIPTION = """\
The creation of FLORES-200 doubles the existing language coverage of FLORES-101.
Given the nature of the new languages, which have less standardization and require
more specialized professional translations, the verification process became more complex.
This required modifications to the translation workflow. FLORES-200 has several languages
which were not translated from English. Specifically, several languages were translated
from Spanish, French, Russian and Modern Standard Arabic. Moreover, FLORES-200 also
includes two script alternatives for four languages. FLORES-200 consists of translations
from 842 distinct web articles, totaling 3001 sentences. These sentences are divided
into three splits: dev, devtest, and test (hidden). On average, sentences are approximately
21 words long.
"""

_HOMEPAGE = "https://github.com/facebookresearch/flores"

_LANGUAGES = [
    "ace",
    "ban",
    "bjn",
    "bug",
    "ceb",
    "ilo",
    "ind",
    "jav",
    "kac",
    "khm",
    "lao",
    "lus",
    "min",
    "mya",
    "pag",
    "shn",
    "sun",
    "tgl",
    "tha",
    "vie",
    "war",
    "zsm",
]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LANGUAGE_NAMES = [
    "ace_Arab",
    "ace_Latn",
    "acm_Arab",
    "acq_Arab",
    "aeb_Arab",
    "afr_Latn",
    "ajp_Arab",
    "aka_Latn",
    "als_Latn",
    "amh_Ethi",
    "apc_Arab",
    "arb_Arab",
    "arb_Latn",
    "ars_Arab",
    "ary_Arab",
    "arz_Arab",
    "asm_Beng",
    "ast_Latn",
    "awa_Deva",
    "ayr_Latn",
    "azb_Arab",
    "azj_Latn",
    "bak_Cyrl",
    "bam_Latn",
    "ban_Latn",
    "bel_Cyrl",
    "bem_Latn",
    "ben_Beng",
    "bho_Deva",
    "bjn_Arab",
    "bjn_Latn",
    "bod_Tibt",
    "bos_Latn",
    "bug_Latn",
    "bul_Cyrl",
    "cat_Latn",
    "ceb_Latn",
    "ces_Latn",
    "cjk_Latn",
    "ckb_Arab",
    "cmn_Hans",
    "cmn_Hant",
    "crh_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "dik_Latn",
    "dyu_Latn",
    "dzo_Tibt",
    "ell_Grek",
    "eng_Latn",
    "epo_Latn",
    "est_Latn",
    "eus_Latn",
    "ewe_Latn",
    "fao_Latn",
    "fij_Latn",
    "fin_Latn",
    "fon_Latn",
    "fra_Latn",
    "fur_Latn",
    "fuv_Latn",
    "gla_Latn",
    "gle_Latn",
    "glg_Latn",
    "grn_Latn",
    "guj_Gujr",
    "hat_Latn",
    "hau_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hne_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Armn",
    "ibo_Latn",
    "ilo_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jav_Latn",
    "jpn_Jpan",
    "kab_Latn",
    "kac_Latn",
    "kam_Latn",
    "kan_Knda",
    "kas_Arab",
    "kas_Deva",
    "kat_Geor",
    "knc_Arab",
    "knc_Latn",
    "kaz_Cyrl",
    "kbp_Latn",
    "kea_Latn",
    "khm_Khmr",
    "kik_Latn",
    "kin_Latn",
    "kir_Cyrl",
    "kmb_Latn",
    "kmr_Latn",
    "kon_Latn",
    "kor_Hang",
    "lao_Laoo",
    "lij_Latn",
    "fil_Latn",
    "lim_Latn",
    "lin_Latn",
    "lit_Latn",
    "lmo_Latn",
    "ltg_Latn",
    "ltz_Latn",
    "lua_Latn",
    "lug_Latn",
    "luo_Latn",
    "lus_Latn",
    "lvs_Latn",
    "mag_Deva",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "min_Arab",
    "min_Latn",
    "mkd_Cyrl",
    "plt_Latn",
    "mlt_Latn",
    "mni_Beng",
    "khk_Cyrl",
    "mos_Latn",
    "mri_Latn",
    "mya_Mymr",
    "nld_Latn",
    "nno_Latn",
    "nob_Latn",
    "npi_Deva",
    "nqo_Nkoo",
    "nso_Latn",
    "nus_Latn",
    "nya_Latn",
    "oci_Latn",
    "gaz_Latn",
    "ory_Orya",
    "pag_Latn",
    "pan_Guru",
    "pap_Latn",
    "pes_Arab",
    "pol_Latn",
    "por_Latn",
    "prs_Arab",
    "pbt_Arab",
    "quy_Latn",
    "ron_Latn",
    "run_Latn",
    "rus_Cyrl",
    "sag_Latn",
    "san_Deva",
    "sat_Olck",
    "scn_Latn",
    "shn_Mymr",
    "sin_Sinh",
    "slk_Latn",
    "slv_Latn",
    "smo_Latn",
    "sna_Latn",
    "snd_Arab",
    "som_Latn",
    "sot_Latn",
    "spa_Latn",
    "srd_Latn",
    "srp_Cyrl",
    "ssw_Latn",
    "sun_Latn",
    "swe_Latn",
    "swh_Latn",
    "szl_Latn",
    "tam_Taml",
    "tat_Cyrl",
    "tel_Telu",
    "tgk_Cyrl",
    "tha_Thai",
    "tir_Ethi",
    "taq_Latn",
    "taq_Tfng",
    "tpi_Latn",
    "tsn_Latn",
    "tso_Latn",
    "tuk_Latn",
    "tum_Latn",
    "tur_Latn",
    "twi_Latn",
    "uig_Arab",
    "ukr_Cyrl",
    "umb_Latn",
    "urd_Arab",
    "uzn_Latn",
    "vec_Latn",
    "vie_Latn",
    "war_Latn",
    "wol_Latn",
    "xho_Latn",
    "ydd_Hebr",
    "yor_Latn",
    "yue_Hant",
    "zgh_Tfng",
    "zsm_Latn",
    "zul_Latn",
]

_LICENSE = Licenses.CC_BY_NC_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://huggingface.co/datasets/openlanguagedata/flores_plus",
}

_SPLITS = ["dev", "devtest"]

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SUPPORTED_SCHEMA_STRINGS = [f"seacrowd_{str(TASK_TO_SCHEMA[task]).lower()}" for task in _SUPPORTED_TASKS]

_SCHEMAS = [str(TASK_TO_SCHEMA[task]) for task in _SUPPORTED_TASKS]

_SOURCE_VERSION = "2.0.0"

_SEACROWD_VERSION = "2024.12.07"


@dataclass
class Flores200SourceConfig(SEACrowdConfig):
    """BuilderConfig for Source Schema."""
    language_name: str = None
    
@dataclass
class Flores200SeacrowdConfig(SEACrowdConfig):
    """BuilderConfig for SEACrowd Schema."""

    first_language_name: str = None
    second_language_name: str = None


class Flores200(datasets.GeneratorBasedBuilder):
    """
    The creation of FLORES-200 doubles the existing language coverage of FLORES-101.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []

    for first_lang_name in _LANGUAGE_NAMES:
        source_subset_id = f"{_DATASETNAME}_{first_lang_name}"
        
        BUILDER_CONFIGS.append(
            Flores200SourceConfig(
                name=f"{source_subset_id}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} source schema",
                schema="source",
                subset_id=source_subset_id,
                language_name=first_lang_name,
            )
        )

        for second_lang_name in _LANGUAGE_NAMES:
            if first_lang_name == second_lang_name or ((first_lang_name.split("_")[0] not in _LANGUAGES) and (second_lang_name.split("_")[0] not in _LANGUAGES)):
                continue

            subset_id = f"{_DATASETNAME}_{first_lang_name}_{second_lang_name}"

            seacrowd_schema_config: list[SEACrowdConfig] = []

            for seacrowd_schema in _SUPPORTED_SCHEMA_STRINGS:

                seacrowd_schema_config.append(
                    Flores200SeacrowdConfig(
                        name=f"{subset_id}_{seacrowd_schema}",
                        version=SEACROWD_VERSION,
                        description=f"{_DATASETNAME} {seacrowd_schema} schema",
                        schema=f"{seacrowd_schema}",
                        subset_id=subset_id,
                        first_language_name=first_lang_name,
                        second_language_name=second_lang_name,
                    )
                )

            BUILDER_CONFIGS.extend(seacrowd_schema_config)

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{_LANGUAGE_NAMES[0]}_{_LANGUAGE_NAMES[1]}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source": 
            features = datasets.Features({
                'id': datasets.Value(dtype='int64', id=None),
                'iso_639_3': datasets.Value(dtype='string', id=None),
                'iso_15924': datasets.Value(dtype='string', id=None),
                'glottocode': datasets.Value(dtype='string', id=None),
                'text': datasets.Value(dtype='string', id=None),
                'url': datasets.Value(dtype='string', id=None),
                'domain': datasets.Value(dtype='string', id=None),
                'topic': datasets.Value(dtype='string', id=None),
                'has_image': datasets.Value(dtype='string', id=None),
                'has_hyperlink': datasets.Value(dtype='string', id=None),
                'last_updated': datasets.Value(dtype='string', id=None)
            })

        else:
            schema = str(self.config.schema).lstrip(f"{_DATASETNAME}_seacrowd_").upper()

            if schema in _SCHEMAS:
                features = SCHEMA_TO_FEATURES[schema]

            else:
                raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        
        hf_path = '/'.join(_URLS[_DATASETNAME].split('/')[-2:])
        flores_dset = datasets.load_dataset(hf_path, trust_remote_code=True)

        if self.config.schema == 'source':
            return [
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "split_df": flores_dset[split].to_pandas(),
                    },
                )
                for split in _SPLITS
            ]            
        else:
            return [
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "split_df": flores_dset[split].to_pandas(),
                    },
                )
                for split in _SPLITS
            ]

    def _generate_examples(self, split_df: pd.DataFrame) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == 'source':
            
            lang = self.config.language_name
            lang_df = split_df.loc[(split_df['iso_639_3'] == lang.split('_')[0]) & (split_df['iso_15924'] == lang.split('_')[1])]
            for id_, row in enumerate(lang_df.to_dict(orient='record')):
                yield id_, row
                
        elif self.config.schema == f"seacrowd_{str(TASK_TO_SCHEMA[Tasks.MACHINE_TRANSLATION]).lower()}":
            
            lang_1, lang_2 = self.config.first_language_name, self.config.second_language_name
            l1_df = split_df.loc[(split_df['iso_639_3'] == lang_1.split('_')[0]) & (split_df['iso_15924'] == lang_1.split('_')[1])]
            l2_df = split_df.loc[(split_df['iso_639_3'] == lang_2.split('_')[0]) & (split_df['iso_15924'] == lang_2.split('_')[1])]

            mt_df = l1_df.merge(l2_df, on='id')
            mt_df = mt_df.rename({'text_x': 'text_1', 'text_y': 'text_2'}, axis='columns')
            mt_df['text_1_name'] = mt_df.apply(lambda x: f"{x['iso_639_3_x']}_{x['iso_15924_x']}", axis='columns')
            mt_df['text_2_name'] = mt_df.apply(lambda x: f"{x['iso_639_3_y']}_{x['iso_15924_y']}", axis='columns')
            
            for id_, row in enumerate(mt_df[['id', 'text_1', 'text_2', 'text_1_name', 'text_2_name']].to_dict(orient='record')):
                yield id_, row
                
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
