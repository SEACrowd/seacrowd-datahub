# coding=utf-8
# Copyright 2024 The HuggingFace Datasets Authors and the current dataset script contributor.
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
SIB-200 is the largest publicly available topic classification dataset based on Flores-200 covering 205 languages and dialects.
The train/validation/test sets are available for all the 205 languages.
"""

from pathlib import Path
from typing import List

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@misc{adelani2023sib200,
      title={SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects},
      author={David Ifeoluwa Adelani and Hannah Liu and Xiaoyu Shen and Nikita Vassilyev and Jesujoba O. Alabi and Yanke Mao and Haonan Gao and Annie En-Shiun Lee},
      year={2023},
      eprint={2309.07445},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "sib_200"

_DESCRIPTION = """\
SIB-200 is the largest publicly available topic classification dataset based on Flores-200 covering 205 languages and dialects.
The train/validation/test sets are available for all the 205 languages.
"""

_HOMEPAGE = "https://github.com/dadelani/sib-200"

_LANGUAGES = ["ace", "ban", "bjn", "bug", "ceb", "ilo", "ind", "jav", "kac", "khm", "lao", "lus", "min", "mya", "pag", "shn", "sun", "tgl", "tha", "vie", "war", "zsm"]

_PAIRS = [
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
    "gaz_Latn",
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
    "kaz_Cyrl",
    "kbp_Latn",
    "kea_Latn",
    "khk_Cyrl",
    "khm_Khmr",
    "kik_Latn",
    "kin_Latn",
    "kir_Cyrl",
    "kmb_Latn",
    "kmr_Latn",
    "knc_Arab",
    "knc_Latn",
    "kon_Latn",
    "kor_Hang",
    "lao_Laoo",
    "lij_Latn",
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
    "mlt_Latn",
    "mni_Beng",
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
    "ory_Orya",
    "pag_Latn",
    "pan_Guru",
    "pap_Latn",
    "pbt_Arab",
    "pes_Arab",
    "plt_Latn",
    "pol_Latn",
    "por_Latn",
    "prs_Arab",
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
    "taq_Latn",
    "taq_Tfng",
    "tat_Cyrl",
    "tel_Telu",
    "tgk_Cyrl",
    "tgl_Latn",
    "tha_Thai",
    "tir_Ethi",
    "tpi_Latn",
    "tsn_Latn",
    "tso_Latn",
    "tuk_Latn",
    "tum_Latn",
    "tur_Latn",
    "twi_Latn",
    "tzm_Tfng",
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
    "zho_Hans",
    "zho_Hant",
    "zsm_Latn",
    "zul_Latn",
]

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "template": "https://huggingface.co/datasets/Davlan/sib200/resolve/main/data/{lang}_{ws}/{split}.tsv",
}

_SUPPORTED_TASKS = [Tasks.DOMAIN_KNOWLEDGE_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

_SEACROWD_SCHEMA = f"seacrowd_dkc"


def _sib_config_constructor(lang: str, ws: str = "*", schema: str = _SEACROWD_SCHEMA, version: str = _SEACROWD_VERSION) -> SEACrowdConfig:
    return SEACrowdConfig(
        name=f"{_DATASETNAME}_{lang}_{schema}",
        version=version,
        description=f"SIB-200 {schema} schema",
        schema=schema,
        subset_id=f"SIB-200 {lang} {ws}",
    )


class Sib200Dataset(datasets.GeneratorBasedBuilder):
    """
    SIB-200 is the largest publicly available topic classification dataset based on Flores-200 covering 205 languages and dialects.
    The train/validation/test sets are available for all the 205 languages.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    def _populate_configs():
        configs = [_sib_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGUAGES] + [_sib_config_constructor(lang, _SEACROWD_SCHEMA, _SEACROWD_VERSION) for lang in _LANGUAGES]

        all_lang_source_config = SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=_SOURCE_VERSION,
            description=f"SIB-200 source schema",
            schema="source",
            subset_id=f"SIB-200 SEA",
        )

        all_lang_t2t_config = SEACrowdConfig(
            name=f"{_DATASETNAME}_{_SEACROWD_SCHEMA}",
            version=_SEACROWD_VERSION,
            description=f"SIB-200 {_SEACROWD_SCHEMA} schema",
            schema=_SEACROWD_SCHEMA,
            subset_id=f"SIB-200 SEA",
        )

        configs.append(all_lang_source_config)
        configs.append(all_lang_t2t_config)
        return configs

    BUILDER_CONFIGS = _populate_configs()

    DEFAULT_CONFIG_NAME = "sib_200_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "inputs": datasets.Value("string"),
                    "targets": datasets.Value("string"),
                    "language": datasets.Value("string"),
                    "language_code": datasets.Value("string"),
                    "annotation_type": datasets.Value("string"),
                    "user_id": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def get_lang_filter(self, config_name: str):
        # sib_200_{lang}_{ws}_{schema}
        tokens = config_name.split("_")
        if len(tokens) == 0 or len(tokens[2]) != 3:
            return None
        return tokens[2]

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        url = _URLS["train"]
        data_dir = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_path": Path(data_dir),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, data_path: Path, split: str):
        """Yields examples as (key, example) tuples."""

        df = pd.read_parquet(data_path)

        lang_filter = self.get_lang_filter(self.config.name)
        if lang_filter is not None:
            df = df[df["language_code"] == lang_filter]
        else:
            df = df[df["language_code"].isin(_LANGUAGES)]

        if self.config.schema == "source":
            for idx, row in df.iterrows():
                data = row.to_dict()
                yield idx, data

        elif self.config.schema == "seacrowd_t2t":
            for idx, row in df.iterrows():
                sample = {
                    "id": str(idx),
                    "text_1": row["inputs"],
                    "text_2": row["targets"],
                    "text_1_name": "inputs",
                    "text_2_name": "targets",
                }
                yield idx, sample
