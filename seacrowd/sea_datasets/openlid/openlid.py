from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{burchell-etal-2023-open,
    title = "An Open Dataset and Model for Language Identification",
    author = "Burchell, Laurie  and
      Birch, Alexandra  and
      Bogoychev, Nikolay  and
      Heafield, Kenneth",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.75",
    doi = "10.18653/v1/2023.acl-short.75",
    pages = "865--879",
    abstract = "Language identification (LID) is a fundamental step in many natural language processing pipelines. However, current LID 
    systems are far from perfect, particularly on lower-resource languages. We present a LID model which achieves a macro-average F1 
    score of 0.93 and a false positive rate of 0.033{\%} across 201 languages, outperforming previous work. We achieve this by training 
    on a curated dataset of monolingual data, which we audit manually to ensure reliability. We make both the model and the dataset 
    available to the research community. Finally, we carry out detailed analysis into our model{'}s performance, both in comparison to 
    existing open models and by language class.",
}
"""

_LOCAL = False
_LANGUAGES = ["ace", "ban", "bjn", "bug", "ceb", "ilo", "ind", "jav", "kac", "khm", "lao", "min", "lus", "mya", "pag", "shn", "sun", "tgl", "tha", "vie", "war", "zsm"]  
_DATASETNAME = "openlid"

_DESCRIPTION = """\
This is an open dataset for language identification covering 201 languages, which are curated and audited manually to 
ensure high confidence in its data and language labels. 22 languages are native to Southeast Asia speakers.
"""

_HOMEPAGE = "https://github.com/laurieburchell/open-lid-dataset"
_LICENSE = Licenses.GPL_3_0.value
_URLS = {
    _DATASETNAME: "https://data.statmt.org/lid/lid201-data.tsv.gz",
}
_SUPPORTED_TASKS = [Tasks.LANGUAGE_IDENTIFICATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

# 201 languages. Each element contains a code for the language, and script (e.g. wol_Latn = Wolof in Latin script)
_TAGS = ['kbp_Latn', 'zul_Latn', 'zho_Hans', 'uig_Arab', 'smo_Latn', 'hrv_Latn', 'tgk_Cyrl', 'guj_Gujr', 'azj_Latn', 'mai_Deva', 'bul_Cyrl', 'hne_Deva', 'wol_Latn', 'ind_Latn', 'lit_Latn', 'epo_Latn', 'prs_Arab', 'kmr_Latn', 'fao_Latn', 'swh_Latn', 'slk_Latn', 'srp_Cyrl', 'bod_Tibt', 'eus_Latn', 'tir_Ethi', 'tam_Taml', 'kas_Deva', 'glg_Latn', 'crh_Latn', 'kon_Latn', 'ayr_Latn', 'por_Latn', 'ben_Beng', 'zho_Hant', 'bug_Latn', 'umb_Latn', 'tzm_Tfng', 'kan_Knda', 'tgl_Latn', 'luo_Latn', 'lij_Latn', 'hun_Latn', 'kin_Latn', 'hat_Latn', 'sag_Latn', 'khm_Khmr', 'heb_Hebr', 'hye_Armn', 'fuv_Latn', 'cjk_Latn', 'ckb_Arab', 'srd_Latn', 'cat_Latn', 'dan_Latn', 'lao_Laoo', 'fra_Latn', 'kam_Latn', 'aeb_Arab', 'ydd_Hebr', 'afr_Latn', 'khk_Cyrl', 'lug_Latn', 'lin_Latn', 'nya_Latn', 'tsn_Latn', 'dzo_Tibt', 'min_Latn', 'war_Latn', 'rus_Cyrl', 'nob_Latn', 'tpi_Latn', 'mlt_Latn', 'mni_Beng', 'ilo_Latn', 'amh_Ethi', 'taq_Latn', 'acq_Arab', 'gaz_Latn', 'ltg_Latn', 'kac_Latn', 'ibo_Latn', 'gle_Latn', 'mya_Mymr', 'grn_Latn', 'kik_Latn', 'jav_Latn', 'awa_Deva', 'ars_Arab', 'swe_Latn', 'uzn_Latn', 'mos_Latn', 'lus_Latn', 'mal_Mlym', 'ita_Latn', 'dik_Latn', 'ewe_Latn', 'sat_Olck', 'pan_Guru', 'est_Latn', 'kab_Latn', 'bam_Latn', 'pag_Latn', 'isl_Latn', 'eng_Latn', 'fon_Latn', 'kas_Arab', 'asm_Beng', 'lim_Latn', 'bjn_Arab', 'taq_Tfng', 'deu_Latn', 'pbt_Arab', 'pap_Latn', 'quy_Latn', 'kea_Latn', 'npi_Deva', 'xho_Latn', 'shn_Mymr', 'nso_Latn', 'urd_Arab', 'bos_Latn', 'ron_Latn', 'fur_Latn', 'gla_Latn', 'nus_Latn', 'ltz_Latn', 'arz_Arab', 'bem_Latn', 'fin_Latn', 'kir_Cyrl', 'tha_Thai', 'mag_Deva', 'azb_Arab', 'tel_Telu', 'ell_Grek', 'sot_Latn', 'spa_Latn', 'vie_Latn', 'yor_Latn', 'ceb_Latn', 'vec_Latn', 'sin_Sinh', 'pol_Latn', 'als_Latn', 'lmo_Latn', 'scn_Latn', 'ces_Latn', 'fij_Latn', 'run_Latn', 'som_Latn', 'mkd_Cyrl', 'mar_Deva', 'ast_Latn', 'san_Deva', 'ary_Arab', 'twi_Latn', 'acm_Arab', 'nno_Latn', 'zsm_Latn', 'mri_Latn', 'kor_Hang', 'sna_Latn', 'pes_Arab', 'ace_Latn', 'bak_Cyrl', 'kat_Geor', 'tur_Latn', 'jpn_Jpan', 'arb_Arab', 'ukr_Cyrl', 'yue_Hant', 'kaz_Cyrl', 'hau_Latn', 'nld_Latn', 'oci_Latn', 'apc_Arab', 'tum_Latn', 'ace_Arab', 'dyu_Latn', 'knc_Latn', 'knc_Arab', 'kmb_Latn', 'bel_Cyrl', 'slv_Latn', 'lvs_Latn', 'bho_Deva', 'tuk_Latn', 'snd_Arab', 'sun_Latn', 'lua_Latn', 'ajp_Arab', 'hin_Deva', 'tso_Latn', 'tat_Cyrl', 'cym_Latn', 'ory_Orya', 'ban_Latn', 'szl_Latn', 'plt_Latn', 'bjn_Latn', 'ssw_Latn']


class OpenLID(datasets.GeneratorBasedBuilder):
    """This is an open dataset for language identification covering 201 languages. 22 languages are native to Southeast Asia speakers."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="openlid_source",
            version=SOURCE_VERSION,
            description="OpenLID source schema",
            schema="source",
            subset_id="openlid",
        ),
        SEACrowdConfig(
            name="openlid_seacrowd_text",
            version=SEACROWD_VERSION,
            description="OpenLID Nusantara schema",
            schema="seacrowd_text",
            subset_id="openlid",
        ),
    ]

    DEFAULT_CONFIG_NAME = "openlid_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "text": datasets.Value("string"), "label": datasets.Value("string"), "source": datasets.Value("string")})
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(_TAGS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # Dataset does not have predetermined split, putting all as TRAIN
        urls = _URLS[_DATASETNAME]
        filepath = Path(dl_manager.download_and_extract(urls))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # Dataset does not have id, using row index as id
        with open(filepath) as f:
            lines = f.readlines()

        if self.config.schema == "source":
            for _id, line in enumerate(lines):
                line = line.split("\t")
                ex = {
                    "id": str(_id),
                    "text": line[0],
                    "label": line[1],
                    "source": line[2].strip(),
                }
                yield _id, ex

        elif self.config.schema == "seacrowd_text":
            for _id, line in enumerate(lines):
                line = line.split("\t")
                ex = {
                    "id": str(_id),
                    "text": line[0],
                    "label": line[1],
                }
                yield _id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
