from pathlib import Path

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@article{thoma2018wili,
  title={The WiLI benchmark dataset for written language identification},
  author={Thoma, Martin},
  journal={arXiv preprint arXiv:1801.07779},
  year={2018}
}
"""

_DATASETNAME = "wili_2018"

_DESCRIPTION = """
WiLI-2018 is a Wikipedia language identification benchmark dataset. It contains 235000 paragraphs from 235 languages.
The dataset is balanced, and a train-test split is provided.
"""

_HOMEPAGE = "https://zenodo.org/records/841984"

_LANGUAGES = ["nrm", "jav", "min", "lao", "mya", "pag", "ind", "cbk", "tet", "tha", "ceb", "tgl", "bjn", "bcl", "vie"]

_LICENSE = Licenses.ODBL.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {"train": "https://drive.google.com/uc?export=download&id=1ZzlIQvw1KNBG97QQCfdatvVrrbeLaM1u", "test": "https://drive.google.com/uc?export=download&id=1Xx4kFc1Xdzz8AhDasxZ0cSa-a35EQSDZ"},
}

_SUPPORTED_TASKS = [Tasks.LANGUAGE_IDENTIFICATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


_CLASSES = [
    "cdo",
    "glk",
    "jam",
    "lug",
    "san",
    "rue",
    "wol",
    "new",
    "mwl",
    "bre",
    "ara",
    "hye",
    "xmf",
    "ext",
    "cor",
    "yor",
    "div",
    "asm",
    "lat",
    "cym",
    "hif",
    "ace",
    "kbd",
    "tgk",
    "rus",
    "nso",
    "mya",
    "msa",
    "ava",
    "cbk",
    "urd",
    "deu",
    "swa",
    "pus",
    "bxr",
    "udm",
    "csb",
    "yid",
    "vro",
    "por",
    "pdc",
    "eng",
    "tha",
    "hat",
    "lmo",
    "pag",
    "jav",
    "chv",
    "nan",
    "sco",
    "kat",
    "bho",
    "bos",
    "kok",
    "oss",
    "mri",
    "fry",
    "cat",
    "azb",
    "kin",
    "hin",
    "sna",
    "dan",
    "egl",
    "mkd",
    "ron",
    "bul",
    "hrv",
    "som",
    "pam",
    "nav",
    "ksh",
    "nci",
    "khm",
    "sgs",
    "srn",
    "bar",
    "cos",
    "ckb",
    "pfl",
    "arz",
    "roa-tara",
    "fra",
    "mai",
    "zh-yue",
    "guj",
    "fin",
    "kir",
    "vol",
    "hau",
    "afr",
    "uig",
    "lao",
    "swe",
    "slv",
    "kor",
    "szl",
    "srp",
    "dty",
    "nrm",
    "dsb",
    "ind",
    "wln",
    "pnb",
    "ukr",
    "bpy",
    "vie",
    "tur",
    "aym",
    "lit",
    "zea",
    "pol",
    "est",
    "scn",
    "vls",
    "stq",
    "gag",
    "grn",
    "kaz",
    "ben",
    "pcd",
    "bjn",
    "krc",
    "amh",
    "diq",
    "ltz",
    "ita",
    "kab",
    "bel",
    "ang",
    "mhr",
    "che",
    "koi",
    "glv",
    "ido",
    "fao",
    "bak",
    "isl",
    "bcl",
    "tet",
    "jpn",
    "kur",
    "map-bms",
    "tyv",
    "olo",
    "arg",
    "ori",
    "lim",
    "tel",
    "lin",
    "roh",
    "sqi",
    "xho",
    "mlg",
    "fas",
    "hbs",
    "tam",
    "aze",
    "lad",
    "nob",
    "sin",
    "gla",
    "nap",
    "snd",
    "ast",
    "mal",
    "mdf",
    "tsn",
    "nds",
    "tgl",
    "nno",
    "sun",
    "lzh",
    "jbo",
    "crh",
    "pap",
    "oci",
    "hak",
    "uzb",
    "zho",
    "hsb",
    "sme",
    "mlt",
    "vep",
    "lez",
    "nld",
    "nds-nl",
    "mrj",
    "spa",
    "ceb",
    "ina",
    "heb",
    "hun",
    "que",
    "kaa",
    "mar",
    "vec",
    "frp",
    "ell",
    "sah",
    "eus",
    "ces",
    "slk",
    "chr",
    "lij",
    "nep",
    "srd",
    "ilo",
    "be-tarask",
    "bod",
    "orm",
    "war",
    "glg",
    "mon",
    "gle",
    "min",
    "ibo",
    "ile",
    "epo",
    "lav",
    "lrc",
    "als",
    "mzn",
    "rup",
    "fur",
    "tat",
    "myv",
    "pan",
    "ton",
    "kom",
    "wuu",
    "tcy",
    "tuk",
    "kan",
    "ltg",
]


class Wili2018Dataset(datasets.GeneratorBasedBuilder):
    """A benchmark dataset for language identification and contains 235000 paragraphs of 235 languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sentence": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=_CLASSES),
                }
            )

        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(_CLASSES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_dir, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir, "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> tuple[int, dict]:
        if self.config.schema == "source":
            with open(filepath[split], encoding="utf-8") as f:
                for i, line in enumerate(f):
                    text, label = line.rsplit(",", 1)
                    text = text.strip('"')
                    label = int(label.strip())
                    yield i, {"sentence": text, "label": _CLASSES[label - 1]}

        elif self.config.schema == "seacrowd_text":
            with open(filepath[split], encoding="utf-8") as f:
                for i, line in enumerate(f):
                    text, label = line.rsplit(",", 1)
                    text = text.strip('"')
                    label = int(label.strip())
                    yield i, {"id": str(i), "text": text, "label": _CLASSES[label - 1]}
