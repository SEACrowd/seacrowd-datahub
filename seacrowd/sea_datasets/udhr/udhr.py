import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from pathlib import Path
from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@misc{united1998universal,
    title={The Universal Declaration of Human Rights, 1948-1998},
    author={United Nations},
    year={1998},
    publisher={United Nations Dept. of Public Information New York}
}
"""
_DATASETNAME = "udhr"
_DESCRIPTION = """\
The Universal Declaration of Human Rights (UDHR) is a milestone document in the history of
human rights. Drafted by representatives with different legal and cultural backgrounds from
all regions of the world, it set out, for the first time, fundamental human rights to be
universally protected. The Declaration was adopted by the UN General Assembly in Paris on
10 December 1948 during its 183rd plenary meeting.
"""
_HOMEPAGE = "https://unicode.org/udhr/index.html"
_LICENSE = Licenses.UNKNOWN.value
_URLS = "https://unicode.org/udhr/assemblies/udhr_txt.zip"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_LANGS = {
    "ace": "Aceh",
    "ban": "Bali",
    "bcl": "Bicolano, Central",
    "blt": "Tai Dam", 
    "bug": "Bugis",
    "ceb": "Cebuano",
    "cfm": "Chin, Falam",  # flm
    "cnh": "Chin, Haka",
    "ctd": "Chin, Tedim",
    "duu": "Drung", 
    "hil": "Hiligaynon", 
    "hlt": "Chin, Matu",
    "hni": "Hani",
    "hnj": "Hmong Njua",  # blu
    "ilo": "Ilocano", 
    "ind": "Indonesian",
    "jav": "Javanese",
    "jav_java": "Javanese (Javanese)",
    "khm": "Khmer",
    "kkh": "Khun", 
    "lao": "Lao",
    "lus": "Mizo",
    "mad": "Madura",
    "min": "Minangkabau", 
    "mnw": "Mon", 
    "mya": "Burmese",
    "pam": "Pampangan", 
    "shn": "Shan",
    "sun": "Sunda",
    "tdt": "Tetun Dili", 
    "tet": "Tetun", 
    "tgl": "Tagalog",
    "tha": "Thai",
    "vie": "Vietnamese",
    "war": "Waray-waray", 
    "zlm": "Malay",  # default mly_latn
}

_LOCAL=False
_LANGUAGES=["ace", "ban", "bcl", "blt", "bug", "ceb", "cfm", "cnh", "ctd", "duu", "hil", "hlt", "hni", "hnj", "ilo", "ind", "jav", "khm", "kkh", "lao", "lus", "mad", "min", "mnw", "mya", "pam", "shn", "sun", "tdt", "tet", "tgl", "tha", "vie", "war", "zlm"]

def seacrowd_config_constructor(src_lang, schema, version):
    if src_lang == "":
        raise ValueError(f"Invalid src_lang {src_lang}")

    if schema not in ["source", "seacrowd_ssp"]:
        raise ValueError(f"Invalid schema: {schema}")

    return SEACrowdConfig(
        name="udhr_{src}_{schema}".format(src=src_lang, schema=schema),
        version=datasets.Version(version),
        description="udhr {schema} schema for {src} language".format(schema=schema, src=_LANGS[src_lang]),
        schema=schema,
        subset_id="udhr_{src}".format(src=src_lang),
    )


class UDHRDataset(datasets.GeneratorBasedBuilder):
    """
    The Universal Declaration of Human Rights (UDHR) is a milestone document in the history of
    human rights. Drafted by representatives with different legal and cultural backgrounds from
    all regions of the world, it set out, for the first time, fundamental human rights to be
    universally protected. The Declaration was adopted by the UN General Assembly in Paris on
    10 December 1948 during its 183rd plenary meeting.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [seacrowd_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGS] + [seacrowd_config_constructor(lang, "seacrowd_ssp", _SEACROWD_VERSION) for lang in _LANGS]

    DEFAULT_CONFIG_NAME = "udhr_ind_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_ssp":
            features = schemas.self_supervised_pretraining.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS
        data_dir = dl_manager.download_and_extract(urls)
        lang = self.config.subset_id.split("_")
        file_key=""
        if lang[1] == "zlm":
            file_key = "mly_latn"
        elif lang[1] == "cfm":
            file_key = "flm"
        elif lang[1] == "hnj":
            file_key = "blu"        
        elif lang[1] == "kkh":
            file_key = "kkh_lana"       
        elif lang[1] == "duu":
            file_key = "020"
        elif lang[1] == "tdt":
            file_key = "010"
        elif len(lang)>2 and f"{lang[1]}_{lang[2]}" == "jav_java":
            file_key = "jav_java"
        else:
            file_key = lang[1]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"udhr_{file_key}.txt".format(file_key=file_key)),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = []

        with open(filepath, "r") as f:
            data = [line.rstrip() for line in f.readlines()]

        if self.config.schema == "source":
            yield 0, {"id": Path(filepath).stem, "text": " ".join(data)}

        elif self.config.schema == "seacrowd_ssp":
            yield 0, {"id": Path(filepath).stem, "text": " ".join(data)}
