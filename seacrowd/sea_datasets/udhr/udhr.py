import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from pathlib import Path
from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@online{UDHR,
    title = {UDHR in Unicode},
    howpublished = {\\url{https://unicode.org/udhr/index.html}},
    note = {Accessed: 2023-11-20}
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

_SEACROWD_VERSION = "1.0.0"

_LANGS = {
    "khm": "Khmer",
    "tha": "Thai",
    "vie": "Vietnamese",
    "ind": "Indonesian",
    "zlm": "Malay",  # default mly_latn
    "lao": "Lao",
    "ban": "Bali",
    "mya": "Burmese",
    "ceb": "Cebuano",
    "ace": "Aceh",
    "tgl": "Tagalog",
    # "abs": "",
    "bug": "Bugis",
    # "cja": "",
    "cfm": "Chin, Falam",  # flm
    "cnh": "Chin, Haka",
    "ctd": "Chin, Tedim",
    # "fil": "",
    "hnj": "Hmong Njua",  # blu
    # "iba": "",
    # "dbj": "",
    "jav": "Javanese",
}  # ada jav_java, default: jav (latin)


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
        lang = self.config.subset_id.split("_")[1]
        if lang == "zlm":
            lang = "mly_latn"

        if lang == "cfm":
            lang = "flm"

        if lang == "hnj":
            lang = "blu"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"udhr_{lang}.txt".format(lang=lang)),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = []
        lang = self.config.subset_id.split("_")[1]

        with open(filepath, "r") as f:
            data = [line.rstrip() for line in f.readlines()]

        if self.config.schema == "source":
            yield 0, {"id": Path(filepath).stem, "text": " ".join(data)}

        elif self.config.schema == "seacrowd_ssp":
            yield 0, {"id": Path(filepath).stem, "text": " ".join(data)}
