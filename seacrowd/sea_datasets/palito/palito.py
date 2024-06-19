from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (DEFAULT_SEACROWD_VIEW_NAME,
                                      DEFAULT_SOURCE_VIEW_NAME, Licenses,
                                      Tasks)

_DATASETNAME = "palito"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME

_CITATION = """
@inproceedings{dita-etal-2009-building,
    title = "Building Online Corpora of {P}hilippine Languages",
    author = "Dita, Shirley N.  and
      Roxas, Rachel Edita O.  and
      Inventado, Paul",
    editor = "Kwong, Olivia",
    booktitle = "Proceedings of the 23rd Pacific Asia Conference on Language, Information and Computation, Volume 2",
    month = dec,
    year = "2009",
    address = "Hong Kong",
    publisher = "City University of Hong Kong",
    url = "https://aclanthology.org/Y09-2024",
    pages = "646--653",
}
"""

# We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LANGUAGES = ["bik", "ceb", "hil", "ilo", "tgl", "pam", "pag", "war"]
_LANG_CONFIG = {
    "bik": "Bikol",
    "ceb": "Cebuano",
    "hil": "Hiligaynon",
    "ilo": "Ilocano",
    "tgl": "Tagalog",
    "pam": "Kapampangan",
    "pag": "Pangasinense",
    "war": "Waray",
}

_LOCAL = False

_DESCRIPTION = """\
This paper aims at describing the building of the online corpora on Philippine
languages as part of the online repository system called Palito. There are five components
of the corpora: the top four major Philippine languages which are Tagalog, Cebuano,
Ilocano and Hiligaynon and the Filipino Sign Language (FSL). The four languages are
composed of 250,000-word written texts each, whereas the FSL is composed of seven
thousand signs in video format. Categories of the written texts include creative writing (such
as novels and stories) and religious texts (such as the Bible). Automated tools are provided
for language analysis such as word count, collocates, and others. This is part of a bigger
corpora building project for Philippine languages that would consider text, speech and
video forms, and the corresponding development of automated tools for language analysis
of these various forms.
"""

_HOMEPAGE = "https://github.com/imperialite/Philippine-Languages-Online-Corpora/tree/master/PALITO%20Corpus"

_LICENSE = Licenses.LGPL.value

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_URLS = {
    "literary": "https://raw.githubusercontent.com/imperialite/Philippine-Languages-Online-Corpora/master/PALITO%20Corpus/Data/{lang}_Literary_Text.txt",
    "religious": "https://raw.githubusercontent.com/imperialite/Philippine-Languages-Online-Corpora/master/PALITO%20Corpus/Data/{lang}_Religious_Text.txt",
}


class PalitoDataset(datasets.GeneratorBasedBuilder):
    """Palito corpus"""

    subsets = [f"{_DATASETNAME}_{lang}" for lang in _LANGUAGES]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="{sub}_source".format(sub=subset),
            version=datasets.Version(_SOURCE_VERSION),
            description="Palito {sub} source schema".format(sub=subset),
            schema="source",
            subset_id="{sub}".format(sub=subset),
        )
        for subset in subsets
    ] + [
        SEACrowdConfig(
            name="{sub}_seacrowd_ssp".format(sub=subset),
            version=datasets.Version(_SEACROWD_VERSION),
            description="Palito {sub} SEACrowd schema".format(sub=subset),
            schema="seacrowd_ssp",
            subset_id="{sub}".format(sub=subset),
        )
        for subset in subsets
    ]

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
        else:
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        lang = self.config.name.split("_")[1]
        filepaths = [Path(dl_manager.download(_URLS["literary"].format(lang=_LANG_CONFIG[lang]))), Path(dl_manager.download(_URLS["religious"].format(lang=_LANG_CONFIG[lang])))]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": filepaths},
            ),
        ]

    def _generate_examples(self, filepaths: list[Path]) -> Tuple[int, Dict]:
        counter = 0
        for path in filepaths:
            with open(path, encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == "":
                        continue

                    if self.config.schema == "source":
                        yield (
                            counter,
                            {
                                "id": str(counter),
                                "text": line.strip(),
                            },
                        )
                    elif self.config.schema == "seacrowd_ssp":
                        yield (
                            counter,
                            {
                                "id": str(counter),
                                "text": line.strip(),
                            },
                        )

                    counter += 1
