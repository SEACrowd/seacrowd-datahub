from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import SCHEMA_TO_FEATURES, TASK_TO_SCHEMA, Tasks

# TODO: Add BibTeX citation
_CITATION = r"""\
@article{oco2015witchebelles,
  author    = {Oco, Nathaniel and Fajutagana, Raymart and Lim, Christine Mae and Mi{\~n}on, Judi Diane and Morano, Julie-Ann and Tinoco, Ryan Christian},
  title     = {Witchebelles Anata Magcharot kay Mudra na Nagsusuba si Akech: Developing a Rule-based Unidirectional Beki Lingo to Filipino Translator},
  journal   = {Journal of Sciences, Technology and Arts Research},
  volume    = {1},
  number    = {1},
  year      = {2015}
}
"""

_LOCAL = False
_LANGUAGES = ["fil"]
_DATASETNAME = "filipino_gay_lang"
_DESCRIPTION = """\
The dataset contains 4000+ Filipino tweets in gay language or lingo also called swardspeak in slang terminology.
The tweet dataset was collected from February 2013 to November 2014 using the following commonly used gay words as filters: jinet ("hot"), ditey ("here"), imbyerna ("annoying"), etc.
The original paper makes use of the corpus to develop a gay language translator to understand the meaning of phrases using gay words in Filipino.
"""

_HOMEPAGE = "https://github.com/imperialite/Philippine-Languages-Online-Corpora/tree/master/Tweets/Gay%20language"
_LICENSE = "CC-BY-SA 4.0"
_URLS = {
    "gl_01": "https://github.com/imperialite/Philippine-Languages-Online-Corpora/raw/master/Tweets/Gay%20language/gl%20-%2001.xlsx",
    "gl_02": "https://github.com/imperialite/Philippine-Languages-Online-Corpora/raw/master/Tweets/Gay%20language/gl%20-%2002.xlsx",
    "gl_03": "https://github.com/imperialite/Philippine-Languages-Online-Corpora/raw/master/Tweets/Gay%20language/gl%20-%2003.xlsx",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class FilipinoGayLangDataset(datasets.GeneratorBasedBuilder):
    """This dataset contains 4000+ Filipino tweets in gay lingo/Beki/Swardspeak."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "text": datasets.Value("string")})
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = SCHEMA_TO_FEATURES[self.SEACROWD_SCHEMA_NAME.upper()]

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        data_files = {
            "gl_01": Path(dl_manager.download(_URLS["gl_01"])),
            "gl_02": Path(dl_manager.download(_URLS["gl_02"])),
            "gl_03": Path(dl_manager.download(_URLS["gl_03"])),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": [data_files["gl_01"], data_files["gl_02"], data_files["gl_03"]], "split": "train"},
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = pd.concat((pd.read_excel(file) for file in filepath), ignore_index=True).reset_index()

        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {"index": str(row.index), "text": row.message}
                yield row.index, ex
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for row in df.itertuples():
                ex = {"id": str(row.index), "text": row.message}
                yield row.index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
