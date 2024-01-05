from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks
from seacrowd.utils import schemas

_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "bhinneka_korpus"
_DESCRIPTION = """The Bhinneka Korpus dataset was parallel dataset for five Indonesian Local Languages conducted 
through a volunteer-driven translation strategy, encompassing sentences in the Indonesian-English pairs and lexical 
terms. The dataset consist of parallel data with 16,000 sentences in total, details with 4,000 sentence pairs for two 
Indonesia local language and approximately 3,000 sentences for other languages, and one lexicon dataset creation for 
Beaye language. In addition, since beaye is a undocumented language, we don't have any information yet about the use 
of language code. Therefore, we used "day" (a code for land dayak language family) to represent the language."""

_HOMEPAGE = "https://github.com/joanitolopo/bhinneka-korpus"
_LICENSE = Licenses.APACHE_2_0.value
_URLS = "https://raw.githubusercontent.com/joanitolopo/bhinneka-korpus/main/"
_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

_LANGUAGES = ["abs", "day", "mkn", "aoz", "mak"]
LANGUAGES_MAP = {
    "abs": "ambonese-malay",
    "day": "beaye",
    "mkn": "kupang-malay",
    "aoz": "uab-meto",
    "mak": "makassarese"
}


def process_list(data):
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], str):
            return data[0]  # Return the single string if it's a list containing only one string
        else:
            return data  # Return the list as is if it contains multiple strings or other elements
    else:
        return data


class BhinnekaKorpus(datasets.GeneratorBasedBuilder):
    """A Collection of Multilingual Parallel Datasets for 5 Indonesian Local Languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "t2t"

    dataset_names = sorted([f"bhinneka_korpus_{lang}" for lang in _LANGUAGES])
    BUILDER_CONFIGS = []
    for name in dataset_names:
        source_config = SEACrowdConfig(
            name=f"{name}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=name
        )
        BUILDER_CONFIGS.append(source_config)
        seacrowd_config = SEACrowdConfig(
            name=f"{name}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=name
        )
        BUILDER_CONFIGS.append(seacrowd_config)

    BUILDER_CONFIGS.extend(
        [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} source schema (all)",
                schema="source",
                subset_id=_DATASETNAME
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} SEACrowd schema (all)",
                schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
                subset_id=_DATASETNAME
            )
        ]
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        schema = self.config.schema
        features = datasets.Features(
            {
                "source_sentence": datasets.Value("string"),
                "target_sentence": datasets.Value("string"),
                "source_lang": datasets.Value("string"),
                "target_lang": datasets.Value("string")
            } if schema == "source" else schemas.text2text_features
            if schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}" else None
        )
        if features is None:
            raise ValueError("Invalid config schema")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        languages = []
        bhinneka_source_data = []

        lang = self.config.name.split("_")[2]
        if lang in _LANGUAGES:
            bhinneka_source_data.append(Path(dl_manager.download(_URLS + f"{LANGUAGES_MAP[lang]}/{lang}.xlsx")))
            languages.append(lang)
        else:
            for i in _LANGUAGES:
                bhinneka_source_data.append(Path(dl_manager.download(_URLS + f"{LANGUAGES_MAP[i]}/{i}.xlsx")))
                languages.append(i)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": bhinneka_source_data,
                    "split": "train",
                    "languages": languages
                }
            )
        ]

    def _generate_examples(self, filepath: Path, split: str, languages: List[str]) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        dfs = pd.read_excel(filepath[0], index_col=0, engine="openpyxl")
        if len(filepath) > 2:
            for path2file in filepath[1:]:
                dfs = pd.merge(dfs, pd.read_excel(path2file, index_col=0, engine="openpyxl"))

        source_sents = dfs.drop(columns=["ind"])
        target_sents = dfs["ind"]

        for idx, (source, target) in enumerate(zip(source_sents.values, target_sents.values)):
            if self.config.schema == "source":
                example = {
                    "source_sentence": process_list(list(source)),
                    "target_sentence": process_list(target),
                    "source_lang": process_list(languages),
                    "target_lang": "ind"
                }
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                example = {
                    "id": str(idx),
                    "text_1": process_list(list(source)),
                    "text_2": process_list(target),
                    "text_1_name": process_list(languages),
                    "text_2_name": "ind",
                }
            yield idx, example
