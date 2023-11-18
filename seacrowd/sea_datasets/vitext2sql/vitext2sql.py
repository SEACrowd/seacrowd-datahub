from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{nguyen2020vitext2sql,
    title      = {{A Pilot Study of Text-to-SQL Semantic Parsing for Vietnamese}},
    author     = {Anh Tuan Nguyen and Mai Hoang Dao and Dat Quoc Nguyen},
    booktitle  = {Findings of the Association for Computational Linguistics: EMNLP 2020},
    year       = {2020},
    pages      = {4079--4085}
}
"""

_DATASETNAME = "vitext2sql"

_DESCRIPTION = """\
This is the first public large-scale Text-to-SQL semantic parsing dataset for Vietnamese.
The dataset is created by manually translating the Spider dataset into Vietnamese.
"""

_HOMEPAGE = "https://github.com/VinAIResearch/ViText2SQL"

_LICENSE = Licenses.UNKNOWN.value

_SOURCE_VERSION = "1.0.0"

_URLS = {
    _DATASETNAME: {
        "word-level": {
            "train": "https://raw.githubusercontent.com/VinAIResearch/ViText2SQL/master/data/word-level/train.json",
            "test": "https://raw.githubusercontent.com/VinAIResearch/ViText2SQL/master/data/word-level/test.json",
            "validation": "https://raw.githubusercontent.com/VinAIResearch/ViText2SQL/master/data/word-level/dev.json",
        },
        "syllable-level": {
            "train": "https://raw.githubusercontent.com/VinAIResearch/ViText2SQL/master/data/syllable-level/train.json",
            "test": "https://raw.githubusercontent.com/VinAIResearch/ViText2SQL/master/data/syllable-level/test.json",
            "validation": "https://raw.githubusercontent.com/VinAIResearch/ViText2SQL/master/data/syllable-level/dev.json",
        },
    }
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SEACROWD_VERSION = "1.0.0"


class Vitext2sqlDataset(datasets.GeneratorBasedBuilder):
    """Vitext2sql dataset is a Text-to-SQL semantic parsing dataset for Vietnamese."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="vitext2sql_source",
            version=SOURCE_VERSION,
            description="Vitext2sql word level source schema",
            schema="source",
            subset_id="vitext2sql",
        ),
        SEACrowdConfig(
            name="vitext2sql_source_syllable",
            version=SOURCE_VERSION,
            description="Vitext2sql syllable level source schema",
            schema="source",
            subset_id="vitext2sql",
        ),
        SEACrowdConfig(
            name="vitext2sql_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description="Vitext2sql SEACrowd schema for word-level",
            schema="seacrowd_t2t",
            subset_id="vitext2sql",
        ),
        SEACrowdConfig(
            name="vitext2sql_seacrowd_syllable_t2t",
            version=SEACROWD_VERSION,
            description="Vitext2sql SEACrowd schema for syllable-level",
            schema="seacrowd_t2t",
            subset_id="vitext2sql",
        ),
    ]

    DEFAULT_CONFIG_NAME = "vitext2sql_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            # The sql column is an unstructured JSON,
            # in the meantime just treat it as large string.
            features = datasets.Features(
                {
                    "db_id": datasets.Value("string"),
                    "query": datasets.Value("string"),
                    "query_toks": [datasets.Value("string")],
                    "query_toks_no_value": [datasets.Value("string")],
                    "question": datasets.Value("string"),
                    "question_toks": [datasets.Value("string")],
                    "sql": datasets.Value("large_string"),
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

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]

        if "syllable" in self.config.name:
            level_urls = urls["syllable-level"]
        else:
            level_urls = urls["word-level"]

        data_files = {
            "train": dl_manager.download_and_extract(level_urls["train"]),
            "test": dl_manager.download_and_extract(level_urls["test"]),
            "validation": dl_manager.download_and_extract(level_urls["validation"]),
        }
        split_generators = [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_files["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_files["validation"],
                },
            ),
        ]

        return split_generators

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        df = pd.read_json(filepath)
        if self.config.schema == "source":
            for i, row in df.iterrows():
                entry = {"db_id": row["db_id"], "query": row["query"], "query_toks": row["query_toks"], "query_toks_no_value": row["query_toks_no_value"], "question": row["question"], "question_toks": row["question_toks"], "sql": str(row["sql"])}
                yield i, entry
        elif self.config.schema == "seacrowd_t2t":
            for i, row in df.iterrows():
                entry = {
                    "id": str(i),
                    "text_1": row["question"],
                    "text_2": row["query"],
                    "text_1_name": "question",
                    "text_2_name": "query",
                }
                yield i, entry
