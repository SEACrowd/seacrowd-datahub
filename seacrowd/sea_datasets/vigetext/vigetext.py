from pathlib import Path

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{10.1145/3628797.3628837,
    author = {Nguyen, Duc-Vu and Nguyen, Quoc-Nam},
    title = {Evaluating the Symbol Binding Ability of Large Language Models for Multiple-Choice Questions in Vietnamese General Education},
    year = {2023},
    isbn = {9798400708916},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3628797.3628837},
    doi = {10.1145/3628797.3628837},
    booktitle = {Proceedings of the 12th International Symposium on Information and Communication Technology},
    pages = {379–386},
    numpages = {8},
    keywords = {Analysis of Language Models, Multiple Choice Symbol Binding, Multiple Choice Question Answering, Language Modeling},
    location = {<conf-loc>, <city>Ho Chi Minh</city>, <country>Vietnam</country>, </conf-loc>},
    series = {SOICT '23}
}
"""

_DATASETNAME = "vigetext"

_DESCRIPTION = """
The high-quality dataset with structured guidelines for typing LaTeX formulas in Mathematics, Physics, Chemistry, and
Biology. Objective was to cover the entire scope of the Vietnamese General Education Examination spanning from 2017 to 2023.
This comprehensive approach included the challenging examinations of the years 2017 and 2018, which have been significant
for nearly all Vietnamese students in recent years. It is important to highlight that the exact and unquestionably correct
answers have been exclusively obtained from the Vietnamese Ministry of Education.
"""

_HOMEPAGE = "https://huggingface.co/datasets/uitnlp/ViGEText_17to23"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "train": "https://huggingface.co/datasets/uitnlp/ViGEText_17to23/resolve/main/data/train-00000-of-00001.parquet",
        "validation": "https://huggingface.co/datasets/uitnlp/ViGEText_17to23/resolve/main/data/validation-00000-of-00001.parquet",
        "test": "https://huggingface.co/datasets/uitnlp/ViGEText_17to23/resolve/main/data/test-00000-of-00001.parquet",
    }
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class VigetextDataset(datasets.GeneratorBasedBuilder):
    """Vigetext is a dataset for evaluating the Symbol Binding Ability of Large Language Models for Multiple-Choice Questions in Vietnamese General Education."""

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
            name=f"{_DATASETNAME}_seacrowd_qa",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_qa",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            )

        else:
            features = schemas.qa_features

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
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_dir, "split": "validation"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir, "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> tuple[int, dict]:
        df = pd.read_parquet(filepath[split])
        data = df.to_dict(orient="records")
        for i, item in enumerate(data):
            if self.config.schema == "source":
                yield i, {
                    "id": item["id"],
                    "input": item["input"],
                    "target": item["target"],
                }
            else:
                question_and_options = item["input"].split("\n")
                answer_map = {opt[0]: opt[2:].strip() for opt in question_and_options[1:]}
                yield i, {
                    "id": str(i),
                    "question_id": item["id"],
                    "document_id": "",
                    "question": question_and_options[0],
                    "type": "multiple_choice",
                    "choices": [opt[2:].strip() for opt in question_and_options[1:]],  # remove A., B., ... in the options
                    "context": "",
                    "answer": [answer_map[item["target"]]],
                    "meta": {}
                }
