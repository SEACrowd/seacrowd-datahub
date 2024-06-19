import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{nguyen-etal-2018-introducing,
    title = "Introducing Two {V}ietnamese Datasets for Evaluating Semantic Models of (Dis-)Similarity and Relatedness",
    author = "Nguyen, Kim Anh  and
      Schulte im Walde, Sabine  and
      Vu, Ngoc Thang",
    editor = "Walker, Marilyn  and
      Ji, Heng  and
      Stent, Amanda",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-2032",
    doi = "10.18653/v1/N18-2032",
    pages = "199--205"
    }
"""

_DATASETNAME = "visim400"

_DESCRIPTION = """\
ViSim-400 is a Vietnamese dataset of semantic relation \
pairs for evaluation of models that reflect the \
continuum between similarity and relatedness.

We choose 'Sim2' instead of 'Sim1' for the label output of \
our SEACrowd dataloader schema because it's been normalized to [1, 10].
"""

_HOMEPAGE = "https://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/vnese-sem-datasets/"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_LOCAL = False

_URLS = {_DATASETNAME: "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/ViData.zip"}

_SUPPORTED_TASKS = [Tasks.SEMANTIC_SIMILARITY]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class ViSim400Dataset(datasets.GeneratorBasedBuilder):
    """
    ViSim-400 is a Vietnamese dataset of semantic relation \
    pairs for evaluation of models that reflect the \
    continuum between similarity and relatedness.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "pairs_score"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=_SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=_SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "Word1": datasets.Value("string"),
                    "Word2": datasets.Value("string"),
                    "POS": datasets.Value("string"),
                    "Sim1": datasets.Value("string"),
                    "Sim2": datasets.Value("string"),
                    "STD": datasets.Value("string"),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.pairs_features_score()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        data_dir = dl_manager.download_and_extract(_URLS[_DATASETNAME])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "ViData/ViSim-400/Visim-400.txt"),
                    "split": "test",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()

        data = []
        for line in lines:
            columns = line.strip().split("\t")
            data.append(columns)

        df = pd.DataFrame(data[1:], columns=data[0])

        for index, row in df.iterrows():

            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":

                example = {
                    "id": str(index),
                    "text_1": str(row["Word1"]),
                    "text_2": str(row["Word2"]),
                    # I choose Sim2 instead of Sim1 because it's been normalized to [1, 10]
                    "label": str(row["Sim2"]),
                }

            yield index, example
