# coding=utf-8
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_DATASETNAME = "melayu_sarawak"


_DESCRIPTION = """\
Korpus Variasi Bahasa Melayu: Sarawak is a language corpus sourced from various folklores in Melayu Sarawak dialect.
"""


_HOMEPAGE = "https://github.com/matbahasa/Melayu_Sarawak"

_CITATION = """\
@misc{melayusarawak,
  author = {Hiroki Nomoto},
  title = {Melayu_Sabah},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/matbahasa/Melayu_Sarawak}},
  commit = {a175f691f9db94d7b4f971e7a93b7cc001c0ed47}
}
"""

_LANGUAGES = ["zlm"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    "sarawak201801": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201801.txt",
    "sarawak201802": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201802.txt",
    "sarawak201803": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201803.txt",
    "sarawak201804": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201804.txt",
    "sarawak201805": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201805.txt",
    "sarawak201806": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201806.txt",
    "sarawak201807": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201807.txt",
    "sarawak201808": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201808.txt",
    "sarawak201809": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201809.txt",
    "sarawak201810": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201810.txt",
    "sarawak201811": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201811.txt",
    "sarawak201812": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201812.txt",
    "sarawak201813": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201813.txt",
    "sarawak201814": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201814.txt",
    "sarawak201815": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201815.txt",
    "sarawak201817": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201817.txt",
    "sarawak201818": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201818.txt",
    "sarawak201819": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201819.txt",
    "sarawak201820": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201820.txt",
    "sarawak201821": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201821.txt",
    "sarawak201822": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201822.txt",
    "sarawak201823": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201823.txt",
    "sarawak201824": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201824.txt",
    "sarawak201825": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201825.txt",
    "sarawak201826": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201826.txt",
    "sarawak201827": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201827.txt",
    "sarawak201828": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201828.txt",
    "sarawak201829": "https://raw.githubusercontent.com/matbahasa/Melayu_Sarawak/master/Sarawak201829.txt",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MelayuSarawakDataset(datasets.GeneratorBasedBuilder):
    """Korpus Variasi Bahasa Melayu:
    Sarawak is a language corpus sourced from various folklores in Melayu Sarawak dialect."""

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

        urls = [_URLS[key] for key in _URLS.keys()]
        data_path = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_path[0], "split": "train", "other_path": data_path[1:]},
            )
        ]

    def _generate_examples(self, filepath: Path, split: str, other_path: List) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        filepaths = [filepath] + other_path
        data = []
        for filepath in filepaths:
            with open(filepath, "r") as f:
                data.append([line.rstrip() for line in f.readlines()])

        for id, text in enumerate(data):
            yield id, {"id": id, "text": text}