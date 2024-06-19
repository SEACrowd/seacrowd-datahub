import gzip
import json
from typing import List

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks


_DATASETNAME = "mc4_indo"
_DESCRIPTION = """\
    A thoroughly cleaned version of the Indonesia split of the multilingual colossal, cleaned version of Common Crawl's web crawl corpus (mC4). This portion represents the Indonesian language content that has been extracted and processed from the larger mC4 dataset. The extraction and cleaning process was conducted by AllenAI and resulted in a curated collection of Indonesian language data. For more information about the original mC4 dataset and its preparation, please refer to the source hosted at the address https://huggingface.co/datasets/allenai/c4.
"""

_HOMEPAGE = "https://huggingface.co/datasets/indonesian-nlp/mc4-id"
_LICENSE = Licenses.ODC_BY.value

_LANGUAGES = ["ind"]

_CITATION = """
@inproceedings{xue-etal-2021-mt5,
    title = "m{T}5: A Massively Multilingual Pre-trained Text-to-Text Transformer",
    author = "Xue, Linting  and
      Constant, Noah  and
      Roberts, Adam  and
      Kale, Mihir  and
      Al-Rfou, Rami  and
      Siddhant, Aditya  and
      Barua, Aditya  and
      Raffel, Colin",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.41",
    doi = "10.18653/v1/2021.naacl-main.41",
    pages = "483--498",
}
"""

_URLS = {"raw": "https://huggingface.co/datasets/munggok/mc4-id/resolve/main/mc4-id-filter/c4-id{split_suffix}.tfrecord-{index:05d}-of-{n_shards:05d}.json.gz"}

_CONFIGS = {"full": {"train": 1016, "validation": 8}}
# The entire dataset is 150 Gigs. You can adjust the number of "parquet" files you want to download here
# _CONFIGS = {
#     "full": {"train": 1, "validation": 1}
# }

_LOCAL = False

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class MC4Indo(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="mc4_indo source schema",
            schema="source",
            subset_id="mc4_indo",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_ssp",
            version=SEACROWD_VERSION,
            description="mc4_indo SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id="mc4_indo",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"text": datasets.Value("string"), "timestamp": datasets.Value("string"), "url": datasets.Value("string")})

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
        data_urls = {}
        for split in ["train", "validation"]:
            data_urls[split] = [
                _URLS["raw"].format(
                    split_suffix="-validation" if split == "validation" else "",
                    index=index,
                    n_shards=8 if split == "validation" else 1024,
                )
                for index in range(_CONFIGS["full"][split])
            ]
        train_downloaded_files = dl_manager.download(data_urls["train"])
        validation_downloaded_files = dl_manager.download(data_urls["validation"])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": train_downloaded_files,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": validation_downloaded_files,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepaths: [Path], split: str) -> Tuple[int, Dict]:
        id_ = 0
        for filepath in filepaths:
            with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for line in f:
                    if line:
                        example = json.loads(line)

                        if self.config.schema == "source":
                            yield id_, example
                        elif self.config.schema == "seacrowd_ssp":
                            seacrowd_json = {
                                "id": str(id_),
                                "text": str(example["text"]),
                            }
                            yield id_, seacrowd_json

                        id_ += 1
