from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{DBLP:conf/nips/LaurenconSWAMSW22,
  author={Hugo Laurençon and Lucile Saulnier and Thomas Wang and Christopher Akiki and Albert Villanova del Moral and
  Teven Le Scao and Leandro von Werra and Chenghao Mou and Eduardo González Ponferrada and Huu Nguyen and Jörg Frohberg
  and Mario Sasko and Quentin Lhoest and Angelina McMillan-Major and Gérard Dupont and Stella Biderman and Anna Rogers
  and Loubna Ben Allal and Francesco De Toni and Giada Pistilli and Olivier Nguyen and Somaieh Nikpoor and Maraim Masoud
  and Pierre Colombo and Javier de la Rosa and Paulo Villegas and Tristan Thrush and Shayne Longpre and Sebastian Nagel
  and Leon Weber and Manuel Muñoz and Jian Zhu and Daniel van Strien and Zaid Alyafeai and Khalid Almubarak and Minh
  Chien Vu and Itziar Gonzalez-Dios and Aitor Soroa and Kyle Lo and Manan Dey and Pedro Ortiz Suarez and Aaron Gokaslan
  and Shamik Bose and David Ifeoluwa Adelani and Long Phan and Hieu Tran and Ian Yu and Suhas Pai and Jenny Chim and
  Violette Lepercq and Suzana Ilic and Margaret Mitchell and Alexandra Sasha Luccioni and Yacine Jernite},
  title={The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset},
  year={2022},
  cdate={1640995200000},
  url={http://papers.nips.cc/paper_files/paper/2022/hash/ce9e92e3de2372a4b93353eb7f3dc0bd-Abstract-Datasets_and_Benchmarks.html},
  booktitle={NeurIPS},
}
"""

_DATASETNAME = "roots_vi_ted"

_DESCRIPTION = """
ROOTS_vi_ted is a subset of Vietnamese in ted_talks_iwslt datasets. ted_talks_iwslt is a collection of the original Ted
talks and their translated version. The translations are available in more than 109+ languages, though the distribution
is not uniform. Before using this dataloader, please accept the acknowledgement at
https://huggingface.co/datasets/bigscience-data/roots_vi_ted_talks_iwslt and use huggingface-cli login for authentication.
"""

_HOMEPAGE = "https://huggingface.co/datasets/bigscience-data/roots_vi_ted_talks_iwslt"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_NC_ND_4_0.value

_LOCAL = False

_URLS = {_DATASETNAME: {"train": "https://huggingface.co/datasets/bigscience-data/roots_vi_ted_talks_iwslt/resolve/main/data/train-00000-of-00001.parquet?download=true"}}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class RootsViTedDataset(datasets.GeneratorBasedBuilder):
    """RootsViTed is a subset of Vietnamese in ted_talks_iwslt datasets."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="roots_vi_ted_source",
            version=SOURCE_VERSION,
            description="roots_vi_ted source schema",
            schema="source",
            subset_id="roots_vi_ted",
        ),
        SEACrowdConfig(
            name="roots_vi_ted_seacrowd_ssp",
            version=SEACROWD_VERSION,
            description="roots_vi_ted SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id="roots_vi_ted",
        ),
    ]

    DEFAULT_CONFIG_NAME = "roots_vi_ted_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "meta": datasets.Value("string"),
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
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_dir, "split": "train"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        if self.config.schema == "source":
            df = pd.read_parquet(filepath[split])
            for i, row in df.iterrows():
                yield i, {
                    "text": row["text"],
                    "meta": row["meta"],
                }

        elif self.config.schema == "seacrowd_ssp":
            df = pd.read_parquet(filepath[split])
            for i, row in df.iterrows():
                yield i, {
                    "id": str(i),
                    "text": row["text"],
                }
