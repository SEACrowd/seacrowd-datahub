import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{cruz2021exploiting,
  title={Exploiting news article structure for automatic corpus generation of entailment datasets},
  author={Cruz, Jan Christian Blaise and Resabal, Jose Kristian and Lin, James and Velasco, Dan John and Cheng, Charibeth},
  booktitle={PRICAI 2021: Trends in Artificial Intelligence: 18th Pacific Rim International Conference on Artificial Intelligence, PRICAI 2021, Hanoi, Vietnam, November 8--12, 2021, Proceedings, Part II 18},
  pages={86--99},
  year={2021},
  organization={Springer}
}
"""
_DATASETNAME = "newsph"
_LANGUAGES = ["fil", "tgl"]
_DESCRIPTION = """\
Raw collection of news articles in Filipino which can be used for language modelling.
"""
_HOMEPAGE = "https://huggingface.co/datasets/newsph"
_LICENSE = Licenses.GPL_3_0.value
_LOCAL = False
_URLS = "https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/newsph/newsph.zip"
_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class NewsPhDataset(datasets.GeneratorBasedBuilder):
    """
    Raw collection of news articles in Filipino which can be used for language modelling.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="newsph_source",
            version=SOURCE_VERSION,
            description="newsph source schema",
            schema="source",
            subset_id="newsph",
        ),
        SEACrowdConfig(
            name="newsph_seacrowd_ssp",
            version=SEACROWD_VERSION,
            description="newsph SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id="newsph",
        ),
    ]

    DEFAULT_CONFIG_NAME = "newsph_source"

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
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "newsph", "train.txt"),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source" or self.config.schema == "seacrowd_ssp":
            with open(filepath, encoding="utf-8") as f:
                for idx, row in enumerate(f):
                    if row.strip():
                        yield idx, {"id": str(idx), "text": row}
                    else:
                        yield idx, {"id": str(idx), "text": ""}
        else:
            raise NotImplementedError
