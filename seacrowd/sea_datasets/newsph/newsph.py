import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{cruz2020investigating,
  title={Investigating the True Performance of Transformers in Low-Resource Languages: A Case Study in Automatic Corpus Creation},
  author={Jan Christian Blaise Cruz and Jose Kristian Resabal and James Lin and Dan John Velasco and Charibeth Cheng},
  journal={arXiv preprint arXiv:2010.11574},
  year={2020}
}
"""
_DATASETNAME = "newsph"
_LANGS = ["fil", "tgl"]
_DESCRIPTION = """\
Raw collection of news articles in Filipino which can be used for language modelling.
"""
_HOMEPAGE = "https://huggingface.co/datasets/newsph"
_LICENSE = Licenses.GPL_3_0.value
_LOCAL = False
_URLS = "https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/newsph/newsph-nli.zip"
_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]
_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


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
            name="newsph_seacrowd_pairs",
            version=SEACROWD_VERSION,
            description="newsph SEACrowd schema",
            schema="seacrowd_pairs",
            subset_id="newsph",
        ),
    ]

    DEFAULT_CONFIG_NAME = "newsph_source"

    def _info(self) -> datasets.DatasetInfo:
        label_classes = [0, 1]

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=label_classes),
                }
            )
        elif self.config.schema == "seacrowd_pairs":
            features = schemas.pairs_features(label_classes)
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
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "newsph-nli", "train.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "newsph-nli", "test.csv"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "newsph-nli", "valid.csv"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source" or self.config.schema == "seacrowd_pairs":
            df = pd.read_csv(filepath)
            for key in df.index:
                yield key, {
                    "id": str(key),
                    "text_1": df["s1"][key],
                    "text_2": df["s2"][key],
                    "label": df["label"][key],
                }
        else:
            raise NotImplementedError
