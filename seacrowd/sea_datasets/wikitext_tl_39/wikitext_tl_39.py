import os
from pathlib import Path

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@article{cruz2019evaluating,
  title={Evaluating Language Model Finetuning Techniques for Low-resource Languages},
  author={Cruz, Jan Christian Blaise and Cheng, Charibeth},
  journal={arXiv preprint arXiv:1907.00409},
  year={2019}
}
"""

_DATASETNAME = "wikitext_tl_39"

_DESCRIPTION = """A benchmark Language Modeling dataset for Tagalog. The dataset construction was done similar to that of the WikiText
Long Term Dependency Language Modeling Dataset, with a some differences, such as in how Wikipedia was scraped and how the vocabulary was
created. The dataset contains 39 Million tokens in the training set.
"""

_HOMEPAGE = "https://huggingface.co/datasets/wikitext_tl39"

_LANGUAGES = ["fil"]

_LICENSE = Licenses.GPL_3_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/wikitext-tl-39/wikitext-tl-39.zip",
}

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class WikiTextTL39Dataset(datasets.GeneratorBasedBuilder):
    """Large scale, unlabeled text dataset with 39 Million tokens in the training set in Tagalog."""

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
            name=f"{_DATASETNAME}_seacrowd_ssp",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        features = schemas.ssp_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        data_dir = dl_manager.download_and_extract(_URLS[_DATASETNAME])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "wikitext-tl-39", "train.txt"), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "wikitext-tl-39", "test.txt"), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "wikitext-tl-39", "valid.txt"), "split": "valid"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> tuple[int, dict]:
        with open(filepath, encoding="utf-8") as f:
            for i, row in enumerate(f):
                if row.strip():
                    yield i, {
                        "id": str(i),
                        "text": row,
                    }
                else:
                    yield i, {
                        "id": str(i),
                        "text": "",
                    }
