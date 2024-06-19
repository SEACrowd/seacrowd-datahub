import pickle
from pathlib import Path
from typing import List

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{andreaschandra2020,
  author    = {Chandra, Andreas},
  title     = {Indonesian News Dataset},
  year      = {2020},
  howpublished = {Online},
  url       = {https://github.com/andreaschandra/indonesian-news},
  note      = {Accessed: 2024-02-13},
}
"""

_LANGUAGES = ["ind"]

_DATASETNAME = "indonesian_news_dataset"

_DESCRIPTION = """An imbalanced dataset to classify Indonesian News articles.
The dataset contains 5 class labels: bola, news, bisnis, tekno, and otomotif.
The dataset comprises of around 6k train and 2.5k test examples, with the more prevalent classes
(bola and news) having roughly 10x the number of train and test examples than the least prevalent class (otomotif).
"""

_HOMEPAGE = "https://github.com/andreaschandra/indonesian-news"

_LICENSE = Licenses.UNKNOWN.value

_URLS = {
    f"{_DATASETNAME}_train": "https://drive.usercontent.google.com/u/0/uc?id=1wCwPMKSyTciv8I3g9xGdUfEraA1SydG6&export=download",
    f"{_DATASETNAME}_test": "https://drive.usercontent.google.com/u/0/uc?id=1AFW_5KQFW86jlFO16S9bt564Y86WoJjV&export=download",
}

_SUPPORTED_TASKS = [Tasks.TOPIC_MODELING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_TAGS = ["bola", "news", "bisnis", "tekno", "otomotif"]

_LOCAL = False


class IndonesianNewsDataset(datasets.GeneratorBasedBuilder):
    """The dataset contains 5 Indonesian News articles with imbalanced classes"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "text"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
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
            features = datasets.Features({"index": datasets.Value("string"), "news": datasets.Value("string"), "label": datasets.Value("string")})
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(_TAGS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        train_dir = Path(dl_manager.download(_URLS[f"{_DATASETNAME}_train"]))
        test_dir = Path(dl_manager.download(_URLS[f"{_DATASETNAME}_test"]))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_dir,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str):
        """Yields examples as (key, example) tuples."""

        with open(filepath, "rb") as file:
            news_file = pickle.load(file)

        news_list = news_file[0]
        label_list = news_file[1]

        if self.config.schema == "source":
            for idx, (news, label) in enumerate(zip(news_list, label_list)):
                example = {"index": str(idx), "news": news, "label": label}
                yield idx, example
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for idx, (news, label) in enumerate(zip(news_list, label_list)):
                example = {"id": str(idx), "text": news, "label": label}
                yield idx, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
