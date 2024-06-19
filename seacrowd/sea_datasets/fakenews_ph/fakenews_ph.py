import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = r"""
@inproceedings{cruz-etal-2020-localization,
    title = "Localization of Fake News Detection via Multitask Transfer Learning",
    author = "Cruz, Jan Christian Blaise  and
      Tan, Julianne Agatha  and
      Cheng, Charibeth",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.316",
    pages = "2596--2604",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
"""

_LOCAL = False
_LANGUAGES = ["fil"]
_DATASETNAME = "fakenews_ph"
_DESCRIPTION = """\
Fake news articles were sourced from online sites that were tagged as fake
news sites by the non-profit independent media fact-checking organization
Verafiles and the National Union of Journalists in the Philippines (NUJP).
Real news articles were sourced from mainstream news websites in the
Philippines, including Pilipino Star Ngayon, Abante, and Bandera.
"""

_HOMEPAGE = "https://github.com/jcblaisecruz02/Tagalog-fake-news"
_LICENSE = Licenses.GPL_3_0.value
_URL = "https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/fakenews/fakenews.zip"

_SUPPORTED_TASKS = [Tasks.HOAX_NEWS_CLASSIFICATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class FakeNewsFilipinoDataset(datasets.GeneratorBasedBuilder):
    """Fake News Filipino Dataset from https://huggingface.co/datasets/fake_news_filipino"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "text"
    LABEL_CLASSES = ["0", "1"]

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
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "article": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=self.LABEL_CLASSES),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_features(self.LABEL_CLASSES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        """Return SplitGenerators."""
        data_dir = Path(dl_manager.download_and_extract(_URL))
        train_path = data_dir / "fakenews" / "full.csv"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_path, "split": "train"},
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
            )
            next(csv_reader)
            for id_, row in enumerate(csv_reader):
                label, article = row
                if self.config.schema == "source":
                    yield id_, {"label": label, "article": article}
                if self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                    yield id_, {"id": id_, "label": label, "text": article}
