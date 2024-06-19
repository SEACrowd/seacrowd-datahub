import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{romadhona2022brcc,
  title={BRCC and SentiBahasaRojak: The First Bahasa Rojak Corpus for Pretraining and Sentiment Analysis Dataset},
  author={Romadhona, Nanda Putri and Lu, Sin-En and Lu, Bo-Han and Tsai, Richard Tzong-Han},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={4418--4428},
  year={2022},
  organization={International Committee on Computational Linguistics},
  address={Taiwan},
  email={nandadona61@gmail.com, {alznn, lu110522028, thtsai}@g.ncu.edu.tw}
}
"""

_DATASETNAME = "senti_bahasa_rojak"

_DESCRIPTION = """\
This dataset contains reviews for products, movies, and stocks in the Bahasa Rojak dialect,
a popular dialect in Malaysia that consists of English, Malay, and Chinese.
Each review is manually annotated as positive (bullish for stocks) or negative (bearish for stocks).
Reviews are generated through data augmentation using English and Malay sentiment analysis datasets.
"""

_HOMEPAGE = "https://data.depositar.io/dataset/brcc_and_sentibahasarojak/resource/8a558f64-98ff-4922-a751-0ce2ce8447bd"

_LANGUAGES = ["zlm", "eng", "cmn"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://data.depositar.io/dataset/304d1572-27d6-4549-8292-b1c8f5e9c086/resource/8a558f64-98ff-4922-a751-0ce2ce8447bd/download/BahasaRojak_Datasets.zip",
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class SentiBahasaRojakDataset(datasets.GeneratorBasedBuilder):
    """The BRCC (Bahasa Rojak Crawled Corpus) is a novel dataset designed for the study of Bahasa Rojak,
    a code-mixed dialect combining English, Malay, and Chinese, prevalent in Malaysia.
    This corpus is intended for pre-training language models and sentiment analysis,
    addressing the unique challenges of processing code-mixed languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    subsets = ["movie", "product", "stock"]

    BUILDER_CONFIGS = [SEACrowdConfig(name=f"{_DATASETNAME}.{sub}_source", version=datasets.Version(_SOURCE_VERSION), description=f"{_DATASETNAME}.{sub} source schema", schema="source", subset_id=f"{_DATASETNAME}.{sub}",) for sub in subsets] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}.{sub}_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME}.{sub} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}.{sub}",
        )
        for sub in subsets
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}.movie_source"
    LABELS = ["positive", "negative"]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=self.LABELS),
                }
            )
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(self.LABELS)

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
        data_dir = os.path.join(data_dir, "BahasaRojak Datasets", "SentiBahasaRojak")

        subset = self.config.name.split(".")[-1].split("_")[0]
        subset_dir = os.path.join(data_dir, f"SentiBahasaRojak-{subset.capitalize()}")
        filepath = {}
        if subset == "stock":
            for split in ["train", "valid", "test"]:
                filepath[split] = os.path.join(subset_dir, f"{split}_labeled.tsv")
        else:
            for split in ["train", "valid", "test"]:
                filepath[split] = os.path.join(subset_dir, f"mix.{split}")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": filepath["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": filepath["valid"],
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        if filepath.endswith(".tsv"):
            with open(filepath, encoding="utf-8") as file:
                reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row_idx, row in enumerate(reader):
                    if self.config.schema == "source":
                        yield row_idx, {
                            "text": row[0],
                            "label": "positive" if row[1] == 1 else "negative",
                        }
                    elif self.config.schema == "seacrowd_text":
                        yield row_idx, {
                            "id": row_idx,
                            "text": row[0],
                            "label": "positive" if row[1] == 1 else "negative",
                        }
        else:
            labelpath = filepath + ".label"
            with open(filepath, encoding="utf-8") as file, open(labelpath, encoding="utf-8") as label_file:
                for row_idx, (text, label) in enumerate(zip(file, label_file)):
                    if self.config.schema == "source":
                        yield row_idx, {
                            "text": text.strip(),
                            "label": label.strip(),
                        }
                    elif self.config.schema == "seacrowd_text":
                        yield row_idx, {
                            "id": row_idx,
                            "text": text.strip(),
                            "label": label.strip(),
                        }
