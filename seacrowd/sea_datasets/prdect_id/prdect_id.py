from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@article{SUTOYO2022108554,
title = {PRDECT-ID: Indonesian product reviews dataset for emotions classification tasks},
journal = {Data in Brief},
volume = {44},
pages = {108554},
year = {2022},
issn = {2352-3409},
doi = {https://doi.org/10.1016/j.dib.2022.108554},
url = {https://www.sciencedirect.com/science/article/pii/S2352340922007612},
author = {Rhio Sutoyo and Said Achmad and Andry Chowanda and Esther Widhi Andangsari and Sani M. Isa},
keywords = {Natural language processing, Text processing, Text mining, Emotions classification, Sentiment analysis},
abstract = {Recognizing emotions is vital in communication. Emotions convey
additional meanings to the communication process. Nowadays, people can
communicate their emotions on many platforms; one is the product review. Product
reviews in the online platform are an important element that affects customers’
buying decisions. Hence, it is essential to recognize emotions from the product
reviews. Emotions recognition from the product reviews can be done automatically
using a machine or deep learning algorithm. Dataset can be considered as the
fuel to model the recognizer. However, only a limited dataset exists in
recognizing emotions from the product reviews, particularly in a local language.
This research contributes to the dataset collection of 5400 product reviews in
Indonesian. It was carefully curated from various (29) product categories,
annotated with five emotions, and verified by an expert in clinical psychology.
The dataset supports an innovative process to build automatic emotion
classification on product reviews.}
}
"""

_LOCAL = False
_LANGUAGES = ["ind"]
_DATASETNAME = "prdect_id"
_DESCRIPTION = """
PRDECT-ID Dataset is a collection of Indonesian product review data annotated
with emotion and sentiment labels. The data were collected from one of the giant
e-commerce in Indonesia named Tokopedia. The dataset contains product reviews
from 29 product categories on Tokopedia that use the Indonesian language. Each
product review is annotated with a single emotion, i.e., love, happiness, anger,
fear, or sadness. The group of annotators does the annotation process to provide
emotion labels by following the emotions annotation criteria created by an
expert in clinical psychology. Other attributes related to the product review
are also extracted, such as Location, Price, Overall Rating, Number Sold, Total
Review, and Customer Rating, to support further research.
"""

_HOMEPAGE = "https://data.mendeley.com/datasets/574v66hf2v/1"
_LICENSE = Licenses.CC_BY_4_0.value
_URL = "https://data.mendeley.com/public-files/datasets/574v66hf2v/files/f258d159-c678-42f1-9634-edf091a0b1f3/file_downloaded"

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS, Tasks.EMOTION_CLASSIFICATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class PrdectIDDataset(datasets.GeneratorBasedBuilder):
    """PRDECT-ID Dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "text"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_emotion_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_emotion",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_sentiment_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_sentiment",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_emotion_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema for emotion classification",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_emotion",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_sentiment_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema for sentiment analysis",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}_sentiment",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"
    CLASS_LABELS_EMOTION = ["Happy", "Sadness", "Anger", "Love", "Fear"]
    CLASS_LABELS_SENTIMENT = ["Positive", "Negative"]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "Category": datasets.Value("string"),
                    "Product Name": datasets.Value("string"),
                    "Location": datasets.Value("string"),
                    "Price": datasets.Value("int32"),
                    "Overall Rating": datasets.Value("float32"),
                    "Number Sold": datasets.Value("int32"),
                    "Total Review": datasets.Value("int32"),
                    "Customer Rating": datasets.Value("int32"),
                    "Customer Review": datasets.Value("string"),
                    "Sentiment": datasets.ClassLabel(names=self.CLASS_LABELS_SENTIMENT),
                    "Emotion": datasets.ClassLabel(names=self.CLASS_LABELS_EMOTION),
                }
            )
        elif self.config.schema == "seacrowd_text":
            if self.config.subset_id == f"{_DATASETNAME}_emotion":
                features = schemas.text_features(label_names=self.CLASS_LABELS_EMOTION)
            elif self.config.subset_id == f"{_DATASETNAME}_sentiment":
                features = schemas.text_features(label_names=self.CLASS_LABELS_SENTIMENT)
            else:
                raise ValueError(f"Invalid subset: {self.config.subset_id}")
        else:
            raise ValueError(f"Schema '{self.config.schema}' is not defined.")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_file = Path(dl_manager.download(_URL))
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_file})]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""
        df = pd.read_csv(filepath, encoding="utf-8")
        for idx, row in df.iterrows():
            if self.config.schema == "source":
                yield idx, dict(row)
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                if self.config.subset_id == f"{_DATASETNAME}_emotion":
                    yield idx, {"id": idx, "text": row["Customer Review"], "label": row["Emotion"]}
                elif self.config.subset_id == f"{_DATASETNAME}_sentiment":
                    yield idx, {"id": idx, "text": row["Customer Review"], "label": row["Sentiment"]}
                else:
                    raise ValueError(f"Invalid subset: {self.config.subset_id}")
