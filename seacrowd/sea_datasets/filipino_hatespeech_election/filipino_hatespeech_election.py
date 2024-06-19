import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@article{Cabasag-2019-hate-speech,
  title={Hate speech in Philippine election-related tweets: Automatic detection and classification using natural language processing.},
  author={Neil Vicente Cabasag, Vicente Raphael Chan, Sean Christian Lim, Mark Edward Gonzales, and Charibeth Cheng},
  journal={Philippine Computing Journal},
  volume={XIV},
  number={1},
  month={August},
  year={2019}
}
"""

_DATASETNAME = "filipino_hatespeech_election"

_DESCRIPTION = """
The dataset used in this study was a subset of the corpus 1,696,613 tweets crawled by Andrade et al. and posted from November 2015 to May 2016 during the campaign period for the Philippine presidential election. They were culled
based on the presence of candidate names (e.g., Binay, Duterte, Poe, Roxas, and Santiago) and election-related hashtags (e.g., #Halalan2016, #Eleksyon2016, and #PiliPinas2016). Data preprocessing was performed to prepare the
tweets for feature extraction and classification. It consisted of the following steps: data de-identification, uniform resource locator (URL) removal, special character processing, normalization, hashtag processing, and tokenization.
"""

_HOMEPAGE = "https://huggingface.co/datasets/hate_speech_filipino"

_LANGUAGES = ["fil"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {_DATASETNAME: "https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/hatenonhate/hatespeech_raw.zip"}

_SUPPORTED_TASKS = [Tasks.ABUSIVE_LANGUAGE_PREDICTION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_CLASSES = ["0", "1"]  # corresponds to  ["non-hate-containing", "hate-containing"]


class FilipinoHatespeechElectionDataset(datasets.GeneratorBasedBuilder):
    """Hate Speech Text Classification Dataset in Filipino."""

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
            name=f"{_DATASETNAME}_seacrowd_text",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        features = schemas.text_features(label_names=_CLASSES)

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
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "hatespeech", "train.csv"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "hatespeech", "test.csv"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "hatespeech", "valid.csv"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
            next(csv_reader)
            for i, row in enumerate(csv_reader):
                try:
                    text, label = row
                    yield i, {"id": str(i), "text": row[0], "label": _CLASSES[int(row[1].strip()) - 1]}
                except ValueError:
                    pass
