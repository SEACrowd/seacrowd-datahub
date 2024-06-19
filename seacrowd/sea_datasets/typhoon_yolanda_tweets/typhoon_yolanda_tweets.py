import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{imperial2019sentiment,
      title={Sentiment Analysis of Typhoon Related Tweets using Standard and Bidirectional Recurrent Neural Networks}, 
      author={Joseph Marvin Imperial and Jeyrome Orosco and Shiela Mae Mazo and Lany Maceda},
      year={2019},
      eprint={1908.01765},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
"""

_DATASETNAME = "typhoon_yolanda_tweets"

_DESCRIPTION = """\
The dataset contains annotated typhoon and disaster-related tweets in Filipino collected before, during, 
and after one month of Typhoon Yolanda in 2013. The dataset has been annotated by an expert into three 
sentiment categories: positive, negative, and neutral.
"""

_HOMEPAGE = "https://github.com/imperialite/Philippine-Languages-Online-Corpora/tree/master/Tweets/Annotated%20Yolanda"

_LOCAL = False
_LANGUAGES = ["fil"]

_LICENSE = Licenses.CC_BY_4_0.value

_ROOT_URL = "https://raw.githubusercontent.com/imperialite/Philippine-Languages-Online-Corpora/master/Tweets/Annotated%20Yolanda/"
_URLS = {"train": {-1: _ROOT_URL + "train/-1.txt", 0: _ROOT_URL + "train/0.txt", 1: _ROOT_URL + "train/1.txt"}, "test": {-1: _ROOT_URL + "test/-1.txt", 0: _ROOT_URL + "test/0.txt", 1: _ROOT_URL + "test/1.txt"}}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

class TyphoonYolandaTweets(datasets.GeneratorBasedBuilder):
    """
    The dataset contains annotated typhoon and disaster-related tweets in Filipino collected before, during, and
    after one month of Typhoon Yolanda in 2013. The dataset has been annotated by an expert into three sentiment
    categories: positive, negative, and neutral.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="typhoon_yolanda_tweets_source",
            version=SOURCE_VERSION,
            description="Typhoon Yolanda Tweets source schema",
            schema="source",
            subset_id="typhoon_yolanda_tweets",
        ),
        SEACrowdConfig(
            name="typhoon_yolanda_tweets_seacrowd_text",
            version=SEACROWD_VERSION,
            description="Typhoon Yolanda Tweets SEACrowd schema",
            schema="seacrowd_text",
            subset_id="typhoon_yolanda_tweets",
        ),
    ]

    DEFAULT_CONFIG_NAME = "typhoon_yolanda_tweets_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(["-1", "0", "1"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        emos = [-1, 0, 1]
        if self.config.name == "typhoon_yolanda_tweets_source" or self.config.name == "typhoon_yolanda_tweets_seacrowd_text":
            train_path = dl_manager.download_and_extract({emo: _URLS["train"][emo] for emo in emos})

            test_path = dl_manager.download_and_extract({emo: _URLS["test"][emo] for emo in emos})

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_path,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        if self.config.schema != "source" and self.config.schema != "seacrowd_text":
            raise ValueError(f"Invalid config: {self.config.name}")

        df = pd.DataFrame(columns=["text", "label"])

        if self.config.name == "typhoon_yolanda_tweets_source" or self.config.name == "typhoon_yolanda_tweets_seacrowd_text":
            for emo, file in filepath.items():
                with open(file) as f:
                    t = f.readlines()
                    l = [str(emo)]*(len(t))
                    tmp_df = pd.DataFrame.from_dict({"text": t, "label": l})
                    df = pd.concat([df, tmp_df], ignore_index=True)

        for row in df.itertuples():
            ex = {"id": str(row.Index), "text": row.text, "label": row.label}
            yield row.Index, ex
