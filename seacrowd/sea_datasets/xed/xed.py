from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{ohman2020xed,
  title={{XED}: A Multilingual Dataset for Sentiment Analysis and Emotion Detection},
  author={{\"O}hman, Emily and P{`a}mies, Marc and Kajava, Kaisla and Tiedemann, J{\"o}rg},
  booktitle={The 28th International Conference on Computational Linguistics (COLING 2020)},
  year={2020}
}
"""
_DATASETNAME = "xed"

_DESCRIPTION = """\
This is the XED dataset. The dataset consists of emotion annotated movie subtitles
from OPUS. We use Plutchik's 8 core emotions to annotate. The data is multilabel.
The original annotations have been sourced for mainly English and Finnish, with the
rest created using annotation projection to aligned subtitles in 41 additional languages,
with 31 languages included in the final dataset (more than 950 lines of annotated subtitle
lines). The dataset is an ongoing project with forthcoming additions such as machine translated datasets.
"""

_HOMEPAGE = "https://github.com/Helsinki-NLP/XED"

_LANGUAGES = ["ind", "vie"]

# This License is from the bottom of homepage's README not Unknown (as from Issues)
_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {"ind": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections/id-projections.tsv", "vie": "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections/vi-projections.tsv"}

# Because of the multi-label attribute, I choose ASPECT_BASED_SENTIMENT_ANALYSIS than SENTIMENT_ANALYSIS
_SUPPORTED_TASKS = [Tasks.ASPECT_BASED_SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class XEDDataset(datasets.GeneratorBasedBuilder):
    """
    This is the XED dataset. The dataset consists of emotion annotated movie subtitles
    from OPUS. We use Plutchik's 8 core emotions to annotate. The data is multilabel.
    The original annotations have been sourced for mainly English and Finnish, with the
    rest created using annotation projection to aligned subtitles in 41 additional languages,
    with 31 languages included in the final dataset (more than 950 lines of annotated subtitle
    lines). The dataset is an ongoing project with forthcoming additions such as machine translated datasets.
    """

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{LANG}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} {LANG} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_{LANG}",
        )
        for LANG in _LANGUAGES
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{LANG}_seacrowd_text_multi",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} {LANG} SEACrowd schema",
            schema="seacrowd_text_multi",
            subset_id=f"{_DATASETNAME}_{LANG}",
        )
        for LANG in _LANGUAGES
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_ind_source"
    _LABELS = ["Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"]

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features({"Sentence": datasets.Value("string"), "Emotions": datasets.Sequence(feature=datasets.ClassLabel(names=self._LABELS))})

        elif self.config.schema == "seacrowd_text_multi":
            features = schemas.text_multi_features(self._LABELS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        language = self.config.name.split("_")[1]

        if language in _LANGUAGES:
            data_path = Path(dl_manager.download_and_extract(_URLS[language]))
        else:
            data_path = [Path(dl_manager.download_and_extract(_URLS[language])) for language in _LANGUAGES]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        emotions_mapping = {1: "Anger", 2: "Anticipation", 3: "Disgust", 4: "Fear", 5: "Joy", 6: "Sadness", 7: "Surprise", 8: "Trust"}

        df = pd.read_csv(filepath, sep="\t", names=["Sentence", "Emotions"], index_col=None)
        df["Emotions"] = df["Emotions"].apply(lambda x: list(map(int, x.split(", "))))
        df["Emotions"] = df["Emotions"].apply(lambda x: [emotions_mapping[emotion] for emotion in x])

        for index, row in df.iterrows():

            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == "seacrowd_text_multi":

                example = {
                    "id": str(index),
                    "text": str(row["Sentence"]),
                    "labels": row["Emotions"],
                }

            yield index, example
