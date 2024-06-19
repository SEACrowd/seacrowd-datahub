from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{hernandez2021bert,
  title={A bert-based hate speech classifier from transcribed online short-form videos},
  author={Hernandez Urbano Jr, Rommel and Uy Ajero, Jeffrey and Legaspi Angeles, Angelic and Hacar Quintos, Maria Nikki and Regalado Imperial, Joseph Marvin and Llabanes Rodriguez, Ramon},
  booktitle={2021 5th International Conference on E-Society, E-Education and E-Technology},
  year={2021}
}
"""

_LOCAL = False
_LANGUAGES = ["tgl", "eng"]
_DATASETNAME = "filipino_hatespeech_tiktok"

_DESCRIPTION = """\
The dataset contains annotated hate speech from transcribed Tiktok videos, mostly in Taglish (codemixed Tagalog and Cebuano) collected via an unofficial Tiktok API.
Most of the domain and context of the videos are related to politics and general elections in the Philippines. Labeling: 0 - no hate speech, 1 - recognized hate speech
"""

_HOMEPAGE = "https://github.com/imperialite/filipino-tiktok-hatespeech"
_LICENSE = Licenses.CC_BY_SA_4_0.value
_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_URL = "https://raw.githubusercontent.com/imperialite/filipino-tiktok-hatespeech/main/data"


class FilipinoHateSpeechTikTokDataset(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="Filipino TikTok Hatespeech source schema",
            schema="source",
            subset_id="filipino_hatespeech_tiktok",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text",
            version=SEACROWD_VERSION,
            description="Filipino TikTok Hatespeech Seacrowd schema",
            schema="seacrowd_text",
            subset_id="filipino_hatespeech_tiktok",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"text": datasets.Value("string"), "label": datasets.Value("string")})
        elif self.config.schema == "seacrowd_text":
            features = schemas.text.features(label_names=["0", "1"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        base_dir = _URL
        data_files = {
            "train": Path(dl_manager.download_and_extract(base_dir + "/train.csv")),
            "test": Path(dl_manager.download_and_extract(base_dir + "/test.csv")),
            "validation": Path(dl_manager.download_and_extract(base_dir + "/valid.csv")),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_files["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_files["validation"],
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        df = pd.read_csv(filepath).reset_index()
        id = 0
        if self.config.schema == "source":
            for row in df.itertuples():
                ex = {"text": row.text, "label": str(row.label)}
                yield id, ex
                id += 1

        elif self.config.schema == "seacrowd_text":
            for row in df.itertuples():
                # Skip NA labels
                if pd.isna(row.label):
                    continue
                ex = {"id": id, "text": row.text, "label": str(int(row.label))}
                yield id, ex
                id += 1
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
