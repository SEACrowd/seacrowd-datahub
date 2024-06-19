# coding=utf-8

from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

# no bibtex citation
_CITATION = ""

_DATASETNAME = "kheng_info"

_DESCRIPTION = """\
The Kheng.info Speech dataset was derived from recordings of Khmer words on the Khmer dictionary website kheng.info.
The recordings were recorded by a native Khmer speaker.
The recordings are short, generally ranging between 1 to 2 seconds only.
"""

_HOMEPAGE = "https://huggingface.co/datasets/seanghay/khmer_kheng_info_speech"

_LANGUAGES = ["khm"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://huggingface.co/datasets/seanghay/khmer_kheng_info_speech/resolve/main/data/train-00000-of-00001-4e7ad082a34164d1.parquet",
}

_SUPPORTED_TASKS = [Tasks.SPEECH_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class KhengInfoDataset(datasets.GeneratorBasedBuilder):
    """This is the Kheng.info Speech dataset, which wasderived from recordings on the Khmer dictionary website kheng.info"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_sptext",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_sptext",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"word": datasets.Value("string"), "duration_ms": datasets.Value("int64"), "audio": datasets.Audio(sampling_rate=16_000)})

        elif self.config.schema == "seacrowd_sptext":
            features = schemas.speech_text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            )
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        df = pd.read_parquet(filepath, engine="pyarrow")
        if self.config.schema == "source":
            for _id, row in df.iterrows():
                yield _id, {"word": row["word"], "duration_ms": row["duration_ms"], "audio": row["audio"]}
        elif self.config.schema == "seacrowd_sptext":
            for _id, row in df.iterrows():
                yield _id, {
                    "id": _id,
                    "path": row["audio"],
                    "audio": row["audio"],
                    "text": row["word"],
                    "speaker_id": None,
                    "metadata": {
                        "speaker_age": None,
                        "speaker_gender": None,
                    },
                }
