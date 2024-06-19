from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (SCHEMA_TO_FEATURES, TASK_TO_SCHEMA,
                                      Licenses, Tasks)

_CITATION = """\
@article{hendria2023msvd,
    author    = {Willy Fitra Hendria},
    title     = {MSVD-Indonesian: A Benchmark for Multimodal Video-Text Tasks in Indonesian},
    journal   = {arXiv preprint arXiv:2306.11341},
    year      = {2023},
    url       = {https://arxiv.org/abs/2306.11341},
}
"""

_DATASETNAME = "id_msvd"
_DESCRIPTION = """\
MSVD-Indonesian is derived from the MSVD (Microsoft Video Description) dataset, which is
obtained with the help of a machine translation service (Google Translate API). This
dataset can be used for multimodal video-text tasks, including text-to-video retrieval,
video-to-text retrieval, and video captioning. Same as the original English dataset, the
MSVD-Indonesian dataset contains about 80k video-text pairs.
"""

_HOMEPAGE = "https://github.com/willyfh/msvd-indonesian"
_LANGUAGES = ["ind"]
_LICENSE = Licenses.MIT.value
_URLS = {"text": "https://raw.githubusercontent.com/willyfh/msvd-indonesian/main/data/MSVD-indonesian.txt", "video": "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"}

_SUPPORTED_TASKS = [Tasks.VIDEO_TO_TEXT_RETRIEVAL]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_LOCAL = False


class IdMsvdDataset(datasets.GeneratorBasedBuilder):
    """MSVD dataset with Indonesian translation."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()  # "vidtext"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
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
            features = datasets.Features(
                {
                    "video_path": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = SCHEMA_TO_FEATURES[self.SEACROWD_SCHEMA_NAME.upper()]  # video_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # expect several minutes to download video data ~1.7GB
        data_path = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "text_path": Path(data_path["text"]),
                    "video_path": Path(data_path["video"]) / "YouTubeClips",
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, text_path: Path, video_path: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        text_data = []
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                id = line.find(" ")
                video = line[:id]
                text = line[id + 1 :].strip()
                text_data.append([video, text])

        df = pd.DataFrame(text_data, columns=["video_path", "text"])
        df["video_path"] = df["video_path"].apply(lambda x: video_path / f"{x}.avi")

        if self.config.schema == "source":
            for i, row in df.iterrows():
                yield i, {
                    "video_path": str(row["video_path"]),
                    "text": row["text"],
                }
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for i, row in df.iterrows():
                yield i, {
                    "id": str(i),
                    "video_path": str(row["video_path"]),
                    "text": row["text"],
                    "metadata": {
                        "resolution": {
                            "width": None,
                            "height": None,
                        },
                        "duration": None,
                        "fps": None,
                    },
                }
