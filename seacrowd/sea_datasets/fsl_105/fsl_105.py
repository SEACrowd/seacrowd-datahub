import os
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple

try:
    import cv2
except:
    print("Install the `cv2` package to use.")
import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{tupal4476867fsl105,
  title={FSL105: The Video Filipino Sign Language Sign Database of Introductory 105 FSL Signs},
  author={Tupal, Isaiah Jassen Lizaso and Melvin, Cabatuan K},
  journal={Available at SSRN 4476867}
}
"""

_DATASETNAME = "fsl_105"

_DESCRIPTION = """\
FSL-105 is a video dataset for 105 different Filipino Sign Language (FSL) signs.
Each sign is categorized into one of 10 categories and is each represented by approximately 20 four-second video samples.
Signs were performed by adult deaf FSL signers on a blank blue background and reviewed by an FSL expert.
"""

_HOMEPAGE = "https://data.mendeley.com/datasets/48y2y99mb9/2"

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    "clips": "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/de95a3c3-02f4-4a3f-9a9e-ce2371160275",
    "train": "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/09c71779-3a2a-4c98-8d9b-0ef74f54d92a",
    "test": "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/39af8117-6b44-47b9-a551-0bdc40837295",
}

_LANGUAGES = ["psp"]

_SUPPORTED_TASKS = [Tasks.VIDEO_TO_TEXT_RETRIEVAL, Tasks.VIDEO_CAPTIONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class FSL105Dataset(datasets.GeneratorBasedBuilder):
    """
    FSL-105 is a video dataset for 105 different Filipino Sign Language (FSL) signs.
    Each sign is categorized into one of 10 categories and is each represented by approximately 20 four-second video samples.
    Signs were performed by adult deaf FSL signers on a blank blue background and reviewed by an FSL expert.
    """

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
            name=f"{_DATASETNAME}_seacrowd_vidtext",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_vidtext",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    category = [
        "CALENDAR",
        "COLOR",
        "DAYS",
        "DRINK",
        "FAMILY",
        "FOOD",
        "GREETING",
        "NUMBER",
        "RELATIONSHIPS",
        "SURVIVAL",
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "video_path": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "labels": datasets.ClassLabel(names=self.category),
                    "metadata": {
                        "resolution": {
                            "width": datasets.Value("int64"),
                            "height": datasets.Value("int64"),
                        },
                        "duration": datasets.Value("float32"),
                        "fps": datasets.Value("float32"),
                    },
                }
            )

        elif self.config.schema == "seacrowd_vidtext":
            features = schemas.video_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        clips = dl_manager.download_and_extract(_URLS["clips"])
        train = dl_manager.download_and_extract(_URLS["train"])
        test = dl_manager.download_and_extract(_URLS["test"])

        train_df = pd.read_csv(train)
        test_df = pd.read_csv(test)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {
                        "clips": clips,
                        "data": train_df,
                    },
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {"clips": clips, "data": test_df},
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        for key, example in filepath["data"].iterrows():
            video = cv2.VideoCapture(os.path.join(filepath["clips"], PureWindowsPath(example["vid_path"]).as_posix()))
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps
            vid_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

            if self.config.schema == "source":
                yield key, {
                    "id": str(key),
                    "video_path": os.path.join(filepath["clips"], example["vid_path"]),
                    "text": example["label"],
                    "labels": example["category"],
                    "metadata": {
                        "resolution": {
                            "width": vid_width,
                            "height": vid_height,
                        },
                        "duration": duration,
                        "fps": fps,
                    },
                }
            elif self.config.schema == "seacrowd_vidtext":
                yield key, {
                    "id": str(key),
                    "video_path": os.path.join(filepath["clips"], example["vid_path"]),
                    "text": example["label"],
                    "metadata": {
                        "resolution": {
                            "width": vid_width,
                            "height": vid_height,
                        },
                        "duration": duration,
                        "fps": fps,
                    },
                }
