import os
from typing import Dict, List, Tuple

import datasets
import jsonlines as jl
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{thapliyal-etal-2022-crossmodal,
    title = "Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset",
    author = "Thapliyal, Ashish V.  and
      Pont Tuset, Jordi  and
      Chen, Xi  and
      Soricut, Radu",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.45",
    doi = "10.18653/v1/2022.emnlp-main.45",
    pages = "715--729",
}
"""

_DATASETNAME = "xm3600"

_DESCRIPTION = """\
Crossmodal-3600 dataset (XM3600 in short), a geographically-diverse set of 3600 images annotated with
human-generated reference captions in 36 languages. The images were selected from across the world,
covering regions where the languages are spoken, and annotated with captions that achieve consistency in
terms of style across all languages, while avoiding annotation artifacts due to direct translation.
The languages covered in the dataset include Filipino, Indonesian, Thai, and Vietnamnese
"""

_HOMEPAGE = "https://google.github.io/crossmodal-3600/"

_LICENSE = Licenses.CC_BY_4_0.value

_URLS = {
    "captions": "https://google.github.io/crossmodal-3600/web-data/captions.zip",
    "images": "https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz",
    "image_attributions": "https://google.github.io/crossmodal-3600/web-data/image_attributions.csv",
}

_SUPPORTED_TASKS = [Tasks.IMAGE_CAPTIONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_LANGUAGES = ["fil", "id", "th", "vi"]

_LOCAL = False


class XM3600Dataset(datasets.GeneratorBasedBuilder):
    """
    Crossmodal-3600 dataset (XM3600 in short), a geographically-diverse set of 3600 images annotated with
    human-generated reference captions in 36 languages. The images were selected from across the world,
    covering regions where the languages are spoken, and annotated with captions that achieve consistency in
    terms of style across all languages, while avoiding annotation artifacts due to direct translation.
    The languages covered in the dataset include Filipino, Indonesian, Thai, and Vietnamnese
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME}_{lang} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_{lang}",
        )
        for lang in _LANGUAGES
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_seacrowd_imtext",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME}_{lang} SEACrowd schema",
            schema="seacrowd_imtext",
            subset_id=f"{_DATASETNAME}_{lang}",
        )
        for lang in _LANGUAGES
    ]

    DEFAULT_CONFIG_NAME = f"xm3600_{sorted(_LANGUAGES)[0]}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "image_paths": datasets.Value("string"),
                    "texts": {
                        "caption": datasets.Value("string"),
                        "caption/tokenized": datasets.Value("string"),
                        "caption/tokenized/lowercase": datasets.Value("string"),
                    },
                }
            )
        elif self.config.schema == "seacrowd_imtext":
            features = schemas.image_text_features()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        captions_path = dl_manager.download_and_extract(_URLS["captions"])
        images_path = dl_manager.download_and_extract(_URLS["images"])
        attr_path = dl_manager.download(_URLS["image_attributions"])

        train_caps = {}
        test_caps = {}
        val_caps = {}

        current_lang = self.config.subset_id.split("_")[1]

        img_df = pd.read_csv(attr_path)

        img_df_train = img_df.loc[img_df["Subset"] == "train"][["ImageID", "Subset"]]
        img_df_test = img_df.loc[img_df["Subset"] == "test"][["ImageID", "Subset"]]
        img_df_val = img_df.loc[img_df["Subset"] == "validation"][["ImageID", "Subset"]]

        with jl.open(os.path.join(captions_path, "captions.jsonl"), mode="r") as jsonl_file:
            for line in jsonl_file:
                if line["image/key"] in img_df_train.ImageID.values:
                    train_caps[line["image/key"]] = line[current_lang]
                elif line["image/key"] in img_df_test.ImageID.values:
                    test_caps[line["image/key"]] = line[current_lang]
                elif line["image/key"] in img_df_val.ImageID.values:
                    val_caps[line["image/key"]] = line[current_lang]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {"img_ids": img_df_train.ImageID.values, "images": {img_id: os.path.join(images_path, img_id + ".jpg") for img_id in img_df_train.ImageID.values}, "captions": train_caps},
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": {"img_ids": img_df_test.ImageID.values, "images": {img_id: os.path.join(images_path, img_id + ".jpg") for img_id in img_df_test.ImageID.values}, "captions": test_caps},
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": {"img_ids": img_df_val.ImageID.values, "images": {img_id: os.path.join(images_path, img_id + ".jpg") for img_id in img_df_val.ImageID.values}, "captions": val_caps},
                },
            ),
        ]

    def _generate_examples(self, filepath: dict) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        counter = 0
        for img_id in filepath["img_ids"]:
            cap = filepath["captions"][img_id]
            for line in cap["caption"]:
                cap_index = cap["caption"].index(line)
                if self.config.schema == "source":
                    yield counter, {
                        "id": img_id + "_" + str(counter),
                        "image_paths": filepath["images"][img_id],
                        "texts": {
                            "caption": line,
                            "caption/tokenized": cap["caption/tokenized"][cap_index],
                            "caption/tokenized/lowercase": cap["caption/tokenized/lowercase"][cap_index],
                        },
                    }

                elif self.config.schema == "seacrowd_imtext":
                    yield counter, {
                        "id": img_id + "_" + str(counter),
                        "image_paths": [filepath["images"][img_id]],
                        "texts": line,
                        "metadata": {
                            "context": None,
                            "labels": None,
                        },
                    }

                else:
                    raise ValueError(f"Invalid config: {self.config.name}")

                counter += 1
