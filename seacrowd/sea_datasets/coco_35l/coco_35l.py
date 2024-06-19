import json
import os
from typing import Dict, List, Tuple

# import csv
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

_DATASETNAME = "coco_35l"

_DESCRIPTION = """\
    COCO-35L is a machine-generated image caption dataset, constructed by translating COCO Captions (Chen et al., 2015) to the other 34 languages using Google’s machine translation API.
    152520 image ids are not found in the coco 2014 training caption. Validation set is ok Using COCO 2014 train and validation set.
    """

_HOMEPAGE = "https://google.github.io/crossmodal-3600/"

_LICENSE = Licenses.CC_BY_4_0.value

_URLS = {
    "coco2017_train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "coco2014_train_images": "http://images.cocodataset.org/zips/train2014.zip",
    "coco2014_val_images": "http://images.cocodataset.org/zips/val2014.zip",
    "coco2014_train_val_annots": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "coco2017_train_val_annots": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "trans_train": "https://storage.googleapis.com/crossmodal-3600/coco_mt_train.jsonl.gz",
    "trans_dev": "https://storage.googleapis.com/crossmodal-3600/coco_mt_dev.jsonl.gz",
}

_SUPPORTED_TASKS = [Tasks.IMAGE_CAPTIONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_LANGUAGES = {"fil": "fil", "ind": "id", "tha": "th", "vie": "vi"}

_LOCAL = False

class Coco35LDataset(datasets.GeneratorBasedBuilder):
    """
    COCO-35L is a machine-generated image caption dataset, constructed by translating COCO Captions (Chen et al., 2015) to the other 34 languages using Google’s machine translation API.
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
        ) for lang in _LANGUAGES
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_seacrowd_imtext",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME}_{lang} SEACrowd schema",
            schema="seacrowd_imtext",
            subset_id=f"{_DATASETNAME}_{lang}",
        ) for lang in _LANGUAGES
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{sorted(_LANGUAGES)[0]}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "image_paths": datasets.Value("string"),
                    "src_lang": datasets.Value("string"),
                    "caption_tokenized": datasets.Value("string"),
                    "trg_lang": datasets.Value("string"),
                    "translation_tokenized": datasets.Value("string"),
                    "backtranslation_tokenized": datasets.Value("string"),
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
        trans_train_path = dl_manager.download_and_extract(_URLS["trans_train"])
        trans_val_path = dl_manager.download_and_extract(_URLS["trans_dev"])

        coco2014_train_val_annots_path = dl_manager.download_and_extract(_URLS["coco2014_train_val_annots"])
        coco2014_val_images_path = dl_manager.download_and_extract(_URLS["coco2014_val_images"])
        coco2014_train_images_path = dl_manager.download_and_extract(_URLS["coco2014_train_images"])

        trans_train_captions = {}
        trans_dev_captions = {}
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()

        current_lang = _LANGUAGES[self.config.subset_id.split("_")[2]]

        # the COCO dataset structure has separated the captions and images information. The caption's "image_id" key will refer to the image's "id" key.
        # load the image informations from COCO 2014 dataset and put it into a dataframe
        with open(os.path.join(coco2014_train_val_annots_path, "annotations", "captions_val2014.json")) as json_captions:
            captions = json.load(json_captions)
            val_df = pd.DataFrame(captions["images"])

        with open(os.path.join(coco2014_train_val_annots_path, "annotations", "captions_train2014.json")) as json_captions:
            captions = json.load(json_captions)
            train_df = pd.DataFrame(captions["images"])

        # the translated caption has "image_id" which refers to the "image_id" in the COCO annotations.
        # however we can skip this and connect it to the images' "id"
        # the example of an "image_id" in the translated caption -> "123456_0" since an image can has many descriptions.
        # thus, the real image_id to map it into the COCO image dataset is the "123456"
        with jl.open(trans_train_path, mode="r") as j:
            total = 0
            not_found = 0
            missing_ids = []
            for line in j:
                if line["trg_lang"] == current_lang:
                    total += 1

                    trans_img_id = line["image_id"]
                    coco2014_img_id = line["image_id"].split("_")[0]

                    # unfortunately, not all image_id in the translated caption can be found in the original COCO 2014.
                    # hence, we need to handle such errors
                    try:
                        filename = train_df.query(f"id=={int(coco2014_img_id)}")["file_name"].values[0]
                        trans_train_captions[trans_img_id] = line
                        trans_train_captions[trans_img_id]["filename"] = os.path.join(coco2014_train_images_path, "train2014", filename)
                    except IndexError:
                        missing_ids.append(trans_img_id)
                        not_found += 1
                        pass

        # the validation set are strangely okay. with no missing image_id(s)
        with jl.open(trans_val_path, mode="r") as j:
            for line in j:
                if line["trg_lang"] == current_lang:
                    trans_img_id = line["image_id"]
                    trans_dev_captions[trans_img_id] = line
                    coco2014_img_id = int(trans_img_id.split("_")[0])
                    filename = val_df.query(f"id=={coco2014_img_id}")["file_name"].values[0]
                    trans_dev_captions[trans_img_id]["filename"] = os.path.join(coco2014_val_images_path, "val2014", filename)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {
                        "images": trans_train_captions,
                    },
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": {
                        "images": trans_dev_captions,
                    },
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: dict, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        counter = 0
        for trans_img_id, data in filepath["images"].items():
            if self.config.schema == "source":
                yield counter, {
                    "id": trans_img_id + "_" + str(counter),
                    "image_paths": data["filename"],
                    "src_lang": data["src_lang"],
                    "caption_tokenized": data["caption_tokenized"],
                    "trg_lang": data["trg_lang"],
                    "translation_tokenized": data["translation_tokenized"],
                    "backtranslation_tokenized": data["backtranslation_tokenized"],
                }

            elif self.config.schema == "seacrowd_imtext":
                yield counter, {
                    "id": trans_img_id + "_" + str(counter),
                    "image_paths": [data["filename"]],
                    "texts": data["translation_tokenized"],
                    "metadata": {
                        "context": None,
                        "labels": None,
                    },
                }

            else:
                raise ValueError(f"Invalid config: {self.config.name}")

            counter += 1
