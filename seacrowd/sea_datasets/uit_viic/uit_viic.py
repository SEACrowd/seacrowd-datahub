# coding=utf-8
import json
import os.path

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_DATASETNAME = "uit_viic"
_CITATION = """\
@InProceedings{10.1007/978-3-030-63007-2_57,
author="Lam, Quan Hoang
and Le, Quang Duy
and Nguyen, Van Kiet
and Nguyen, Ngan Luu-Thuy",
editor="Nguyen, Ngoc Thanh
and Hoang, Bao Hung
and Huynh, Cong Phap
and Hwang, Dosam
and Trawi{\'{n}}ski, Bogdan
and Vossen, Gottfried",
title="UIT-ViIC: A Dataset for the First Evaluation on Vietnamese Image Captioning",
booktitle="Computational Collective Intelligence",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="730--742",
abstract="Image Captioning (IC), the task of automatic generation of image captions, has attracted
attentions from researchers in many fields of computer science, being computer vision, natural language
processing and machine learning in recent years. This paper contributes to research on Image Captioning
task in terms of extending dataset to a different language - Vietnamese. So far, there has been no existed
Image Captioning dataset for Vietnamese language, so this is the foremost fundamental step for developing
Vietnamese Image Captioning. In this scope, we first built a dataset which contains manually written
captions for images from Microsoft COCO dataset relating to sports played with balls, we called this dataset
UIT-ViIC (University Of Information Technology - Vietnamese Image Captions). UIT-ViIC consists of 19,250
Vietnamese captions for 3,850 images. Following that, we evaluated our dataset on deep neural network models
and did comparisons with English dataset and two Vietnamese datasets built by different methods. UIT-ViIC
is published on our lab website (https://sites.google.com/uit.edu.vn/uit-nlp/) for research purposes.",
isbn="978-3-030-63007-2"
}
"""

_DESCRIPTION = """
UIT-ViIC contains manually written captions for images from Microsoft COCO dataset relating to sports
played with ball. UIT-ViIC consists of 19,250 Vietnamese captions for 3,850 images. For each image,
UIT-ViIC provides five Vietnamese captions annotated by five annotators.
"""

_HOMEPAGE = "https://drive.google.com/file/d/1YexKrE6o0UiJhFWpE8M5LKoe6-k3AiM4"
_PAPER_URL = "https://arxiv.org/abs/2002.00175"
_LICENSE = Licenses.UNKNOWN.value
_HF_URL = ""
_LANGUAGES = ["vi"]
_LOCAL = False
_SUPPORTED_TASKS = [Tasks.IMAGE_CAPTIONING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_URLS = "https://drive.google.com/uc?export=download&id=1YexKrE6o0UiJhFWpE8M5LKoe6-k3AiM4"
_Split_Path = {
    "train": "UIT-ViIC/uitviic_captions_train2017.json",
    "validation": "UIT-ViIC/uitviic_captions_val2017.json",
    "test": "UIT-ViIC/uitviic_captions_test2017.json",
}


class UITViICDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SEACrowdConfig(name=f"{_DATASETNAME}_source", version=datasets.Version(_SOURCE_VERSION), description=_DESCRIPTION, subset_id=f"{_DATASETNAME}", schema="source"),
        SEACrowdConfig(name=f"{_DATASETNAME}_seacrowd_imtext", version=datasets.Version(_SEACROWD_VERSION), description=_DESCRIPTION, subset_id=f"{_DATASETNAME}", schema="seacrowd_imtext"),
    ]

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "license": datasets.Value("int32"),
                    "file_name": datasets.Value("string"),
                    "coco_url": datasets.Value("string"),
                    "flickr_url": datasets.Value("string"),
                    "height": datasets.Value("int32"),
                    "width": datasets.Value("int32"),
                    "date_captured": datasets.Value("string"),
                    "image_id": datasets.Value("int32"),
                    "caption": datasets.Value("string"),
                    "cap_id": datasets.Value("int32"),
                }
            )
        elif self.config.schema == "seacrowd_imtext":
            features = schemas.image_text_features()
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            license=_LICENSE,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        file_paths = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(file_paths, _Split_Path["train"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(file_paths, _Split_Path["validation"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(file_paths, _Split_Path["test"])},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            json_dict = json.load(f)
            images = {itm["id"]: itm for itm in json_dict["images"]}
            captns = json_dict["annotations"]

            for idx, capt in enumerate(captns):
                image_id = capt["image_id"]
                if self.config.schema == "source":
                    yield idx, {
                        "license": images[image_id]["license"],
                        "file_name": images[image_id]["file_name"],
                        "coco_url": images[image_id]["coco_url"],
                        "flickr_url": images[image_id]["flickr_url"],
                        "height": images[image_id]["height"],
                        "width": images[image_id]["width"],
                        "date_captured": images[image_id]["date_captured"],
                        "image_id": capt["image_id"],
                        "caption": capt["caption"],
                        "cap_id": capt["id"],
                    }
                elif self.config.schema == "seacrowd_imtext":
                    yield idx, {
                        "id": capt["id"],
                        "image_paths": [images[image_id]["coco_url"], images[image_id]["flickr_url"]],
                        "texts": capt["caption"],
                        "metadata": {
                            "context": "",
                            "labels": ["Yes"],
                        },
                    }
