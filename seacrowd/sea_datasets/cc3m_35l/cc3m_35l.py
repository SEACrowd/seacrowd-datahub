import os
from typing import Dict, List, Tuple

import datasets
import jsonlines as jl

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

_DATASETNAME = "cc3m_35l"

_DESCRIPTION = """\
    CC3M-35L is created by translating Conceptual Captions 3M (Sharma et al., 2018),
    originally in English, to the other 34 languages using Google's machine translation API.
"""

_HOMEPAGE = "https://google.github.io/crossmodal-3600/"

_LICENSE = Licenses.CC_BY_4_0.value

_URLS = {
    "images": "https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz",
    "train": "https://storage.googleapis.com/crossmodal-3600/cc3m_mt_train.jsonl.gz",
    "dev": "https://storage.googleapis.com/crossmodal-3600/cc3m_mt_dev.jsonl.gz",
}

_SUPPORTED_TASKS = [Tasks.IMAGE_CAPTIONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

_LANGS = ["fil", "id", "th", "vi"]


class CC3M35L(datasets.GeneratorBasedBuilder):
    """
    CC3M-35L is created by translating Conceptual Captions 3M (Sharma et al., 2018),
    originally in English, to the other 34 languages using Google's machine translation API.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [SEACrowdConfig(name=f"cc3m_35l_{lang}_source", version=datasets.Version(_SOURCE_VERSION), description=f"cc3m_35l_{lang} source schema", schema="source", subset_id=f"cc3m_35l_{lang}",) for lang in _LANGS] + [
        SEACrowdConfig(
            name=f"cc3m_35l_{lang}_seacrowd_imtext",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"cc3m_35l_{lang} SEACrowd schema",
            schema="seacrowd_imtext",
            subset_id=f"cc3m_35l_{lang}",
        )
        for lang in _LANGS
    ]

    DEFAULT_CONFIG_NAME = "cc3m_35l_id_source"

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
        train_path = dl_manager.download_and_extract(_URLS["train"])
        dev_path = dl_manager.download_and_extract(_URLS["dev"])
        images_path = dl_manager.extract("/project/dataset/images.tgz")  # remove on PR
        # images_path = dl_manager.download_and_extract(_URLS["images"])

        train_caps = {}
        dev_caps = {}
        # print(self.config.subset_id.split("_"))
        current_lang = self.config.subset_id.split("_")[2]

        with jl.open(os.path.join(train_path), mode="r") as j:
            for line in j:
                if line["trg_lang"] == current_lang:
                    train_caps[line["image_id"]] = line

        with jl.open(os.path.join(dev_path), mode="r") as j:
            for line in j:
                if line["trg_lang"] == current_lang:
                    dev_caps[line["image_id"]] = line

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": {"img_ids": train_caps.keys(), "images": {img_id: os.path.join(images_path, img_id + ".jpg") for img_id in train_caps.keys()}, "captions": train_caps},
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": {"img_ids": dev_caps.keys(), "images": {img_id: os.path.join(images_path, img_id + ".jpg") for img_id in dev_caps.keys()}, "captions": dev_caps},
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: dict, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        counter = 0
        for img_id in filepath["img_ids"]:
            if self.config.schema == "source":
                yield counter, {
                    "id": img_id + "_" + str(counter),
                    "image_paths": filepath["images"][img_id],
                    "src_lang": filepath["captions"][img_id]["src_lang"],
                    "caption_tokenized": filepath["captions"][img_id]["caption_tokenized"],
                    "trg_lang": filepath["captions"][img_id]["trg_lang"],
                    "translation_tokenized": filepath["captions"][img_id]["translation_tokenized"],
                    "backtranslation_tokenized": filepath["captions"][img_id]["backtranslation_tokenized"],
                }

            elif self.config.schema == "seacrowd_imtext":
                yield counter, {
                    "id": img_id + "_" + str(counter),
                    "image_paths": [filepath["images"][img_id]],
                    "texts": filepath["captions"][img_id]["translation_tokenized"],
                    "metadata": {
                        "context": None,
                        "labels": None,
                    },
                }

            else:
                raise ValueError(f"Invalid config: {self.config.name}")

            counter += 1
