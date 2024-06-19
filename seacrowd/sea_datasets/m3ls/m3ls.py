"""
SEACrowd Data Loader for M3LS.
"""
import json
import os
from collections.abc import Iterable
from copy import deepcopy
from typing import Dict, Generator, List, Tuple, Union

try:
    import PIL
except (ImportError, ModuleNotFoundError):
    print("Please install `PIL` to load image-based data from M3LS dataloader.")
else:
    PIL.__version__  # to avoid being marked by formatter

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = r"""
@inproceedings{verma-etal-2023-large,
    title = "Large Scale Multi-Lingual Multi-Modal Summarization Dataset",
    author = "Verma, Yash  and
      Jangra, Anubhav  and
      Verma, Raghvendra  and
      Saha, Sriparna",
    editor = "Vlachos, Andreas  and
      Augenstein, Isabelle",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.263",
    doi = "10.18653/v1/2023.eacl-main.263",
    pages = "3620--3632",
}
"""

logger = datasets.logging.get_logger(__name__)

_LOCAL = False
_LANGUAGES = ["ind"]


_DATASETNAME = "m3ls"
_DESCRIPTION = r"""
The multilingual multimodal summarization dataset (M3LS) consists of over a million instances of document-image pairs
along with a professionally annotated multimodal summary for each pair.
It is derived from news articles published by the British Broadcasting Corporation (BBC) over a decade and spans 20 total languages,
which Indonesian is the only SEA language available on this dataset.
"""

_HOMEPAGE = "https://github.com/anubhav-jangra/M3LS/tree/main"
_LICENSE = Licenses.MIT.value

_URL = "https://drive.google.com/uc?id=1Kznkw7YpRiWpdgH4_SVNwp0uGf3j-5e2"

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION, Tasks.IMAGE_CAPTIONING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]


class M3LSDataset(datasets.GeneratorBasedBuilder):
    """M3LS dataset of Indonesian Language (from BBC Indonesian)"""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        *[
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_{cfg_sufix}",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} seacrowd schema for {task.name}",
                schema=f"seacrowd_{cfg_sufix}",
                subset_id=f"{_DATASETNAME}",
            )
            for task, cfg_sufix in zip(_SUPPORTED_TASKS, _CONFIG_SUFFIXES_FOR_TASK)
        ],
    ]

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")

        if _config_schema_name == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "date": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                    "keyword": datasets.Sequence(datasets.Value("string")),
                    "related": datasets.Sequence(datasets.Value("string")),
                    "section_headers": datasets.Sequence(datasets.Value("string")),
                    "paragraphs": datasets.Sequence(datasets.Value("string")),
                    "images": datasets.Sequence(datasets.Image()),
                    "captions": datasets.Sequence(datasets.Value("string")),
                }
            )

        # speech-text schema
        elif _config_schema_name == "seacrowd_t2t":
            features = schemas.text2text_features

        elif _config_schema_name == "seacrowd_imtext":
            features = schemas.image_text_features()

        else:
            raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        try:
            import gdown
        except ImportError:
            raise ImportError("Please install `gdown` to enable downloading data from google drive.")

        # Download from Google drive
        output_dir = os.path.join(os.getcwd(), "data", "m3ls")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = output_dir + "/m3ls.zip"
        if not os.path.exists(output_file):
            gdown.download(_URL, str(output_file), fuzzy=True)
        else:
            logger.info(f"File already downloaded: {str(output_file)}")

        local_path = os.path.join(dl_manager.extract(output_file).title(), "bbcindonesia")

        # there are two folders all containing json files, namely "processed" and "articles"
        # both are having articles info with url, text, and accompanied resource scrapped (i.e image & captions, related articles)

        # the "processed" contains only 244 data, which 156 of them doesn't have any title info
        # whereas "articles" contains 56108 data (the same reported as the wholly data in paper), all having title info

        # no intersection of links for both, nor information provided, hence we will only take "articles" due to matched info w/ their paper
        # the original paper mentioned 80:10:10 splits for over, but there is no info for such splitting index on the extracted folder
        article_data_dir = os.path.join(local_path, "articles")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "article_data_dir": article_data_dir,
                    "image_folder": os.path.join(local_path, "imagefolder"),
                },
            )
        ]

    def _generate_examples(self, article_data_dir: str, image_folder: str) -> Generator[Tuple[int, Dict], None, None]:
        _config_schema_name = self.config.schema
        all_image_filename = os.listdir(image_folder)

        idx = 1
        im_data_idx = 1
        for filename in os.listdir(article_data_dir):
            root_data, content_data = self.__json_read_and_process(os.path.join(article_data_dir, filename))

            # for images, it has around 6.7% missing rate (15625 out of 230163)
            if _config_schema_name == "source":
                content_data = self.__m3ls_content_data_reconstructor_and_validator(content_data, mode="all")
                image_path, captions = self.__m3ls_filter_image_and_captions_data(content_data["image_paths"], content_data["captions"], image_folder, all_image_filename)

                yield idx, {
                    "id": idx,
                    "date": root_data["date"],
                    "url": root_data["url"],
                    "title": root_data["title"],
                    "summary": root_data["summary"],
                    "keyword": root_data["keyword"],
                    "related": root_data["related"],
                    "section_headers": content_data["section_headers"],
                    "paragraphs": content_data["paragraphs"],
                    "images": image_path,
                    "captions": captions,
                }

            elif _config_schema_name == "seacrowd_t2t":
                content_data = self.__m3ls_content_data_reconstructor_and_validator(content_data, mode="text")
                yield idx, {
                    "id": idx,
                    "text_1": "\n".join(content_data["paragraphs"]),
                    "text_2": root_data["summary"],
                    "text_1_name": "texts",
                    "text_2_name": "summary",
                }

            elif _config_schema_name == "seacrowd_imtext":
                content_data = self.__m3ls_content_data_reconstructor_and_validator(content_data, mode="image")
                image_path, captions = self.__m3ls_filter_image_and_captions_data(content_data["image_paths"], content_data["captions"], image_folder, all_image_filename, both_exists=True)

                if image_path == []:
                    continue

                for path_idx in range(len(image_path)):
                    yield im_data_idx, {
                        "id": im_data_idx,
                        "image_paths": [image_path[path_idx]],
                        "texts": captions[path_idx],
                        "metadata": {
                            "context": root_data["url"],
                            "labels": None,
                        },
                    }
                    im_data_idx += 1

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

            idx += 1

    @staticmethod
    def __check_only_1level_iterables(iter_obj):
        return all([not isinstance(data, Iterable) or isinstance(data, str) for data in iter_obj])

    @classmethod
    def __json_read_and_process(cls, path: str) -> Dict:

        # to check (for compulsory keys) and reconstruct (for optional keys) the json data
        def base_data_reconstructor(json_data: dict, return_split: bool = True) -> Union[Dict, Tuple[Dict, Dict]]:

            # for detecting content-based dict-keys (it's denoted by int-based keys in string type)
            def parse_or_check_int(val: Union[int, str, float], is_parse: bool = True):
                try:
                    int(val)
                except (ValueError, TypeError):
                    return val if is_parse else False
                else:
                    return int(val) if is_parse else True

            compulsory_keys = ["summary", "url", "title"]
            optional_keys = ["date", "keyword", "related"]
            optional_key_mapper = list(zip(optional_keys, ["Not available", [], []]))

            if any(key not in json_data.keys() for key in compulsory_keys):
                raise KeyError(f"Missing keys of {list(set(compulsory_keys).difference(json_data.keys()))}")

            for key, default_val in optional_key_mapper:
                _existing_val = json_data.get(key)
                new_data = {key: json_data.get(key) if _existing_val is not None else default_val}
                json_data.update(new_data)

            all_content_keys = [key for key in json_data.keys() if parse_or_check_int(key, is_parse=False)]

            if sorted(compulsory_keys + optional_keys + all_content_keys) != sorted(json_data.keys()):
                raise KeyError("Some keys are unexpectedly missing or present!")

            content_data = {key: json_data[key] for key in all_content_keys}

            if not return_split:
                json_data.update(content_data)
                return json_data
            else:
                root_data = {key: val for key, val in json_data.items() if key not in all_content_keys}
                return root_data, content_data

        def non_content_data_validator(json_data: dict):
            non_content_dtypes = [("url", str), ("title", str), ("date", str), ("summary", str), ("keyword", list), ("related", list)]

            for key, _type in non_content_dtypes:
                if not isinstance(json_data[key], _type):
                    raise TypeError(f"The dict has key {key} that doesn't match with expected type {_type}!")

                # assert only 1-level for list types
                if _type == list:
                    if not cls.__check_only_1level_iterables(json_data[key]):
                        raise ValueError(f"Found iterables in {key} for val {json_data[key]}")

        with open(path, "r") as f:
            json_input = json.load(f)

        base_data, content_data = base_data_reconstructor(json_input)

        non_content_data_validator(base_data)

        return base_data, content_data

    @classmethod
    def __m3ls_content_data_reconstructor_and_validator(cls, json_content_data: Dict, mode: str = "all") -> Dict:
        # `mode` variable scope will be shared to all subfunctions under this fn
        if mode not in ("all", "image", "text"):
            raise ValueError("Unexpected `mode`! Accepted: 'all', 'image', or 'text'.")

        all_content_ftrs = ("images", "para", "subheading")
        expected_dtypes = (list, list, str)
        default_values = ([["", ""]], [], "")

        _all_ftr_validation_info = {all_content_ftrs[_idx]: {"dtype": expected_dtypes[_idx], "default_val": default_values[_idx]} for _idx in range(len(all_content_ftrs))}

        if mode == "all":
            ftr_idx = list(range(3))
        elif mode == "image":
            ftr_idx = list(range(1))
        elif mode == "text":
            ftr_idx = list(range(1, 3))

        ftr_validation_info = {all_content_ftrs[_idx]: _all_ftr_validation_info[all_content_ftrs[_idx]] for _idx in ftr_idx}

        def content_data_reconstructor(json_data: dict):
            json_data = deepcopy(json_data)

            for key, content_dict in json_data.items():
                for ftr, ftr_info in ftr_validation_info.items():
                    if content_dict.get(ftr) is None:
                        json_data[key][ftr] = ftr_info["default_val"]

            return json_data

        def content_data_validator(content_data: dict):
            for content_dict in content_data.values():
                if not isinstance(content_dict, dict):
                    raise TypeError("Unexpected type found on content data!")

                for ftr_name, ftr_info in ftr_validation_info.items():
                    _type = ftr_info["dtype"]
                    if not isinstance(content_dict[ftr_name], _type):
                        raise TypeError(f"Unexpected type found on content {ftr_name} data! Expected {_type}, got {type(content_dict[ftr_name])}")

                if "para" in ftr_validation_info.keys() and not cls.__check_only_1level_iterables(content_dict["para"]):
                    raise ValueError("Found iterable in the 'paragraph' data!")

                if "images" in ftr_validation_info.keys() and not all([isinstance(image_data, list) for image_data in content_dict["images"]]):
                    raise ValueError("Found non-list in the 'images' data!")

                if "images" in ftr_validation_info.keys() and not all([len(image_data) == 2 for image_data in content_dict["images"]]):
                    raise ValueError("Found non-paired tuples in the 'images' data!")

                if "images" in ftr_validation_info.keys() and not all([cls.__check_only_1level_iterables(image_data) for image_data in content_dict["images"]]):
                    raise ValueError("Found iterable in the 'images' individual data!")

        def m3ls_content_data_post_process(content_data: dict) -> Dict:
            output_json = {}
            for _ftr in ftr_validation_info.keys():
                output_data = []
                for value in content_data.values():
                    output_data.append(value[_ftr])
                output_json[_ftr] = output_data

            # post process each features
            if "para" in ftr_validation_info.keys():
                paragraphs = []
                for section_data in output_json.pop("para"):
                    paragraphs.append("".join([val for val in section_data if val.strip() != ""]))
                output_json["paragraphs"] = paragraphs

            if "images" in ftr_validation_info.keys():
                list_image_paths = []
                list_captions = []
                for sectioned_data in output_json.pop("images"):
                    for val in sectioned_data:
                        list_image_paths.append(val[0])
                        list_captions.append("" if val[1] is None else val[1].strip())
                output_json["image_paths"] = list_image_paths
                output_json["captions"] = list_captions

            if "subheading" in ftr_validation_info.keys():
                output_json["section_headers"] = output_json.pop("subheading")

            return output_json

        content_data = content_data_reconstructor(json_content_data)

        content_data_validator(content_data)

        content_data = m3ls_content_data_post_process(content_data)

        return content_data

    @staticmethod
    def __m3ls_filter_image_and_captions_data(image_data: list, captions_data: list, base_image_folder: str, all_images: list, both_exists: bool = False) -> Tuple[List, List]:
        image_path, captions = [], []

        if len(captions_data) != len(image_data):
            raise ValueError("Not a 1-1 mapping of image-captions!")

        for idx, img_path in enumerate(image_data):
            if img_path in all_images:
                if both_exists and captions_data[idx] == "":
                    continue
                image_path.append(os.path.join(base_image_folder, img_path))
                captions.append(captions_data[idx])

        return image_path, captions
