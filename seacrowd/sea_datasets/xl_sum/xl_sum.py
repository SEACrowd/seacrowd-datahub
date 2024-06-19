"""
This new update refers to the this HF dataloader script
https://huggingface.co/datasets/csebuetnlp/xlsum/blob/main/xlsum.py
while conforming to SEACrowd schema
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import json
import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import TASK_TO_SCHEMA, Licenses, Tasks


_CITATION = """\
@inproceedings{hasan2021xl,
  title={XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages},
  author={Hasan, Tahmid and Bhattacharjee, Abhik and Islam, Md Saiful and Mubasshir, Kazi and Li, Yuan-Fang and Kang, Yong-Bin and Rahman, M Sohel and Shahriyar, Rifat},
  booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  pages={4693--4703},
  year={2021}
}
"""

_LOCAL = False
_LANGUAGES = ["ind", "mya", "tha", "vie", "eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LANG_TO_DATASOURCE_LANG = {
    "ind": "indonesian",
    "mya": "burmese",
    "vie": "vietnamese",
    "tha": "thai"}

_DATASETNAME = "xl_sum"

_DESCRIPTION = """\
XL-Sum, a comprehensive and diverse dataset comprising 1 million professionally annotated article-summary pairs from BBC, was extracted using a set of carefully designed heuristics.
The dataset covers 44 languages ranging from low to high-resource, including 4 indigenous languages spoken in Southeast Asia region.
"""

_HOMEPAGE = "https://github.com/csebuetnlp/xl-sum"

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_URLS = "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/{}_XLSum_v{}.tar.bz2"

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "2.0.0"

_SEACROWD_VERSION = "2024.06.20"


def construct_configs_on_langs() -> List[SEACrowdConfig]:
    """
    The function `construct_configs` constructs a list of SEACrowdConfig objects based on `_LANGUAGES` var, and returns the list.

    output:
        a list of `SEACrowdConfig` objects based on instantiated init variables
    """

    # set output var
    config_list = []

    # construct zipped arg for config instantiation
    CONFIG_SUFFIXES_FOR_TASK = [TASK_TO_SCHEMA.get(task).lower() for task in _SUPPORTED_TASKS]
    TASKS_AND_CONFIG_SUFFIX_PAIRS = list(zip(_SUPPORTED_TASKS, CONFIG_SUFFIXES_FOR_TASK))

    # implement source schema
    version, config_name_prefix = _SOURCE_VERSION, "source"
    config_list += [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{_LANG}_{config_name_prefix}",
            version=datasets.Version(version),
            description=f"{_DATASETNAME} {config_name_prefix} schema for language code {_LANG}",
            schema=f"{config_name_prefix}",
            subset_id=_LANG,
        )
        #skip english lang
        for _LANG in _LANGUAGES if _LANG != "eng"
    ]

    # implement SEACrowd schema
    version, config_name_prefix = _SEACROWD_VERSION, "seacrowd"
    for task_obj, config_name_suffix in TASKS_AND_CONFIG_SUFFIX_PAIRS:
        config_list += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{_LANG}_{config_name_prefix}_{config_name_suffix}",
                version=datasets.Version(version),
                description=f"{_DATASETNAME} {config_name_prefix} schema for {task_obj.name} and language code {_LANG}",
                schema=f"{config_name_prefix}_{config_name_suffix}",
                subset_id=_LANG,
            )
            #skip english lang
            for _LANG in _LANGUAGES if _LANG != "eng"
        ]
    return config_list


class XLSum(datasets.GeneratorBasedBuilder):
    """XL-Sum is a large-scale multilingual summarization dataset that covers 45 languages including Indonesian text summarization."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = construct_configs_on_langs()

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
               {
                    "id": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "summary": datasets.Value("string")
               }
            )
        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        lang = _LANG_TO_DATASOURCE_LANG[self.config.subset_id]
        url = _URLS.format(lang, self.SOURCE_VERSION.version_str[:-2])

        data_dir = dl_manager.download_and_extract(url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_train.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_test.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, lang + "_val.jsonl"),
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:

        if self.config.schema == "source":
            with open(filepath, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    ex = {
                        "id": data["id"],
                        "url": data["url"],
                        "title": data["title"],
                        "text": data["text"],
                        "summary": data["summary"],
                    }
                    yield data["id"], ex

        elif self.config.schema == "seacrowd_t2t":
            # the title is unused for this schema
            with open(filepath, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    ex = {
                        "id": data["id"],
                        "text_1": data["text"],
                        "text_2": data["summary"],
                        "text_1_name": "text",
                        "text_2_name": "summary"
                    }
                    yield data["id"], ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
