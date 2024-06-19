# coding=utf-8
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{Lowphansirikul2021,
    author={Lowphansirikul, Lalita
            and Polpanumas, Charin
            and Rutherford, Attapol T.
            and Nutanong, Sarana},
    title={A large English--Thai parallel corpus from the web and machine-generated text},
    journal={Language Resources and Evaluation},
    year={2021},
    month={Mar},
    day={30},
    issn={1574-0218},
    doi={10.1007/s10579-021-09536-6},
    url={https://doi.org/10.1007/s10579-021-09536-6}
"""

_DATASETNAME = "scb_mt_en_th"

_DESCRIPTION = """\
A Large English-Thai Parallel Corpus The primary objective of our work is to build a large-scale English-Thai dataset
for machine translation. We construct an English-Thai machine translation dataset with over 1 million segment pairs,
curated from various sources, namely news, Wikipedia articles, SMS messages, task-based dialogs, web-crawled data and
government documents. Methodology for gathering data, building parallel texts and removing noisy sentence pairs are
presented in a reproducible manner. We train machine translation models based on this dataset. Our models' performance
are comparable to that of Google Translation API (as of May 2020) for Thai-English and outperform Google when the Open
Parallel Corpus (OPUS) is included in the training data for both Thai-English and English-Thai translation. The dataset,
pre-trained models, and source code to reproduce our work are available for public use.

"""

_HOMEPAGE = "https://github.com/vistec-AI/thai2nmt"

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LANGUAGES = ["tha", "eng"]
_LOCAL = False

_URLS = {
    _DATASETNAME: "https://archive.org/download/scb_mt_enth_2020/data.zip",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

SEACROWD_TO_SOURCE_LANGCODE_DICT = {"eng": "en", "tha": "th"}


class ScbMtEnThDataset(datasets.GeneratorBasedBuilder):
    """
    A Large English-Thai Parallel Corpus The primary objective of our work is to build a large-scale English-Thai
    dataset for machine translation. We construct an English-Thai machine translation dataset with over 1 million
    segment pairs, curated from various sources, namely news, Wikipedia articles, SMS messages, task-based dialogs,
     web-crawled data and government documents.
     Methodology for gathering data, building parallel texts and removing noisy sentence pairs are presented in a
     reproducible manner. We train machine translation models based on this dataset. Our models' performance are
     comparable to that of Google Translation API (as of May 2020) for Thai-English and outperform Google when the Open
     Parallel Corpus (OPUS) is included in the training data for both Thai-English and English-Thai translation.
     The dataset,pre-trained models, and source code to reproduce our work are available for public use."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_tha_eng_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema: Thai to English",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_tha_eng_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema: Thai to English",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_eng_tha_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema: English to Thai",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_eng_tha_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema: English to Thai",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_tha_eng_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            language_pair = [SEACROWD_TO_SOURCE_LANGCODE_DICT[lang] for lang in self.config.name.split("_")[4:6]]
            features = datasets.Features(
                {
                    "translation": datasets.features.Translation(language_pair),
                    "subdataset": datasets.Value("string"),
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
        urls = _URLS[_DATASETNAME]
        data_path = dl_manager.download_and_extract(urls)
        data_dir = os.path.join(data_path, "data")

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_dir, "train.jsonl")}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_dir, "valid.jsonl")}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(data_dir, "test.jsonl")}),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        with open(filepath, encoding="utf-8") as f:
            if self.config.schema == "source":
                language_pair = [SEACROWD_TO_SOURCE_LANGCODE_DICT[lang] for lang in self.config.name.split("_")[4:6]]
                source, target = language_pair
                for id_, row in enumerate(f):
                    data = json.loads(row)
                    yield id_, {
                        "translation": {source: data[source], target: data[target]},
                        "subdataset": data["subdataset"],
                    }

            elif self.config.schema == "seacrowd_t2t":
                source, target = self.config.name.split("_")[4:6]
                for id_, row in enumerate(f):
                    data = json.loads(row)
                    ex = {
                        "id": str(id_),
                        "text_1": data[SEACROWD_TO_SOURCE_LANGCODE_DICT[source]],
                        "text_2": data[SEACROWD_TO_SOURCE_LANGCODE_DICT[target]],
                        "text_1_name": source,
                        "text_2_name": target,
                    }
                    yield id_, ex
