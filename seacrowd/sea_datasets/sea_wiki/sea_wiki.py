import re
import json

from typing import Dict, List, Tuple

import datasets
from datasets import load_dataset
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@ONLINE{wikidump,
    author = "Wikimedia Foundation",
    title  = "Wikimedia Downloads",
    url    = "https://dumps.wikimedia.org"}
@ONLINE{wikipedia-hf,
    title  = "Huggingface Wikipedia Dataset",
    url    = "https://huggingface.co/datasets/wikipedia"}
@ONLINE{wikipedia-hf,
    title  = "Huggingface SEA Wikipedia Dataset",
    url    = "https://huggingface.co/datasets/sabilmakbar/sea_wiki"}
"""

logger = datasets.logging.get_logger(__name__)


with open(DownloadManager().download_and_extract("seacrowd/sea_datasets/sea_wiki/lang_config.json"), "r") as f:
    _LANG_CONFIG = json.load(f)

_LOCAL = False
_LANGUAGES = list(_LANG_CONFIG.keys())

_DATASETNAME = "sea_wiki"
_DESCRIPTION = """\
	SEA Lang & Local Langs Wikipedia Archives, dumped from WIkipedia HF and processed by boilerplate removal.
    This dataset consists of URL of referred Wikipedia Article, its Title, and its Text Data (Article Contents).
"""

_HOMEPAGE = "https://huggingface.co/datasets/sabilmakbar/sea_wiki"
_LICENSE = Licenses.CC_BY_SA_4_0.value

#url won't be used since it will implement load_dataset method on HF URL provided
_URL = "https://huggingface.co/datasets/sabilmakbar/sea_wiki"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING, Tasks.SUMMARIZATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

IDENTIFIER_FOR_TASKS = [re.sub("\W", "_", task_name.value.lower()) for task_name in _SUPPORTED_TASKS]

class SEAWikiDataset(datasets.GeneratorBasedBuilder):
    """SEA Wiki dataset from https://huggingface.co/datasets/sabilmakbar/sea_wiki"""

    BUILDER_CONFIGS = [
        #source schema self-supervised pretraining (task attribute value: SSP -> identifier: ssp)
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source_{IDENTIFIER_FOR_TASKS[0]}_{LANG}",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema for {_SUPPORTED_TASKS[0].name}",
            schema=f"source_{IDENTIFIER_FOR_TASKS[0]}",
            subset_id=LANG) for LANG in _LANGUAGES
    ] + [
        #seacrowd schema self-supervised pretraining (task attribute value: SSP -> identifier: ssp)
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{IDENTIFIER_FOR_TASKS[0]}_{LANG}",
            version=datasets.Version(_SEACROWD_VERSION), 
            description=f"{_DATASETNAME} SEACrowd schema for {_SUPPORTED_TASKS[0].name}",
            schema=f"seacrowd_{IDENTIFIER_FOR_TASKS[0]}",
            subset_id=LANG) for LANG in _LANGUAGES
    ] + [
        #source schema article summarization (task attribute value: SUM -> identifier: sum)
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source_{IDENTIFIER_FOR_TASKS[1]}_{LANG}",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema for {_SUPPORTED_TASKS[1].name}", 
            schema=f"source_{IDENTIFIER_FOR_TASKS[1]}",
            subset_id=LANG) for LANG in _LANGUAGES
    ] + [
        #seacrowd schema article summarization (task attribute value: SUM -> identifier: sum)
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{IDENTIFIER_FOR_TASKS[1]}_{LANG}",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema for {_SUPPORTED_TASKS[1].name}",
            schema=f"seacrowd_{IDENTIFIER_FOR_TASKS[1]}",
            subset_id=LANG) for LANG in _LANGUAGES
    ]

    def _info(self) -> datasets.DatasetInfo:
        _config_schema_name = self.config.schema
        logger.info(f"Received schema name: {self.config.schema}")
        # self supervised training schema
        if IDENTIFIER_FOR_TASKS[0] in _config_schema_name:
            if "source" in _config_schema_name:
                features = datasets.Features(
                    {
                        "url": datasets.Value("string"),
                        "text": datasets.Value("string")
                    }
                )

            elif "seacrowd" in _config_schema_name:
                features = schemas.ssp_features

            else:
                raise ValueError(f"Unexpected schema received! {_config_schema_name}")

        elif IDENTIFIER_FOR_TASKS[1] in _config_schema_name:
            if "source" in _config_schema_name:
                features = datasets.Features(
                    {
                        "url": datasets.Value("string"),
                        "title": datasets.Value("string"), 
                        "text": datasets.Value("string")
                    }
                )

            elif "seacrowd" in _config_schema_name:
                features = schemas.text2text_features

            else:
                raise ValueError(f"Unexpected schema received! {_config_schema_name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        # args of dl_manager is useless since this data loader will wrap the hf `load_dataset` from given _URL
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN)
        ]


    def _load_hf_data_from_remote(self):
        #construct remote_hf_reference by the last 2 of string-spliited of "/"
        _remote_hf_reference = "/".join(_URL.split("/")[-2:])
        _lang_args = _LANG_CONFIG[self.config.subset_id]["source_subset"]
        _split = "train"

        print(f"\nLoading dataset from remote HF {_remote_hf_reference} with seacrowd lang args of {self.config.subset_id} and source lang args of {_lang_args} and split args of {_split}")
        _hf_dataset_source = load_dataset(_remote_hf_reference, lang=_lang_args, split=_split)

        return _hf_dataset_source


    def _generate_examples(self) -> Tuple[int, Dict]:

        _config_schema_name = self.config.schema
        loaded_data = self._load_hf_data_from_remote()

        # iterate over datapoints and arrange hf dataset schema in source to match w/ config args:
        for id_, _data in enumerate(loaded_data):
            if "source" in _config_schema_name:
                yield id_, {colname: _data[colname] for colname in self.info.features}
        
            # for ssp schema
            elif "seacrowd" in _config_schema_name and IDENTIFIER_FOR_TASKS[0] in _config_schema_name:
                yield id_, {"id": id_, "text": _data["text"]}

            # for summary schema
            elif "seacrowd" in _config_schema_name and IDENTIFIER_FOR_TASKS[1] in _config_schema_name:
                yield id_, {
                    "id": id_,
                    "text_1": _data["text"],
                    "text_2": _data["title"],
                    "text_1_name": "text_to_be_summarized",
                    "text_2_name": "title_of_text"}

            else:
                raise ValueError(f"Received unexpected config schema of {_config_schema_name}!")

if __name__ == "__main__":
    datasets.load_dataset(__file__)
