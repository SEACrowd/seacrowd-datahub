import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@inproceedings{ding-etal-2022-globalwoz,
    title = "{G}lobal{W}o{Z}: Globalizing {M}ulti{W}o{Z} to Develop Multilingual Task-Oriented Dialogue Systems",
    author = "Ding, Bosheng  and
      Hu, Junjie  and
      Bing, Lidong  and
      Aljunied, Mahani  and
      Joty, Shafiq  and
      Si, Luo  and
      Miao, Chunyan",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
}
"""

_DATASETNAME = "[globalwoz]"

_DESCRIPTION = """\
This is the data of the paper “GlobalWoZ: Globalizing MultiWoZ to Develop Multilingual Task-Oriented Dialogue Systems” accepted by ACL 2022. The dataset contains several sub-datasets in 20 languages and 3 schemes (F&E, E&F, F&F), including Indonesian (id), Thai (th), and Vietnamese (vi) language. The method is based on translating dialogue templates and filling them with local entities in the target language countries.
"""


_HOMEPAGE = "https://github.com/bosheng2020/globalwoz"


_LANGUAGES = ["ind", "tha", "vie"]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = True

_URLS = {}

_SUPPORTED_TASKS = [Tasks.E2E_TASK_ORIENTED_DIALOGUE]

_SOURCE_VERSION = "2.0.0"

_SEACROWD_VERSION = "1.0.0"


def seacrowd_config_constructor(dial_type, lang, schema, version):
    if dial_type not in ["E&F", "F&E", "F&F"]:
        raise ValueError(f"Invalid dialogue type {dial_type}")

    if lang == "":
        raise ValueError(f"Invalid lang {lang}")

    if schema not in ["source", "seacrowd_tod"]:
        raise ValueError(f"Invalid schema: {schema}")

    return SEACrowdConfig(
        name="globalwoz_{dial_type}_{lang}_{schema}".format(dial_type=dial_type, lang=lang, schema=schema),
        version=datasets.Version(version),
        description="GlobalWoZ schema for {schema}: {dial_type}_{lang}".format(schema=schema, dial_type=dial_type, lang=lang),
        schema=schema,
        subset_id="globalwoz_{dial_type}_{lang}".format(dial_type=dial_type, lang=lang),
    )


class GlobalWoZ(datasets.GeneratorBasedBuilder):
    """This is the data of the paper “GlobalWoZ: Globalizing MultiWoZ to Develop Multilingual Task-Oriented Dialogue Systems” accepted by ACL 2022.
    The dataset contains several sub-datasets in 20 languages and 3 schemes (F&E, E&F, F&F), including Indonesian (id), Thai (th),
    and Vietnamese (vi) language. The method is based on translating dialogue templates and filling them with local entities in the target language countries.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        seacrowd_config_constructor("E&F", "id", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("E&F", "th", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("E&F", "vi", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("F&E", "id", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("F&E", "th", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("F&E", "vi", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("F&F", "id", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("F&F", "th", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("F&F", "vi", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("E&F", "id", "seacrowd_tod", _SEACROWD_VERSION),
        seacrowd_config_constructor("E&F", "th", "seacrowd_tod", _SEACROWD_VERSION),
        seacrowd_config_constructor("E&F", "vi", "seacrowd_tod", _SEACROWD_VERSION),
        seacrowd_config_constructor("F&E", "id", "seacrowd_tod", _SEACROWD_VERSION),
        seacrowd_config_constructor("F&E", "th", "seacrowd_tod", _SEACROWD_VERSION),
        seacrowd_config_constructor("F&E", "vi", "seacrowd_tod", _SEACROWD_VERSION),
        seacrowd_config_constructor("F&F", "id", "seacrowd_tod", _SEACROWD_VERSION),
        seacrowd_config_constructor("F&F", "th", "seacrowd_tod", _SEACROWD_VERSION),
        seacrowd_config_constructor("F&F", "vi", "seacrowd_tod", _SEACROWD_VERSION),
    ]

    DEFAULT_CONFIG_NAME = "globalwoz_E&F_id_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "goal": {
                        "attraction": datasets.Value("string"),
                        "hospital": datasets.Value("string"),
                        "hotel": datasets.Value("string"),
                        "police": datasets.Value("string"),
                        "restaurant": datasets.Value("string"),
                        "taxi": datasets.Value("string"),
                        "train": datasets.Value("string"),
                    },
                    "log": [
                        {
                            "dialog_act": datasets.Value("string"),
                            "metadata": datasets.Value("string"),
                            "span_info": [[datasets.Value("string")]],
                            "text": datasets.Value("string"),
                        }
                    ],
                }
            )

        elif self.config.schema == "seacrowd_[seacrowdschema_name]":
            # e.g. features = schemas.kb_features
            # TODO: Choose your seacrowd schema here
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        _split_generators = []

        type_and_lang = {"dial_type": self.config.subset_id.split("_")[1], "lang": self.config.subset_id.split("_")[2]}  # globalwoz_{dial_type}_{lang}

        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    # "filepath": data_dir + f"_{type_and_lang['dial_type']}_{type_and_lang['lang']}.json",
                    "filepath": os.path.join(data_dir, f"{type_and_lang['dial_type']}_{type_and_lang['lang']}.json"),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # For local datasets you will have access to self.config.data_dir and self.config.data_files
        with open(filepath, "r+", encoding="utf8") as fw:
            data = json.load(fw)
        # {
        #     "id": datasets.Value("string"),
        #     "goal": {
        #         "attraction": datasets.Value("string"),
        #         "hospital": datasets.Value("string"),
        #         "hotel": datasets.Value("string"),
        #         "police": datasets.Value("string"),
        #         "restaurant": datasets.Value("string"),
        #         "taxi": datasets.Value("string"),
        #         "train": datasets.Value("string"),
        #     },
        #     "log": [
        #         {
        #             "dialog_act": datasets.Value("string"),
        #             "metadata": datasets.Value("string"),
        #             "span_info": [[datasets.Value("string")]],
        #             "text": datasets.Value("string"),
        #         }
        #     ]
        # }
        if self.config.schema == "source":
            for idx, tod_dialogue in enumerate(data.values()):
                example = {}
                example["id"] = str(idx)
                example["goal"] = {}

                for goal_key in ["attraction", "hospital", "hotel", "police", "restaurant", "taxi", "train"]:
                    example["goal"][goal_key] = json.dumps(tod_dialogue["goal"][goal_key])
                example["log"] = []

                for dial_log in tod_dialogue["log"]:
                    dial = {}
                    dial["dialog_act"] = json.dumps(dial_log["dialog_act"])
                    dial["metadata"] = json.dumps(dial_log["metadata"])
                    for i in range(len(dial_log["span_info"])):
                        for j in range(len(dial_log["span_info"][i])):
                            dial_log["span_info"][i][j] = str(dial_log["span_info"][i][j])  # casting to str
                    dial["span_info"] = [[str(span)] if isinstance(span, str) else span for span in dial_log["span_info"]]
                    dial["text"] = dial_log["text"]

                    example["log"].append(dial)

                yield example["id"], example

        elif self.config.schema == "seacrowd_[seacrowd_schema_name]":
            # TODO: yield (key, example) tuples in the seacrowd schema
            for key, example in thing:
                yield key, example


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
