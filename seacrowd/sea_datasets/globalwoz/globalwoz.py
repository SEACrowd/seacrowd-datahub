import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import itertools

from seacrowd.utils import schemas
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

_DATASETNAME = "globalwoz"

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

_SEACROWD_VERSION = "2024.06.20"


def seacrowd_config_constructor(dial_type, lang, schema, version):
    if dial_type not in ["EandF", "FandE", "FandF"]:
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
        seacrowd_config_constructor(tod_format, lang, schema, _SOURCE_VERSION if schema == "source" else _SEACROWD_VERSION) for tod_format, lang, schema in itertools.product(("EandF", "FandE", "FandF"), ("id", "th", "vi"), ("source", "seacrowd_tod"))
    ]

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

        elif self.config.schema == "seacrowd_tod":
            features = schemas.tod_features
        else:
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

        type_and_lang = {"dial_type": self.config.subset_id.split("_")[1].replace("and", "&"), "lang": self.config.subset_id.split("_")[2]}  # globalwoz_{dial_type}_{lang}

        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        if not os.path.exists(os.path.join(data_dir, f"{type_and_lang['dial_type']}_{type_and_lang['lang']}.json")):
            raise FileNotFoundError()

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

        elif self.config.schema == "seacrowd_tod":
            for idx, tod_dialogue in enumerate(data.values()):
                example = {}
                example["dialogue_idx"] = idx

                dialogue = []
                # NOTE: the dialogue always started with `user` as first utterance
                for turn, i in enumerate(range(0, len(tod_dialogue["log"]) + 2, 2)):
                    dial = {}
                    dial["turn_idx"] = turn

                    # system_utterance properties
                    dial["system_utterance"] = ""
                    dial["system_acts"] = []
                    if turn != 0:
                        dial["system_utterance"] = tod_dialogue["log"][i - 1]["text"]
                    if i < len(tod_dialogue["log"]):
                        # NOTE: "system_acts will be populated with the `dialog_act` from the user utterance in the original dataset, as our schema dictates
                        # that `system_acts` should represent the system's intended actions based on the user's utterance."
                        for acts in tod_dialogue["log"][i]["dialog_act"].values():
                            for act in acts:
                                dial["system_acts"].append([act[0]])

                    # user_utterance properties
                    dial["turn_label"] = []  # left as an empty array
                    dial["belief_state"] = []
                    if i == len(tod_dialogue["log"]):
                        # case if turn_idx > len(dialogue) --> add dummy user_utterance
                        dial["user_utterance"] = ""
                    else:
                        dial["user_utterance"] = tod_dialogue["log"][i]["text"]
                        # NOTE: "the belief_state will be populated with the `span_info` from the user utterance in the original dataset, as our schema dictates
                        # that `belief_state` should represent the system's belief state based on the user's utterance."
                        for span in tod_dialogue["log"][i]["span_info"]:
                            if span[0].split("-")[1] == "request":  # Request action
                                dial["belief_state"].append({"slots": [["slot", span[1]]], "act": "request"})
                            else:
                                dial["belief_state"].append({"slots": [[span[1], span[2]]], "act": span[0].split("-")[1]})

                    # append to dialogue
                    dialogue.append(dial)

                example["dialogue"] = dialogue

                yield example["dialogue_idx"], example
