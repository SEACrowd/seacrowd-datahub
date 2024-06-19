import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@article{kautsar2023indotod,
    author={Kautsar, Muhammad Dehan Al and Nurdini, Rahmah Khoirussyifa' and Cahyawijaya, Samuel and Winata, Genta Indra and Purwarianti, Ayu},
    title={IndoToD: A Multi-Domain Indonesian Benchmark For End-to-End Task-Oriented Dialogue Systems},
    journal={arXiv preprint arXiv:2311.00958},
    year={2023},
}
"""

_LANGUAGES = ["ind"]
_LOCAL = False

_DATASETNAME = "indocamrest"

_DESCRIPTION = """\
IndoCamRest is a synthetic task-oriented dialogue system dataset that translated from Cambridge Restaurant 676 (CamRest) dataset (Wen et al., 2016) into the new Indonesian parallel dataset using the translation pipeline method including the delexicalization, translation, and delexicalization.
The dataset consists of 676 dialogues in the restaurant reservation domain, with a user and an agent talking to each other to search the restaurant near the user.
It also consists of slots and dialogue acts from the user and the agent.
"""

_HOMEPAGE = "https://github.com/dehanalkautsar/IndoToD/tree/main/IndoCamRest"

_LICENSE = Licenses.CC_BY_SA_4_0.value

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/dehanalkautsar/IndoToD/main/IndoCamRest/IndoCamRest676.json",
}

_SUPPORTED_TASKS = [Tasks.E2E_TASK_ORIENTED_DIALOGUE]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class IndoCamRest(datasets.GeneratorBasedBuilder):
    """IndoToD: A Multi-Domain Indonesian Benchmark For End-to-End Task-Oriented Dialogue Systems"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="indocamrest_source",
            version=SOURCE_VERSION,
            description="IndoToD: IndoCamRest source schema",
            schema="source",
            subset_id="indocamrest",
        ),
        SEACrowdConfig(
            name="indocamrest_seacrowd_tod",
            version=SEACROWD_VERSION,
            description="IndoToD: IndoCamRest SEACrowd End-to-end Task Oriented Dialogue schema",
            schema="seacrowd_tod",
            subset_id="indocamrest",
        ),
    ]

    DEFAULT_CONFIG_NAME = "indocamrest_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "dialogue_id": datasets.Value("int32"),
                    "finished": datasets.Value("string"),
                    "goal": {"constraints": [[datasets.Value("string")]], "request-slots": [datasets.Value("string")], "text": datasets.Value("string")},
                    "dial": [
                        {
                            "turn": datasets.Value("int32"),
                            "usr": {
                                "transcript": datasets.Value("string"),
                                "delex_transcript": datasets.Value("string"),
                                "slu": [{"act": datasets.Value("string"), "slots": [[datasets.Value("string")]]}],
                            },
                            "sys": {"sent": datasets.Value("string"), "delex_sent": datasets.Value("string"), "DA": [datasets.Value("string")]},
                        }
                    ],
                }
            )
        elif self.config.schema == "seacrowd_tod":
            features = schemas.tod_features
        else:
            raise NotImplementedError(f"Schema {self.config.schema} has not been implemented")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        with open(filepath, "r+") as fw:
            data = json.loads(fw.read())

        if self.config.schema == "source":
            for idx, example in enumerate(data):
                example["index"] = str(idx)
                yield str(idx), example

        elif self.config.schema == "seacrowd_tod":
            for idx, tod_dialogue in enumerate(data):
                example = {}
                example["dialogue_idx"] = idx

                dialogue = []
                for i in range(len(tod_dialogue["dial"]) + 1):
                    dial = {}
                    dial["turn_idx"] = i

                    # system_utterance properties
                    if i == 0:
                        # case if turn_idx == 0
                        dial["system_utterance"] = ""
                        dial["system_acts"] = []
                    else:
                        dial["system_utterance"] = tod_dialogue["dial"][i - 1]["sys"]["sent"]
                        # some system_acts is either to string or list of strings,
                        # converting all to list of strings
                        dial["system_acts"] = [[act] if isinstance(act, str) else act for act in tod_dialogue["dial"][i - 1]["sys"]["DA"]]

                    # user_utterance properties
                    dial["turn_label"] = []
                    dial["belief_state"] = []
                    if i == len(tod_dialogue["dial"]):
                        # case if turn_idx > len(dialogue) --> add dummy user_utterance
                        dial["user_utterance"] = ""
                    else:
                        dial["user_utterance"] = tod_dialogue["dial"][i]["usr"]["transcript"]
                        for j in range(len(tod_dialogue["dial"][i]["usr"]["slu"])):
                            dial["belief_state"].append({"slots": tod_dialogue["dial"][i]["usr"]["slu"][j]["slots"], "act": tod_dialogue["dial"][i]["usr"]["slu"][j]["act"]})

                    # append to dialogue
                    dialogue.append(dial)
                example["dialogue"] = dialogue
                yield str(idx), example
