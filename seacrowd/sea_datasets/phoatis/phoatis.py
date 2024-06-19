from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@article{dao2021intent,
      title={Intent Detection and Slot Filling for Vietnamese},
      author={Mai Hoang Dao and Thinh Hung Truong and Dat Quoc Nguyen},
      year={2021},
      eprint={2104.02021},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "phoatis"

_DESCRIPTION = """\
This is first public intent detection and slot filling dataset for Vietnamese. The data contains 5871 English utterances from ATIS that are manually translated by professional translators into Vietnamese.
"""

_HOMEPAGE = "https://github.com/VinAIResearch/JointIDSF/"

_LICENSE = Licenses.UNKNOWN.value

_URLS = {
    _DATASETNAME: {
        "syllable": {
            "syllable_train": [
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/train/seq.in",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/train/seq.out",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/train/label",
            ],
            "syllable_dev": [
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/dev/seq.in",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/dev/seq.out",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/dev/label",
            ],
            "syllable_test": [
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/test/seq.in",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/test/seq.out",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/syllable-level/test/label",
            ],
        },
        "word": {
            "word_train": [
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/train/seq.in",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/train/seq.out",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/train/label",
            ],
            "word_dev": [
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/dev/seq.in",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/dev/seq.out",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/dev/label",
            ],
            "word_test": [
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/test/seq.in",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/test/seq.out",
                "https://raw.githubusercontent.com/VinAIResearch/JointIDSF/main/PhoATIS/word-level/test/label",
            ],
        },
    }
}

_LOCAL = False
_LANGUAGES = ["vie"]

_SUPPORTED_TASKS = [Tasks.INTENT_CLASSIFICATION, Tasks.SLOT_FILLING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


def config_constructor_intent_cls(schema: str, version: str, phoatis_subset: str = "syllable") -> SEACrowdConfig:
    assert phoatis_subset == "syllable" or phoatis_subset == "word"

    return SEACrowdConfig(
        name="phoatis_intent_cls_{phoatis_subset}_{schema}".format(phoatis_subset=phoatis_subset.lower(), schema=schema),
        version=version,
        description="PhoATIS Intent Classification: {subset} {schema} schema".format(subset=phoatis_subset, schema=schema),
        schema=schema,
        subset_id=phoatis_subset,
    )


def config_constructor_slot_filling(schema: str, version: str, phoatis_subset: str = "syllable") -> SEACrowdConfig:
    assert phoatis_subset == "syllable" or phoatis_subset == "word"

    return SEACrowdConfig(
        name="phoatis_slot_filling_{phoatis_subset}_{schema}".format(phoatis_subset=phoatis_subset.lower(), schema=schema),
        version=version,
        description="PhoATIS Slot Filling: {subset} {schema} schema".format(subset=phoatis_subset, schema=schema),
        schema=schema,
        subset_id=phoatis_subset,
    )


class PhoATIS(datasets.GeneratorBasedBuilder):
    """This is first public intent detection and slot filling dataset for Vietnamese. The data contains 5871 English utterances from ATIS that are manually translated by professional translators into Vietnamese."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    # BUILDER_CONFIGS = [config_constructor_intent_cls("source", _SOURCE_VERSION, subset) for subset in ["syllable", "word"]]
    BUILDER_CONFIGS = []
    BUILDER_CONFIGS.extend([config_constructor_intent_cls("seacrowd_text", _SEACROWD_VERSION, subset) for subset in ["syllable", "word"]])
    # BUILDER_CONFIGS.extend([config_constructor_slot_filling("source", _SOURCE_VERSION, subset) for subset in ["syllable", "word"]])
    BUILDER_CONFIGS.extend([config_constructor_slot_filling("seacrowd_seq_label", _SEACROWD_VERSION, subset) for subset in ["syllable", "word"]])

    BUILDER_CONFIGS.extend(
        [  # Default config
            SEACrowdConfig(
                name="phoatis_source",
                version=SOURCE_VERSION,
                description="PhoATIS source schema (Syllable version)",
                schema="source",
                subset_id="syllable",
            ),
            SEACrowdConfig(
                name="phoatis_intent_cls_seacrowd_text",
                version=SEACROWD_VERSION,
                description="PhoATIS Intent Classification SEACrowd schema (Syllable version)",
                schema="seacrowd_text",
                subset_id="syllable",
            ),
            SEACrowdConfig(
                name="phoatis_slot_filling_seacrowd_seq_label",
                version=SEACROWD_VERSION,
                description="PhoATIS Slot Filling SEACrowd schema (Syllable version)",
                schema="seacrowd_seq_label",
                subset_id="syllable",
            ),
        ]
    )

    DEFAULT_CONFIG_NAME = "phoatis_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "intent_label": datasets.Value("string"),
                    "slot_label": datasets.Sequence(datasets.Value("string")),
                }
            )

        elif self.config.schema == "seacrowd_text":
            with open("./seacrowd/sea_datasets/phoatis/intent_label.txt", "r+", encoding="utf8") as fw:
                intent_label = fw.read()
                intent_label = intent_label.split("\n")
            features = schemas.text_features(intent_label)

        elif self.config.schema == "seacrowd_seq_label":
            with open("./seacrowd/sea_datasets/phoatis/slot_label.txt", "r+", encoding="utf8") as fw:
                slot_label = fw.read()
                slot_label = slot_label.split("\n")
            features = schemas.seq_label_features(slot_label)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        schema = self.config.subset_id
        urls = _URLS[_DATASETNAME][schema]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir[f"{schema}_train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir[f"{schema}_test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir[f"{schema}_dev"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        with open(filepath[0], "r+", encoding="utf8") as fw:
            data_input = fw.read()
            data_input = data_input.split("\n")
        with open(filepath[1], "r+", encoding="utf8") as fw:
            data_slot = fw.read()
            data_slot = data_slot.split("\n")
        with open(filepath[2], "r+", encoding="utf8") as fw:
            data_intent = fw.read()
            data_intent = data_intent.split("\n")

        if self.config.schema == "source":
            for idx, text in enumerate(data_input):
                example = {}
                example["id"] = str(idx)
                example["text"] = text
                example["intent_label"] = data_intent[idx]
                example["slot_label"] = data_slot[idx].split()
                yield example["id"], example

        elif self.config.schema == "seacrowd_text":
            for idx, text in enumerate(data_input):
                example = {}
                example["id"] = str(idx)
                example["text"] = text
                example["label"] = data_intent[idx]
                yield example["id"], example

        elif self.config.schema == "seacrowd_seq_label":
            for idx, text in enumerate(data_input):
                example = {}
                example["id"] = str(idx)
                example["tokens"] = text.split()
                example["labels"] = data_slot[idx].split()
                yield example["id"], example
