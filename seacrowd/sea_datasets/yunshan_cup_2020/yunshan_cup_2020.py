from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{DBLP:journals/corr/abs-2204-02658,
 author    = {Yingwen Fu and
              Jinyi Chen and
              Nankai Lin and
              Xixuan Huang and
              Xin Ying Qiu and
              Shengyi Jiang},
 title     = {Yunshan Cup 2020: Overview of the Part-of-Speech Tagging Task for
              Low-resourced Languages},
 journal   = {CoRR},
 volume    = {abs/2204.02658},
 year      = {2022},
 url       = {https://doi.org/10.48550/arXiv.2204.02658},
 doi       = {10.48550/arXiv.2204.02658},
 eprinttype = {arXiv},
 eprint    = {2204.02658},
 timestamp = {Tue, 12 Apr 2022 18:42:14 +0200},
 biburl    = {https://dblp.org/rec/journals/corr/abs-2204-02658.bib},
 bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DATASETNAME = "yunshan_cup_2020"

_DESCRIPTION = """\
Lao POS dataset containing 11,000 sentences was released as part of Yunshan-Cup-2020 evaluation track.
"""

_HOMEPAGE = "https://github.com/GKLMIP/Yunshan-Cup-2020"

_LOCAL = False
_LANGUAGES = ["lao"]

_LICENSE = Licenses.UNKNOWN.value  # example: Licenses.MIT.value, Licenses.CC_BY_NC_SA_4_0.value, Licenses.UNLICENSE.value, Licenses.UNKNOWN.value

_URLS = {
    "train": "https://raw.githubusercontent.com/GKLMIP/Yunshan-Cup-2020/main/train.txt",
    "val": "https://raw.githubusercontent.com/GKLMIP/Yunshan-Cup-2020/main/dev.txt",
    "test": "https://raw.githubusercontent.com/GKLMIP/Yunshan-Cup-2020/main/test.txt",
}
_SUPPORTED_TASKS = [Tasks.POS_TAGGING]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]
_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class YunshanCup2020Dataset(datasets.GeneratorBasedBuilder):
    """Lao POS dataset containing 11,000 sentences was released as part of Yunshan-Cup-2020 evaluation track."""

    class_labels = ["IAC", "COJ", "ONM", "PRE", "PRS", "V", "DBQ", "IBQ", "FIX", "N", "ADJ", "DMN", "IAQ", "CLF", "PRA", "DAN", "NEG", "NTR", "REL", "PVA", "TTL", "DAQ", "PRN", "ADV", "PUNCT", "CNM"]

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="yunshan_cup_2020 source schema",
            schema="source",
            subset_id="yunshan_cup_2020",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="yunshan_cup_2020 SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id="yunshan_cup_2020",
        ),
    ]

    DEFAULT_CONFIG_NAME = "yunshan_cup_2020_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    "pos_tags": [datasets.Value("string")],
                }
            )
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(self.class_labels)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        path_dict = dl_manager.download_and_extract(_URLS)
        train_path, val_path, test_path = path_dict["train"], path_dict["val"], path_dict["test"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_path
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": val_path,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        df = load_postagging_data(filepath)
        if self.config.schema == "source":
            for i, row in enumerate(df):
                ex = {
                    "index": str(i),
                    "tokens": row["sentence"],
                    "pos_tags": row["label"],
                }
                yield i, ex

        elif self.config.schema == "seacrowd_seq_label":
            for i, row in enumerate(df):
                ex = {
                    "id": str(i),
                    "tokens": row["sentence"],
                    "labels": row["label"],
                }
                yield i, ex


def load_postagging_data(file_path):
    data = open(file_path, "r").readlines()
    dataset = []
    sentence, seq_label = [], []
    for line in data:
        if len(line.strip()) > 0:
            token, label = " ", ""
            if len(line.strip().split(" ")) < 2:
                label = line.strip()
            else:
                token, label = line[:-1].split(" ")
            sentence.append(token)
            seq_label.append(label)
        else:
            dataset.append({"sentence": sentence, "label": seq_label})
            sentence = []
            seq_label = []
    return dataset
