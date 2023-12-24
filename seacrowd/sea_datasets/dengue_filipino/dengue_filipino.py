import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@INPROCEEDINGS{8459963,
      author={E. D. {Livelo} and C. {Cheng}},
      booktitle={2018 IEEE International Conference on Agents (ICA)},
      title={Intelligent Dengue Infoveillance Using Gated Recurrent Neural Learning and Cross-Label Frequencies},
      year={2018},
      volume={},
      number={},
      pages={2-7},
      doi={10.1109/AGENTS.2018.8459963}}
    }
"""

_LANGUAGES = ["fil", "tgl"]

_DATASETNAME = "dengue_filipino"

_DESCRIPTION = """\
Benchmark dataset for low-resource multi-label classification, with 4,015 training, 500 testing, and 500 validation examples, each labeled as part of five classes. Each sample can be a part of multiple classes. Collected as tweets.
"""

_HOMEPAGE = "https://github.com/jcblaisecruz02/Filipino-Text-Benchmarks"

_LICENSE = Licenses.GPL_3_0.value

_SUPPORTED_TASKS = [] # TODO: What's the appropriate task? Seems like we need to add a general text_multi_features task const

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

class DengueFilipinoDataset(datasets.GeneratorBasedBuilder):
    """Dengue Dataset Low-Resource Multi-label Text Classification Dataset in Filipino"""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd",
            subset_id=f"{_DATASETNAME}",
        )
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "absent": datasets.features.ClassLabel(names=["0", "1"]),
                    "dengue": datasets.features.ClassLabel(names=["0", "1"]),
                    "health": datasets.features.ClassLabel(names=["0", "1"]),
                    "mosquito": datasets.features.ClassLabel(names=["0", "1"]),
                    "sick": datasets.features.ClassLabel(names=["0", "1"]),
                }
            )
        elif self.config.schema == "seacrowd_qa":
            features = schemas.text_multi_features(["absent", "dengue", "health", "mosquito", "sick"])

        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION
            )
        ]

    def _generate_examples(self) -> Tuple[int, Dict]:
        pass # TODO