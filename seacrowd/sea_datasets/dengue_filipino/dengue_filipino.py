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

_LANGUAGES = ["fil"]

# copied from https://huggingface.co/datasets/dengue_filipino/blob/main/dengue_filipino.py
_URL = "https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/dengue/dengue_raw.zip"
_DATASETNAME = "dengue_filipino"

_DESCRIPTION = """\
Benchmark dataset for low-resource multi-label classification, with 4,015 training, 500 testing, and 500 validation examples, each labeled as part of five classes. Each sample can be a part of multiple classes. Collected as tweets.
"""

_HOMEPAGE = "https://github.com/jcblaisecruz02/Filipino-Text-Benchmarks"

_LICENSE = Licenses.UNKNOWN.value

_SUPPORTED_TASKS = [Tasks.DOMAIN_KNOWLEDGE_MULTICLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_LOCAL = False


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
            name=f"{_DATASETNAME}_seacrowd_text_multi",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema text multi",
            schema="seacrowd_text_multi",
            subset_id=f"{_DATASETNAME}",
        ),
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
        elif self.config.schema == "seacrowd_text_multi":
            features = schemas.text_multi_features(["0", "1"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, split: str) -> Tuple[int, Dict]:
        dataset = datasets.load_dataset(_DATASETNAME, split=split)
        for id, data in enumerate(dataset):
            if self.config.schema == "source":
                yield id, {
                    "text": data["text"],
                    "absent": data["absent"],
                    "dengue": data["dengue"],
                    "health": data["health"],
                    "mosquito": data["mosquito"],
                    "sick": data["sick"],
                }

            elif self.config.schema == "seacrowd_text_multi":
                yield id, {
                    "id": id,
                    "text": data["text"],
                    "labels": [
                        data["absent"],
                        data["dengue"],
                        data["health"],
                        data["mosquito"],
                        data["sick"],
                    ],
                }
