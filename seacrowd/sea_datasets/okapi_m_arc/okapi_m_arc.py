import json
import os
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{dac2023okapi,
  title={Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback},
  author={Dac Lai, Viet and Van Nguyen, Chien and Ngo, Nghia Trung and Nguyen, Thuat and Dernoncourt, Franck and Rossi, Ryan A and Nguyen, Thien Huu},
  journal={arXiv e-prints},
  pages={arXiv--2307},
  year={2023}
}

@article{Clark2018ThinkYH,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal={ArXiv},
  year={2018},
  volume={abs/1803.05457}
}
"""

_DATASETNAME = "okapi_m_arc"

_DESCRIPTION = """\
mARC is a Multilingual translation of AI2's Arc Challenge from the paper "Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback" (Lai et al., 2023).
The original ARC dataset is a multiple-choice question answering dataset of 7,787 genuine grade-school level science questions assembled to encourage research in advanced question-answering.
The dataset is partitioned into a Challenge Set and an Easy Set, where the former contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm.
We also include a corpus of over 14 million science sentences relevant to the task and an implementation of three neural baseline models for this dataset. We pose ARC as a challenge to the community.
"""


_HOMEPAGE = "https://huggingface.co/datasets/jon-tow/okapi_arc_challenge"
_LICENSE = Licenses.CC_BY_NC_4_0.value
_LOCAL = False
_LANGUAGES = ["ind", "vie"]

_LANG_MAP = {"ind": "id", "vie": "vi"}
_URLS = {
    "base_url": "https://huggingface.co/datasets/jon-tow/okapi_arc_challenge/resolve/main",
}
_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class MultilingualArc(datasets.GeneratorBasedBuilder):
    """mARC is a Multilingual translation of AI2's Arc Challenge which is a multiple-choice question answering dataset
    of 7,787 genuine grade-school level science questions assembled to encourage research in advanced question-answering"""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="okapi_m_arc_vie_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Vietnamese mARC source schema",
            schema="source",
            subset_id="okapi_m_arc_vie_source",
        ),
        SEACrowdConfig(
            name="okapi_m_arc_ind_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Indonesian mARC source schema",
            schema="source",
            subset_id="okapi_m_arc_ind_source",
        ),
        SEACrowdConfig(
            name="okapi_m_arc_vie_seacrowd_qa",
            version=datasets.Version(_SEACROWD_VERSION),
            description="Vietnamese mARC SEACrowd question answering schema",
            schema="seacrowd_qa",
            subset_id="okapi_m_arc_vie_seacrowd_qa",
        ),
        SEACrowdConfig(
            name="okapi_m_arc_ind_seacrowd_qa",
            version=datasets.Version(_SEACROWD_VERSION),
            description="Indonesian mARC SEACrowd question answering schema",
            schema="seacrowd_qa",
            subset_id="okapi_m_arc_ind_seacrowd_qa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "okapi_m_arc_ind_seacrowd_qa"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "label": datasets.Value("string"),
                        }
                    ),
                    "answerKey": datasets.Value("string"),
                }
            )
        else:
            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        lang = self.config.subset_id[: -(len(self.config.schema) + 1)].split("_")[-1]
        train_path = Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/data/{_LANG_MAP[lang]}_train.json"))
        valid_path = Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/data/{_LANG_MAP[lang]}_validation.json"))
        test_path = Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/data/{_LANG_MAP[lang]}_test.json"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": train_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": valid_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_path},
            ),
        ]

    def _generate_examples(self, filepath) -> Generator[Tuple[int, Dict], None, None]:
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        for i, d in enumerate(data):
            text_choices = []
            label_choices = []
            if "option_a" in d:
                text_choices.append(d["option_a"])
                label_choices.append("A")
            if "option_b" in d:
                text_choices.append(d["option_b"])
                label_choices.append("B")
            if "option_c" in d:
                text_choices.append(d["option_c"])
                label_choices.append("C")
            if "option_d" in d:
                text_choices.append(d["option_d"])
                label_choices.append("D")
            if "option_e" in d:
                text_choices.append(d["option_e"])
                label_choices.append("E")

            if self.config.schema == "source":
                yield i, {
                    "id": d["id"],
                    "answerKey": d["answer"],
                    "question": d["instruction"],
                    "choices": {"text": text_choices, "label": label_choices},
                }
            else:
                yield i, {
                    "id": i,
                    "question_id": d["id"],
                    "document_id": d["id"],
                    "question": d["instruction"],
                    "type": "multiple_choice",
                    "choices": [text for text in text_choices],
                    "context": None,
                    "answer": [text_choices[ord(d["answer"])-65]],
                    "meta": {}
                }
