import json
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

@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
"""

_DATASETNAME = "okapi_m_mmlu"

_DESCRIPTION = """\
mMMLU is a Multilingual translation of MMLU from the paper "Measuring Massive Multitask Language Understanding" (Hendrycks et al., 2021).
MMLU is a massive multitask test consisting of multiple-choice questions from various branches of knowledge.
The test spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn.
MMLU covers 57 tasks including elementary mathematics, US history, computer science, law, and more.
To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability.
"""

_HOMEPAGE = "https://huggingface.co/datasets/jon-tow/okapi_mmlu"
_LICENSE = Licenses.CC_BY_NC_4_0.value
_LOCAL = False
_LANGUAGES = ["ind", "vie"]

_LANG_MAP = {"ind": "id", "vie": "vi"}
_URLS = {
    "base_url": "https://huggingface.co/datasets/jon-tow/okapi_mmlu/resolve/main"
}
_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class MMLU(datasets.GeneratorBasedBuilder):
    # mMMLU is a Multilingual translation of MMLU from the paper "Measuring Massive Multitask Language Understanding" (Hendrycks et al., 2021)
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="okapi_m_mmlu_vie_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Vietnamese MMLU source schema",
            schema="source",
            subset_id="okapi_m_mmlu_vie_source",
        ),
        SEACrowdConfig(
            name="okapi_m_mmlu_ind_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Indonesian MMLU source schema",
            schema="source",
            subset_id="okapi_m_mmlu_ind_source",
        ),
        SEACrowdConfig(
            name="okapi_m_mmlu_vie_seacrowd_qa",
            version=datasets.Version(_SEACROWD_VERSION),
            description="Vietnamese MMLU SEACrowd question answering schema",
            schema="seacrowd_qa",
            subset_id="okapi_m_mmlu_vie_seacrowd_qa",
        ),
        SEACrowdConfig(
            name="okapi_m_mmlu_ind_seacrowd_qa",
            version=datasets.Version(_SEACROWD_VERSION),
            description="Indonesian MMLU SEACrowd question answering schema",
            schema="seacrowd_qa",
            subset_id="okapi_m_mmlu_ind_seacrowd_qa",
        ),
    ]

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
        dev_path = Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/data/{_LANG_MAP[lang]}_dev.json"))
        valid_path = Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/data/{_LANG_MAP[lang]}_val.json"))
        test_path = Path(dl_manager.download_and_extract(f"{_URLS['base_url']}/data/{_LANG_MAP[lang]}_test.json"))
        return [
            datasets.SplitGenerator(
                name="dev",
                gen_kwargs={"filepath": dev_path},
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

    def _generate_examples(self, filepath: str) -> Generator[Tuple[int, Dict], None, None]:
        with open(filepath, encoding="utf-8") as f:
            contents = json.load(f)

        for i, d in enumerate(contents):
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
                    "question": d["instruction"],
                    "choices": {"text": text_choices, "label": label_choices},
                    "answerKey": d["answer"],
                }
            else:
                yield i, {
                    "id": i,
                    "question_id": d["id"],
                    "document_id": d["id"],
                    "question": d["instruction"],
                    "type": "multiple_choice",
                    "choices": [f"{label}. {text}" for label, text in zip(label_choices, text_choices)],
                    "context": None,
                    "answer": [f'{d["answer"]}. {text_choices[ord(d["answer"])-65]}'],
                    "meta": {}
                }
