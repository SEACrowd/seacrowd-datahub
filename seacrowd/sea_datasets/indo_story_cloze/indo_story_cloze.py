import csv
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{koto-etal-2022-cloze,
    title = "Cloze Evaluation for Deeper Understanding of Commonsense Stories in {I}ndonesian",
    author = "Koto, Fajri  and
      Baldwin, Timothy  and
      Lau, Jey Han",
    editor = "Bosselut, Antoine  and
      Li, Xiang  and
      Lin, Bill Yuchen  and
      Shwartz, Vered  and
      Majumder, Bodhisattwa Prasad  and
      Lal, Yash Kumar  and
      Rudinger, Rachel  and
      Ren, Xiang  and
      Tandon, Niket  and
      Zouhar, Vil{\'e}m",
    booktitle = "Proceedings of the First Workshop on Commonsense Representation and Reasoning (CSRR 2022)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.csrr-1.2",
    doi = "10.18653/v1/2022.csrr-1.2",
    pages = "8--16",
}
"""

_DATASETNAME = "indo_story_cloze"

_DESCRIPTION = """
A Story Cloze Test framework in Indonesian. A story in our dataset consists of four-sentence premise, one-sentence
correct ending, and one-sentence incorrect ending. In total, we have created 2,325 Indonesian stories with the
train/dev/test split 1,000/200/1,135.
"""

_HOMEPAGE = "https://huggingface.co/datasets/indolem/indo_story_cloze"

_LANGUAGES = ["ind"]

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "train": "https://huggingface.co/datasets/indolem/indo_story_cloze/resolve/main/train.csv",
        "dev": "https://huggingface.co/datasets/indolem/indo_story_cloze/resolve/main/dev.csv",
        "test": "https://huggingface.co/datasets/indolem/indo_story_cloze/resolve/main/test.csv",
    },
}

_SUPPORTED_TASKS = [Tasks.COMMONSENSE_REASONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class IndoStoryClozeDataset(datasets.GeneratorBasedBuilder):
    """IndoStoryCloze is a Story Cloze dataset in Indonesian."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_qa",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_qa",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sentence-1": datasets.Value("string"),
                    "sentence-2": datasets.Value("string"),
                    "sentence-3": datasets.Value("string"),
                    "sentence-4": datasets.Value("string"),
                    "correct_ending": datasets.Value("string"),
                    "incorrect_ending": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_dir, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir, "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_dir, "split": "dev"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        if self.config.schema == "source":
            data = csv.DictReader(open(filepath[split], newline="", encoding="utf-8"))
            for i, row in enumerate(data):
                yield i, {
                    "sentence-1": row["Kalimat-1"],
                    "sentence-2": row["Kalimat-2"],
                    "sentence-3": row["Kalimat-3"],
                    "sentence-4": row["Kalimat-4"],
                    "correct_ending": row["Correct Ending"],
                    "incorrect_ending": row["Incorrect Ending"],
                }

        elif self.config.schema == "seacrowd_qa":
            data = csv.DictReader(open(filepath[split], newline="", encoding="utf-8"))

            def build_question(line):
                # Concatenate the 4 sentences, this can either be the question of the context. Set is as question for
                # now. Some sentences do not have punctuation, hence adding . before concatenation.
                sentences = []
                for k in ["Kalimat-1", "Kalimat-2", "Kalimat-3", "Kalimat-4"]:
                    if line[k].strip()[-1] not in string.punctuation:
                        sentences.append(line[k] + ".")
                    else:
                        sentences.append(line[k])
                return " ".join(sentences)

            for i, row in enumerate(data):
                yield i, {
                    "id": str(i),
                    "question_id": str(i),
                    "document_id": str(i),
                    "question": build_question(row),
                    "type": "multiple_choice",
                    # Reorder choices based on the randomly generated labels, avoiding correct answer at the same order.
                    "choices": [row["Correct Ending"], row["Incorrect Ending"]] if random.randint(0, 1) == 0 else [row["Incorrect Ending"], row["Correct Ending"]],
                    "context": "",
                    "answer": [row["Correct Ending"]],
                    "meta": {},
                }
