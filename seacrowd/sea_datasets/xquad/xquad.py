import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@article{Artetxe:etal:2019,
      author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      title     = {On the cross-lingual transferability of monolingual representations},
      journal   = {CoRR},
      volume    = {abs/1910.11856},
      year      = {2019},
      archivePrefix = {arXiv},
      eprint    = {1910.11856}
}
"""

_DATASETNAME = "xquad"

_DESCRIPTION = """\
XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering performance. The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 (Rajpurkar et al., 2016) together with their professional translations into ten languages: Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, and Hindi. Consequently, the dataset is entirely parallel across 11 languages.
"""

_HOMEPAGE = "https://github.com/google-deepmind/xquad"

_LICENSE = Licenses.CC_BY_SA_4_0.value

_URLS = "https://raw.githubusercontent.com/google-deepmind/xquad/master/"

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class XQuADDataset(datasets.GeneratorBasedBuilder):
    """
    XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering performance.
    The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 together
    with their professional translations into ten languages: Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, and Hindi.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    subsets = ["xquad.vi", "xquad.th"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="{sub}_source".format(sub=subset),
            version=datasets.Version(_SOURCE_VERSION),
            description="{sub} source schema".format(sub=subset),
            schema="source",
            subset_id="{sub}".format(sub=subset),
        )
        for subset in subsets
    ] + [
        SEACrowdConfig(
            name="{sub}_seacrowd_qa".format(sub=subset),
            version=datasets.Version(_SEACROWD_VERSION),
            description="{sub} SEACrowd schema".format(sub=subset),
            schema="seacrowd_qa",
            subset_id="{sub}".format(sub=subset),
        )
        for subset in subsets
    ]

    DEFAULT_CONFIG_NAME = "xquad.vi_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {"context": datasets.Value("string"), "question": datasets.Value("string"), "answers": datasets.Features({"answer_start": [datasets.Value("int64")], "text": [datasets.Value("string")]}), "id": datasets.Value("string")}
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
        name_split = self.config.name.split("_")
        filepath = dl_manager.download_and_extract(_URLS + name_split[0] + ".json")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for paragraphs in data["data"]:
            for example in paragraphs["paragraphs"]:
                context = example["context"]
                for qa in example["qas"]:
                    question = qa["question"]
                    id_ = qa["id"]
                    answers = qa["answers"]
                    answers_start = [answer["answer_start"] for answer in answers]
                    answers_text = [answer["text"] for answer in answers]

                    if self.config.schema == "source":
                        yield count, {
                            "context": context,
                            "question": question,
                            "answers": {"answer_start": answers_start, "text": answers_text},
                            "id": id_,
                        }
                        count += 1

                    elif self.config.schema == "seacrowd_qa":
                        yield count, {"question_id": id_, "context": context, "question": question, "answer": {"answer_start": answers_start[0], "text": answers_text[0]}, "id": id_, "choices": [], "type": "extractive", "document_id": count}
                        count += 1


if __name__ == "__main__":
    datasets.load_dataset(__file__)
