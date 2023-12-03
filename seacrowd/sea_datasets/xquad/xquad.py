import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

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
XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering performance.
The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 together (Rajpurkar et al., 2016)
with their professional translations into ten languages in their original implementation: Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, and Hindi and two in this dataloader: Vietnamese & Thai
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

    subsets = ["xquad", "xquad.vi", "xquad.th"]

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

    DEFAULT_CONFIG_NAME = "xquad_source"

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
        subset_name = name_split[0]

        if subset_name == "xquad":
            # For 'xquad' subset, download and combine Vietnamese and Thai subsets
            vi_filepath = dl_manager.download_and_extract(_URLS + "xquad.vi.json")
            th_filepath = dl_manager.download_and_extract(_URLS + "xquad.th.json")
            filepaths = [vi_filepath, th_filepath]
        else:
            # For specific language subsets
            filepath = dl_manager.download_and_extract(_URLS + subset_name + ".json")
            filepaths = [filepath]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": filepaths,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepaths: List[Path], split: str) -> Tuple[int, Dict]:
        count = 0
        document_count = 0
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            for paragraphs in data["data"]:
                for example in paragraphs["paragraphs"]:
                    context = example["context"]
                    for qa in example["qas"]:
                        question = qa["question"]
                        answers = qa["answers"]
                        answers_start = [answer["answer_start"] for answer in answers]
                        answers_text = [answer["text"] for answer in answers]

                        if self.config.schema == "source":
                            yield count, {
                                "context": context,
                                "question": question,
                                "answers": {"answer_start": answers_start, "text": answers_text},
                                "id": count,
                            }
                        elif self.config.schema == "seacrowd_qa":
                            yield count, {
                                "question_id": count,
                                "context": context,
                                "question": question,
                                "answer": {"answer_start": answers_start[0], "text": answers_text[0]},
                                "id": count,
                                "choices": [],
                                "type": "",
                                "document_id": document_count,
                                "meta": {},
                            }

                        count += 1
                    document_count += 1
