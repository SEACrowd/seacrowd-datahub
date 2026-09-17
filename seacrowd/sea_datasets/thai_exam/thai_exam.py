"""ThaiExam multiple-choice questions from five Thai examinations."""

import json
from copy import deepcopy

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@article{pipatanakul2023typhoon,
  title={Typhoon: Thai Large Language Models},
  author={Pipatanakul, Kunat and Jirabovonvisut, Phatrasek and Manakul, Potsawee and Sripaisarnmongkol, Sittipong and Patomwong, Ruangsak and Chokchainant, Pathomporn and Tharnpipitchai, Kasima},
  journal={arXiv preprint arXiv:2312.13951},
  year={2023}
}
"""
_DATASETNAME = "thai_exam"
_DESCRIPTION = """
ThaiExam is a Thai knowledge benchmark developed to evaluate Typhoon. It contains
590 multiple-choice questions from ONET, IC, TGAT, TPAT-1, and A-Level examinations.
Each examination has a train split of five few-shot examples and a test split.
"""
_HOMEPAGE = "https://huggingface.co/datasets/typhoon-ai/thai_exam"
_LANGUAGES = ["tha"]
_LICENSE = Licenses.APACHE_2_0.value
_LOCAL = False
_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2026.09.17"
_SUBSETS = ("onet", "ic", "tgat", "tpat1", "a_level")
_URLS = {f"{_DATASETNAME}_{subset}": {split: f"{_HOMEPAGE}/resolve/main/data/{subset}/{subset}_{split}.jsonl" for split in ("train", "test")} for subset in _SUBSETS}


class ThaiExamDataset(datasets.GeneratorBasedBuilder):
    """Source and SEACrowd QA views of each ThaiExam examination."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_{schema}",
            version=datasets.Version(_SOURCE_VERSION if schema == "source" else _SEACROWD_VERSION),
            description=f"ThaiExam {subset} {schema} schema",
            schema=schema,
            subset_id=f"{_DATASETNAME}_{subset}",
        )
        for subset in _SUBSETS
        for schema in ("source", "seacrowd_qa")
    ]
    DEFAULT_CONFIG_NAME = "thai_exam_onet_source"

    def _info(self):
        metadata = {"subject": datasets.Value("string")}
        if self.config.subset_id == "thai_exam_onet":
            metadata.update({"no": datasets.Value("float64"), "year": datasets.Value("int64")})
        if self.config.schema == "source":
            fields = {name: datasets.Value("string") for name in ("question", "a", "b", "c", "d", "answer")}
            if self.config.subset_id != "thai_exam_ic":
                fields["e"] = datasets.Value("string")
            features = datasets.Features({**fields, **metadata})
        else:
            features = deepcopy(schemas.qa_features)
            features["meta"] = metadata
        return datasets.DatasetInfo(description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION)

    def _split_generators(self, dl_manager):
        paths = dl_manager.download(_URLS[self.config.subset_id])
        return [datasets.SplitGenerator(name=split, gen_kwargs={"filepath": path, "split": split}) for split, path in paths.items()]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as source:
            for index, line in enumerate(source):
                if not line.strip():
                    continue
                row = json.loads(line)
                example_id = f"{self.config.subset_id}_{split}_{index}"
                if self.config.schema == "source":
                    yield example_id, row
                else:
                    choices = [row[label] for label in "abcde" if row.get(label)]
                    label = row["answer"]
                    if label not in ("a", "b", "c", "d", "e") or not row.get(label):
                        raise ValueError(f"Invalid answer label {label!r} in {example_id}")
                    yield example_id, {
                        "id": example_id,
                        "question_id": example_id,
                        "document_id": "",
                        "question": row["question"],
                        "type": "multiple_choice",
                        "choices": choices,
                        "context": "",
                        "answer": [row[label]],
                        "meta": {key: row[key] for key in self.info.features["meta"]},
                    }
