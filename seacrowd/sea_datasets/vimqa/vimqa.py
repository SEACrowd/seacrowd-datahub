import json
import os
from pathlib import Path

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{le-etal-2022-vimqa,
    title = "{VIMQA}: A {V}ietnamese Dataset for Advanced Reasoning and Explainable Multi-hop Question Answering",
    author = "Le, Khang  and
      Nguyen, Hien  and
      Le Thanh, Tung  and
      Nguyen, Minh",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\'e}ne  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.700",
    pages = "6521--6529",
}
"""

_DATASETNAME = "vimqa"

_DESCRIPTION = """
VIMQA, a new Vietnamese dataset with over 10,000 Wikipedia-based multi-hop question-answer pairs. The dataset is human-generated and has four main features:
The questions require advanced reasoning over multiple paragraphs.
Sentence-level supporting facts are provided, enabling the QA model to reason and explain the answer.
The dataset offers various types of reasoning to test the model's ability to reason and extract relevant proof.
The dataset is in Vietnamese, a low-resource language
"""

_HOMEPAGE = "https://github.com/vimqa/vimqa"

_LANGUAGES = ["vie"]

_LICENSE = f"""{Licenses.OTHERS.value} | \
    The licence terms for VimQA follows this EULA docs on their repo.
    Please refer to the following doc of EULA (to review the permissions and request for access)
    VIMQA EULA -- https://github.com/vimqa/vimqa/blob/main/VIMQA_EULA.pdf
"""

_LOCAL = True

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class VimqaDataset(datasets.GeneratorBasedBuilder):
    """VIMQA, a new Vietnamese dataset with over 10,000 Wikipedia-based multi-hop question-answer pairs."""

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
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "supporting_facts": datasets.features.Sequence(
                        {
                            "title": datasets.Value("string"),
                            "sent_id": datasets.Value("int32"),
                        }
                    ),
                    "context": datasets.features.Sequence(
                        {
                            "title": datasets.Value("string"),
                            "sentences": datasets.features.Sequence(datasets.Value("string")),
                        }
                    ),
                }
            )
        else:
            features = schemas.qa_features
            features["meta"] = {
                "supporting_facts": datasets.features.Sequence(
                    {
                        "title": datasets.Value("string"),
                        "sent_id": datasets.Value("int32"),
                    }
                ),
                "context": datasets.features.Sequence(
                    {
                        "title": datasets.Value("string"),
                        "sentences": datasets.features.Sequence(datasets.Value("string")),
                    }
                ),
            }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "vimqa_train.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "vimqa_dev.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "vimqa_test.json")},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> tuple[int, dict]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, item in enumerate(data):
            if self.config.schema == "source":
                yield i, {
                    "id": item["_id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "type": item["type"],
                    "supporting_facts": [{"title": f[0], "sent_id": f[1]} for f in item["supporting_facts"]],
                    "context": [{"title": f[0], "sentences": f[1]} for f in item["context"]],
                }
            else:
                yield i, {
                    "id": str(i),
                    "question_id": item["_id"],
                    "document_id": "",
                    "question": item["question"],
                    "type": item["type"],
                    "choices": [],
                    "context": "",
                    "answer": [item["answer"]],
                    "meta": {
                        "supporting_facts": [{"title": f[0], "sent_id": f[1]} for f in item["supporting_facts"]],
                        "context": [{"title": f[0], "sentences": f[1]} for f in item["context"]],
                    },
                }
