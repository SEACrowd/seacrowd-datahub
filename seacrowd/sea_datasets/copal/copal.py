import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{wibowo2023copal,
  title={COPAL-ID: Indonesian Language Reasoning with Local Culture and Nuances},
  author={Wibowo, Haryo Akbarianto and Fuadi, Erland Hilman and Nityasya, Made Nindyatama and Prasojo, Radityo Eko and Aji, Alham Fikri},
  journal={arXiv preprint arXiv:2311.01012},
  year={2023}
}
"""
_DATASETNAME = "copal"

_DESCRIPTION = """\
COPAL is a novel Indonesian language common sense reasoning dataset. Unlike the previous Indonesian COPA dataset (XCOPA-ID), COPAL-ID incorporates Indonesian local and cultural nuances,
providing a more natural portrayal of day-to-day causal reasoning within the Indonesian cultural sphere.
Professionally written by natives from scratch, COPAL-ID is more fluent and free from awkward phrases, unlike the translated XCOPA-ID.
Additionally, COPAL-ID is presented in both standard Indonesian and Jakartan Indonesian–a commonly used dialect.
It consists of premise, choice1, choice2, question, and label, similar to the COPA dataset.
"""

_HOMEPAGE = "https://huggingface.co/datasets/haryoaw/COPAL"

_LICENSE = Licenses.CC_BY_SA_4_0.value

_URLS = {"test": "https://huggingface.co/datasets/haryoaw/COPAL/resolve/main/test_copal.csv?download=true", "test_colloquial": "https://huggingface.co/datasets/haryoaw/COPAL/resolve/main/test_copal_colloquial.csv?download=true"}

_SUPPORTED_TASKS = [Tasks.COMMONSENSE_REASONING]

_LOCAL = False
_LANGUAGES = ["ind"]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class COPAL(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="COPAL test source schema",
            schema="source",
            subset_id="copal",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_colloquial_source",
            version=SOURCE_VERSION,
            description="COPAL test colloquial source schema",
            schema="source",
            subset_id="copal",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_qa",
            version=SEACROWD_VERSION,
            description="COPAL test seacrowd schema",
            schema="seacrowd_qa",
            subset_id="copal",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_colloquial_seacrowd_qa",
            version=SEACROWD_VERSION,
            description="COPAL test colloquial seacrowd schema",
            schema="seacrowd_qa",
            subset_id="copal",
        ),
    ]

    DEFAULT_CONFIG_NAME = "copal_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "choice1": datasets.Value("string"),
                    "choice2": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "idx": datasets.Value("int64"),
                    "label": datasets.Value("int64"),
                    "terminology": datasets.Value("int64"),
                    "culture": datasets.Value("int64"),
                    "language": datasets.Value("int64"),
                }
            )
        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features
            features["meta"] = {"terminology": datasets.Value("int64"), "culture": datasets.Value("int64"), "language": datasets.Value("int64")}

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URLS)
        if "colloquial" in self.config.name:
            data_url = data_dir["test_colloquial"]
        else:
            data_url = data_dir["test"]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_url},
            ),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath, sep=",", header="infer").reset_index()
        if self.config.schema == "source":
            for row in df.itertuples():
                entry = {
                    "premise": row.premise,
                    "choice1": row.choice1,
                    "choice2": row.choice2,
                    "question": row.question,
                    "idx": row.idx,
                    "label": row.label,
                    "terminology": row.Terminology,
                    "culture": row.Culture,
                    "language": row.Language,
                }
                yield row.index, entry

        elif self.config.schema == "seacrowd_qa":
            for row in df.itertuples():
                entry = {
                    "id": row.idx,
                    "question_id": str(row.idx),
                    "document_id": str(row.idx),
                    "question": row.question,
                    "type": "multiple_choice",
                    "choices": [row.choice1, row.choice2],
                    "context": row.premise,
                    "answer": [row.choice1 if row.label == 0 else row.choice2],
                    "meta": {"terminology": row.Terminology, "culture": row.Culture, "language": row.Language},
                }
                yield row.index, entry
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
