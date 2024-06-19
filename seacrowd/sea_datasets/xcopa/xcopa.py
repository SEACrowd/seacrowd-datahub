"""This code is partially taken from https://github.com/huggingface/datasets/blob/main/datasets/xcopa/xcopa.py."""

import json

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_HOMEPAGE = "https://github.com/cambridgeltl/xcopa"

_CITATION = """\
@inproceedings{ponti2020xcopa,
  title={{XCOPA: A} Multilingual Dataset for Causal Commonsense Reasoning},
  author={Edoardo M. Ponti, Goran Glava\v{s}, Olga Majewska, Qianchu Liu, Ivan Vuli\'{c} and Anna Korhonen},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020},
  url={https://ducdauge.github.io/files/xcopa.pdf}
}
@inproceedings{roemmele2011choice,
  title={Choice of plausible alternatives: An evaluation of commonsense causal reasoning},
  author={Roemmele, Melissa and Bejan, Cosmin Adrian and Gordon, Andrew S},
  booktitle={2011 AAAI Spring Symposium Series},
  year={2011},
  url={https://people.ict.usc.edu/~gordon/publications/AAAI-SPRING11A.PDF},
}
"""

_LANGUAGES = ["ind", "tha", "vie"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False

_DATASETNAME = "xcopa"

_DESCRIPTION = """\
  XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning
The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across
languages. The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around
the globe. The dataset is challenging as it requires both the command of world knowledge and the ability to generalise to new languages. All the details about the
creation of XCOPA and the implementation of the baselines are available in the paper.
"""

_HOMEPAGE = "https://github.com/cambridgeltl/xcopa"

_LICENSE = Licenses.CC_BY_4_0.value

_URLS = {
    "ind": [
        "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/id/val.id.jsonl",
        "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/id/test.id.jsonl",
    ],
    "tha": [
        "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/th/val.th.jsonl",
        "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/th/test.th.jsonl",
    ],
    "vie": [
        "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/vi/val.vi.jsonl",
        "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/vi/test.vi.jsonl",
    ],
}

_SUPPORTED_TASKS = [Tasks.COMMONSENSE_REASONING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


def _xcopa_config_constructor(lang: str, schema: str, version: str) -> SEACrowdConfig:
    return SEACrowdConfig(
        name="xcopa_{}_{}".format(lang, schema),
        version=version,
        description="XCOPA {} schema".format(schema),
        schema=schema,
        subset_id="xcopa",
    )


class Xcopa(datasets.GeneratorBasedBuilder):
    """The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across
    languages. The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around
    the globe."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [_xcopa_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGUAGES] + [_xcopa_config_constructor(lang, "seacrowd_qa", _SEACROWD_VERSION) for lang in _LANGUAGES]

    DEFAULT_CONFIG_NAME = "xcopa_ind_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "choice1": datasets.Value("string"),
                    "choice2": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                    "idx": datasets.Value("int32"),
                    "changed": datasets.Value("bool"),
                }
            )
        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features
            features_in_dict = features.to_dict()
            features_in_dict["meta"] = {"is_changed": {"dtype": "bool", "_type": "Value"}, "reasoning_type": {"dtype": "string", "_type": "Value"}}
            features = datasets.Features.from_dict(features_in_dict)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def get_lang(self, name: str):
        # xcopa_ind|
        # [xcopa, ind]
        names_splitted = name.split("_")
        if len(names_splitted) == 0:
            return "ind"
        return names_splitted[1]

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls = _URLS[self.get_lang(self.config.name)]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir[0],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir[1],
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        if self.config.schema == "source":
            with open(filepath, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    idx = data["idx"]
                    yield idx, data

        elif self.config.schema == "seacrowd_qa":
            with open(filepath, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    idx = data["idx"]
                    sample = {
                        "id": str(idx),
                        "question_id": str(idx),
                        "document_id": str(idx),
                        "question": "",
                        "type": "multiple_choice",
                        "choices": [data["choice1"], data["choice2"]],
                        "context": data["premise"],
                        "answer": [data["choice1"] if data["label"] == 0 else data["choice2"]],
                        "meta": {"is_changed": data["changed"], "reasoning_type": data["question"]},
                    }
                    yield idx, sample

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
