# Some code referenced from https://huggingface.co/datasets/xnli/blob/main/xnli.py

"""
The Cross-lingual Natural Language Inference (XNLI) corpus is a crowd-sourced collection of 5,000 test and 2,500 dev pairs for the MultiNLI corpus. The pairs are annotated with textual entailment and translated into 14 languages: French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili and Urdu. This results in 112.5k annotated pairs. Each premise can be associated with the corresponding hypothesis in the 15 languages, summing up to more than 1.5M combinations.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@InProceedings{conneau2018xnli,
  author = "Conneau, Alexis
        and Rinott, Ruty
        and Lample, Guillaume
        and Williams, Adina
        and Bowman, Samuel R.
        and Schwenk, Holger
        and Stoyanov, Veselin",
  title = "XNLI: Evaluating Cross-lingual Sentence Representations",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  location = "Brussels, Belgium",
}
"""

_DATASETNAME = "xnli"

_DESCRIPTION = """\
XNLI is a subset of a few thousand examples from MNLI which has been translated into a 14 different languages (some low-ish resource). As with MNLI, the goal is to predict textual entailment (does sentence A imply/contradict/neither sentence B) and is a classification task (given two sentences, predict one of three labels).
"""

_HOMEPAGE = "https://github.com/facebookresearch/XNLI"

# We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LANGUAGES = ["tha", "vie"]
_LANGUAGE_MAPPER = {"tha": "th", "vie": "vi"}

_LICENSE = Licenses.CC_BY_NC_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "train": "https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip",
        "test": "https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip",
    }
}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.1.0"

_SEACROWD_VERSION = "2024.06.20"


class XNLIDataset(datasets.GeneratorBasedBuilder):
    """
    XNLI is an evaluation corpus for language transfer and cross-lingual sentence classification in 15 languages.
    In SeaCrowd, we currently only have Thailand and Vietnam Language.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    subsets = ["xnli.tha", "xnli.vie"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{sub}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{sub} source schema",
            schema="source",
            subset_id=f"{sub}",
        )
        for sub in subsets
    ] + [
        SEACrowdConfig(
            name=f"{sub}_seacrowd_pairs",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{sub} SEACrowd schema",
            schema="seacrowd_pairs",
            subset_id=f"{sub}",
        )
        for sub in subsets
    ]

    DEFAULT_CONFIG_NAME = "xnli.vie_source"
    labels = ["contradiction", "entailment", "neutral"]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=self.labels),
                }
            )

        elif self.config.schema == "seacrowd_pairs":
            features = schemas.pairs_features(self.labels)

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

        xnli_train = os.path.join(data_dir["train"], "XNLI-MT-1.0", "multinli")
        train_data_path = os.path.join(xnli_train, "multinli.train.{}.tsv")

        xnli_test = os.path.join(data_dir["test"], "XNLI-1.0")
        val_data_path = os.path.join(xnli_test, "xnli.dev.tsv")
        test_data_path = os.path.join(xnli_test, "xnli.test.tsv")

        lang = self.config.name.split("_")[0].split(".")[-1]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_data_path,
                    "split": "train",
                    "language": _LANGUAGE_MAPPER[lang],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": val_data_path,
                    "split": "dev",
                    "language": _LANGUAGE_MAPPER[lang],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_data_path,
                    "split": "test",
                    "language": _LANGUAGE_MAPPER[lang],
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str, language: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            if split == "train":
                file = open(filepath.format(language), encoding="utf-8")
                reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row_idx, row in enumerate(reader):
                    key = str(row_idx)
                    yield key, {
                        "premise": row["premise"],
                        "hypothesis": row["hypo"],
                        "label": row["label"].replace("contradictory", "contradiction"),
                    }
            else:
                with open(filepath, encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                    for row in reader:
                        if row["language"] == language:
                            yield row["pairID"], {
                                "premise": row["sentence1"],
                                "hypothesis": row["sentence2"],
                                "label": row["gold_label"],
                            }
        elif self.config.schema == "seacrowd_pairs":
            if split == "train":
                file = open(filepath.format(language), encoding="utf-8")
                reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row_idx, row in enumerate(reader):
                    yield str(row_idx), {
                        "id": str(row_idx),
                        "text_1": row["premise"],
                        "text_2": row["hypo"],
                        "label": row["label"].replace("contradictory", "contradiction"),
                    }
            else:
                with open(filepath, encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                    skip = set()
                    for row in reader:
                        if row["language"] == language:
                            if row["pairID"] in skip:
                                continue
                            skip.add(row["pairID"])
                            yield row["pairID"], {
                                "id": row["pairID"],
                                "text_1": row["sentence1"],
                                "text_2": row["sentence2"],
                                "label": row["gold_label"],
                            }
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
