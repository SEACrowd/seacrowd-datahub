from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{tatoeba,
    title     = {Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond},
    author    = {Mikel, Artetxe and Holger, Schwenk,},
    journal   = {arXiv:1812.10464v2},
    year      = {2018}
}
"""

_LOCAL = False
_LANGUAGES = ["ind", "vie", "tgl", "jav", "tha"]
_DATASETNAME = "tatoeba"
_DESCRIPTION = """\
This dataset is a subset of the Tatoeba corpus containing language pairs for Indonesian, Vietnamese, Tagalog, Javanese, and Thai.
The original dataset description can be found below:

This data is extracted from the Tatoeba corpus, dated Saturday 2018/11/17.
For each languages, we have selected 1000 English sentences and their translations, if available. Please check
this paper for a description of the languages, their families and scripts as well as baseline results.
Please note that the English sentences are not identical for all language pairs. This means that the results are
not directly comparable across languages. In particular, the sentences tend to have less variety for several
low-resource languages, e.g. "Tom needed water", "Tom needs water", "Tom is getting water", ...
"""

_HOMEPAGE = "https://github.com/facebookresearch/LASER/blob/main/data/tatoeba/v1/README.md"
_LICENSE = Licenses.APACHE_2_0.value
_URL = "https://github.com/facebookresearch/LASER/raw/main/data/tatoeba/v1/"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class TatoebaDatset(datasets.GeneratorBasedBuilder):
    """Tatoeba subset for Indonesian, Vietnamese, Tagalog, Javanese, and Thai."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "t2t"

    dataset_names = sorted([f"tatoeba.{lang}" for lang in _LANGUAGES])
    BUILDER_CONFIGS = []
    for name in dataset_names:
        source_config = SEACrowdConfig(
            name=f"{name}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=name,
        )
        BUILDER_CONFIGS.append(source_config)
        seacrowd_config = SEACrowdConfig(
            name=f"{name}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=name,
        )
        BUILDER_CONFIGS.append(seacrowd_config)

    # Choose first language as default
    DEFAULT_CONFIG_NAME = f"{dataset_names[0]}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "source_sentence": datasets.Value("string"),
                    "target_sentence": datasets.Value("string"),
                    "source_lang": datasets.Value("string"),
                    "target_lang": datasets.Value("string"),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text2text_features
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        """Return SplitGenerators."""
        lang_source = self.config.name.split(".")[1]
        lang = lang_source.split("_")[0]
        tatoeba_source_data = dl_manager.download_and_extract(_URL + f"tatoeba.{lang}-eng.{lang}")
        tatoeba_eng_data = dl_manager.download_and_extract(_URL + f"tatoeba.{lang}-eng.eng")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": (tatoeba_source_data, tatoeba_eng_data),
                    "split": "dev",
                    "lang": lang,
                },
            )
        ]

    def _generate_examples(self, filepath: Tuple[Path, Path], split: str, lang: str) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""
        source_file = filepath[0]
        target_file = filepath[1]
        source_sentences = []
        target_sentences = []
        with open(source_file, encoding="utf-8") as f1:
            for row in f1:
                source_sentences.append(row.strip())
        with open(target_file, encoding="utf-8") as f2:
            for row in f2:
                target_sentences.append(row.strip())
        for idx in range(len(source_sentences)):
            if self.config.schema == "source":
                example = {
                    "source_sentence": source_sentences[idx],
                    "target_sentence": target_sentences[idx],
                    # The source_lang in the HuggingFace source seems incorrect
                    # I am overriding it with the actual language code.
                    "source_lang": lang,
                    "target_lang": "eng",
                }
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                example = {
                    "id": str(idx),
                    "text_1": source_sentences[idx],
                    "text_2": target_sentences[idx],
                    # The source_lang in the HuggingFace source seems incorrect
                    # I am overriding it with the actual language code.
                    "text_1_name": lang,
                    "text_2_name": "eng",
                }
            yield idx, example
