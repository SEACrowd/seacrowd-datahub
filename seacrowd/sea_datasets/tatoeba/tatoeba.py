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
_LANGUAGES = ["ind", "vie", "tgl", "jav", "tha", "eng"]
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
_SEACROWD_VERSION = "2024.06.20"


class TatoebaDataset(datasets.GeneratorBasedBuilder):
    """Tatoeba subset for Indonesian, Vietnamese, Tagalog, Javanese, and Thai."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "t2t"

    # Add configurations for loading a dataset per language.
    dataset_names = sorted([f"tatoeba_{lang}_eng" for lang in _LANGUAGES[:-1]]) + sorted([f"tatoeba_eng_{lang}" for lang in _LANGUAGES[:-1]])
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

    # Add configuration that allows loading all datasets at once.
    BUILDER_CONFIGS.extend(
        [
            # tatoeba_source
            SEACrowdConfig(
                name=f"{_DATASETNAME}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} source schema (all)",
                schema="source",
                subset_id=_DATASETNAME,
            ),
            # tatoeba_seacrowd_t2t
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} SEACrowd schema (all)",
                schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
                subset_id=_DATASETNAME,
            ),
        ]
    )

    # Choose first language as default
    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

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
        language_pairs = []
        tatoeba_source_data = []
        tatoeba_eng_data = []

        lang_1 = self.config.name.split("_")[1]
        lang_2 = self.config.name.split("_")[2]
        if lang_1 == "eng":
            lang = lang_2
        else:
            lang = lang_1

        if lang in _LANGUAGES:
            # Load data per language
            tatoeba_source_data.append(dl_manager.download_and_extract(_URL + f"tatoeba.{lang}-eng.{lang_1}"))
            tatoeba_eng_data.append(dl_manager.download_and_extract(_URL + f"tatoeba.{lang}-eng.{lang_2}"))
            language_pairs.append((lang_1, lang_2))
        else:
            # Load examples from all languages at once
            # We just want to run this part when tatoeba_source / tatoeba_seacrowd_t2t was chosen.
            for lang in _LANGUAGES[:-1]:
                tatoeba_source_data.append(dl_manager.download_and_extract(_URL + f"tatoeba.{lang}-eng.{lang}"))
                tatoeba_eng_data.append(dl_manager.download_and_extract(_URL + f"tatoeba.{lang}-eng.eng"))
                language_pairs.append((lang, "eng"))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": (tatoeba_source_data, tatoeba_eng_data),
                    "split": "dev",
                    "language_pairs": language_pairs,
                },
            )
        ]

    def _generate_examples(self, filepaths: Tuple[List[Path], List[Path]], split: str, language_pairs: List[str]) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""
        source_files, target_files = filepaths
        source_sents = []
        target_sents = []
        source_langs = []
        target_langs = []

        for source_file, target_file, (lang_1, lang_2) in zip(source_files, target_files, language_pairs):
            with open(source_file, encoding="utf-8") as f1:
                for row in f1:
                    source_sents.append(row.strip())
                    source_langs.append(lang_1)
            with open(target_file, encoding="utf-8") as f2:
                for row in f2:
                    target_sents.append(row.strip())
                    target_langs.append(lang_2)

        for idx, (source, target, lang_src, lang_tgt) in enumerate(zip(source_sents, target_sents, source_langs, target_langs)):
            if self.config.schema == "source":
                example = {
                    "source_sentence": source,
                    "target_sentence": target,
                    # The source_lang in the HuggingFace source seems incorrect
                    # I am overriding it with the actual language code.
                    "source_lang": lang_src,
                    "target_lang": lang_tgt,
                }
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                example = {
                    "id": str(idx),
                    "text_1": source,
                    "text_2": target,
                    # The source_lang in the HuggingFace source seems incorrect
                    # I am overriding it with the actual language code.
                    "text_1_name": lang_src,
                    "text_2_name": lang_tgt,
                }
            yield idx, example
