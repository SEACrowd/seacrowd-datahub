from pathlib import Path
from typing import List

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (DEFAULT_SEACROWD_VIEW_NAME,
                                      DEFAULT_SOURCE_VIEW_NAME, Licenses,
                                      Tasks)

_DATASETNAME = "wikimatrix"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME

# ilo min sun are actually not available
_LANGUAGES = ["ilo", "min", "jav", "sun", "ceb", "ind", "tgl", "vie"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{schwenk-etal-2021-wikimatrix,
    title = "{W}iki{M}atrix: Mining 135{M} Parallel Sentences in 1620 Language Pairs from {W}ikipedia",
    author = "Schwenk, Holger  and
      Chaudhary, Vishrav  and
      Sun, Shuo  and
      Gong, Hongyu  and
      Guzm{\'a}n, Francisco",
    editor = "Merlo, Paola  and
      Tiedemann, Jorg  and
      Tsarfaty, Reut",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-main.115",
    doi = "10.18653/v1/2021.eacl-main.115",
    pages = "1351--1361",
    abstract = "We present an approach based on multilingual sentence embeddings to automatically extract parallel sentences from the content
    of Wikipedia articles in 96 languages, including several dialects or low-resource languages. We do not limit the extraction process to
    alignments with English, but we systematically consider all possible language pairs. In total, we are able to extract 135M parallel sentences
    for 16720 different language pairs, out of which only 34M are aligned with English. This corpus is freely available. To get an indication
    on the quality of the extracted bitexts, we train neural MT baseline systems on the mined data only for 1886 languages pairs, and evaluate
    them on the TED corpus, achieving strong BLEU scores for many language pairs. The WikiMatrix bitexts seem to be particularly interesting
    to train MT systems between distant languages without the need to pivot through English.",
}
"""

_DESCRIPTION = """\
WikiMatrix is automatically extracted parallel sentences from the content of Wikipedia articles in 96 languages, including several dialects
or low-resource languages. 8 languages among them are spoken in Southeast Asia region. In total, there are 135M parallel sentences from 1620
different language pairs.
"""

_HOMEPAGE = "https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix"

_LICENSE = Licenses.CC_BY_SA_4_0.value

_URLs = "https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.{lang1}-{lang2}.tsv.gz"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

config = {
    "jv": ["en", "es", "fr", "id", "it", "pt"],
    "ceb": ["bg", "ar", "ca", "cs", "de", "en", "es", "fi", "fr", "hu", "it", "ja", "nl", "no", "pl", "pt", "ro", "ru", "sv", "uk"],
    "id": [
        "jv",
        "is",
        "it",
        "ja",
        "ko",
        "lt",
        "mk",
        "ml",
        "mr",
        "ne",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sh",
        "si",
        "sk",
        "sl",
        "sq",
        "sr",
        "sv",
        "sw",
        "ta",
        "te",
        "tl",
        "tr",
        "tt",
        "uk",
        "vi",
        "zh",
        "ar",
        "az",
        "ba",
        "bg",
        "bn",
        "bs",
        "ca",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "eo",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "gl",
        "he",
        "hi",
        "hr",
        "hu",
    ],
    "tl": ["ar", "bg", "bs", "ca", "cs", "da", "de", "el", "en", "eo", "es", "et", "fi", "fr", "gl", "he", "hr", "hu", "id", "it", "ja", "lt", "mk", "nl", "no", "pl", "pt", "ro", "ru", "sh", "sk", "sl", "sq", "sr", "sv", "tr", "uk", "vi", "zh"],
    "vi": [
        "ar",
        "az",
        "bg",
        "bn",
        "bs",
        "ca",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "eo",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "gl",
        "he",
        "hi",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ja",
        "ko",
        "lt",
        "mk",
        "ml",
        "mr",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sh",
        "si",
        "sk",
        "sl",
        "sq",
        "sr",
        "sv",
        "sw",
        "ta",
        "te",
        "tl",
        "tr",
        "uk",
        "zh",
    ],
}
_SUBSETS = set()
for lang, pairs in config.items():
    for pair in pairs:
        _SUBSETS.add("{}-{}".format(lang, pair) if lang < pair else "{}-{}".format(pair, lang))
_SUBSETS = list(_SUBSETS)


class WikiMatrixDataset(datasets.GeneratorBasedBuilder):
    """WikiMatrix is automatically extracted parallel sentences from the content of Wikipedia articles in 96 languages, including several dialects
    or low-resource languages."""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"wikimatrix_{subset.replace('-', '_')}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="WikiMatrix source schema",
            schema="source",
            subset_id=f"wikimatrix_{subset.replace('-', '_')}",
        )
        for subset in _SUBSETS
    ] + [
        SEACrowdConfig(
            name=f"wikimatrix_{subset.replace('-', '_')}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description="WikiMatrix Nusantara schema",
            schema="seacrowd_t2t",
            subset_id=f"wikimatrix_{subset.replace('-', '_')}",
        )
        for subset in _SUBSETS
    ]

    DEFAULT_CONFIG_NAME = "wikimatrix_en_id_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "text_1_name": datasets.Value("string"),
                    "text_2_name": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        lang1, lang2 = self.config.name.split("_")[1], self.config.name.split("_")[2]
        filepath = Path(dl_manager.download_and_extract(_URLs.format(lang1=lang1, lang2=lang2)))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": filepath},
            ),
        ]

    def _generate_examples(self, filepath: Path):
        with open(filepath, "r") as f:
            data = f.readlines()

        lang1, lang2 = self.config.name.split("_")[1], self.config.name.split("_")[2]
        if self.config.schema == "source":
            for _id, line in enumerate(data):
                line = line.strip().split("\t")
                ex = {
                    "id": str(_id),
                    "text_1": line[1],
                    "text_2": line[2],
                    "text_1_name": lang1,
                    "text_2_name": lang2,
                }
                yield _id, ex

        elif self.config.schema == "seacrowd_t2t":
            for _id, line in enumerate(data):
                line = line.strip().split("\t")
                ex = {
                    "id": str(_id),
                    "text_1": line[1],
                    "text_2": line[2],
                    "text_1_name": lang1,
                    "text_2_name": lang2,
                }
                yield _id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
