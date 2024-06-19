from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses

_CITATION = """\
@InProceedings{NOMOTO18.8,
  author = {Hiroki Nomoto ,Hannah Choi ,David Moeljadi and Francis Bond},
  title = {MALINDO Morph: Morphological dictionary and analyser for Malay/Indonesian},
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year = {2018},
  month = {may},
  date = {7-12},
  location = {Miyazaki, Japan},
  editor = {Kiyoaki Shirai},
  publisher = {European Language Resources Association (ELRA)},
  address = {Paris, France},
  isbn = {979-10-95546-24-5},
  language = {english}
  }
"""


_DATASETNAME = "malindo_morph"

_DESCRIPTION = """\
MALINDO Morph is a morphological dictionary for Malay (bahasa Melayu) and Indonesian (bahasa Indonesia) language.
It contains over 200,000 lines, with each containing an analysis for one (case-sensitive) token.
Each line is made up of the following six items, separated by tabs: root, surface form, prefix, suffix, circumfix, reduplication.
"""

_HOMEPAGE = "https://github.com/matbahasa/MALINDO_Morph"

_LANGUAGES = ["zlm", "ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_4_0.value  # example: Licenses.MIT.value, Licenses.CC_BY_NC_SA_4_0.value, Licenses.UNLICENSE.value, Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/matbahasa/MALINDO_Morph/master/malindo_dic_2023.tsv",
}

_SUPPORTED_TASKS = []

_SOURCE_VERSION = "2023.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MalindoMorph(datasets.GeneratorBasedBuilder):
    """MALINDO Morph is a morphological dictionary for Malay (bahasa Melayu) and Indonesian (bahasa Indonesia) language. It provides morphological information (root, prefix, suffix, circumfix, reduplication) for over 200,000 surface forms."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "root": datasets.Value("string"),
                    "bentuk_jadian": datasets.Value("string"),
                    "prefix": datasets.Value("string"),
                    "suffix": datasets.Value("string"),
                    "circumfix": datasets.Value("string"),
                    "reduplication": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "stem": datasets.Value("string"),
                    "lemma": datasets.Value("string"),
                }
            )

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
        file = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        rows = []
        with open(filepath, encoding="utf8") as file:
            for line in file:
                row = line.split("\t")
                row[-1] = row[-1].split("\n")[0] # remove newlines from lemma feature
                rows.append(row)

        if self.config.schema == "source":
            for key, row in enumerate(rows):
                example = {"id": row[0], "root": row[1], "bentuk_jadian": row[2], "prefix": row[3], "suffix": row[4], "circumfix": row[5], "reduplication": row[6], "source": row[7], "stem": row[8], "lemma": row[9]}
                yield key, example
