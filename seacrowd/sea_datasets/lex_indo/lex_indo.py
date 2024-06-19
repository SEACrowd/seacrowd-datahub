from pathlib import Path
from typing import List

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{magichubLEXIndoIndonesian,
  author    = {},
  title     = {LEX-INDO: AN INDONESIAN LEXICON},
  year      = {},
  howpublished = {Online},
  url       = {https://magichub.com/datasets/indonesian-lexicon/},
  note      = {Accessed 19-03-2024},
}
"""

_DATASETNAME = "lex_indo"

_DESCRIPTION = """This open-source lexicon consists of 2,000 common Indonesian words, with phoneme series attached.
It is intended to be used as the lexicon for an automatic speech recognition system or a text-to-speech system.
The dictionary presents words as well as their pronunciation transcribed with an ARPABET(phone set of CMU)-like phone set. Syllables are separated with dots.
"""

_HOMEPAGE = "https://magichub.com/datasets/indonesian-lexicon/"
_LICENSE = Licenses.CC_BY_NC_ND_4_0.value
_LOCAL = True

_URLS = {}

_SUPPORTED_TASKS = [Tasks.MULTILEXNORM]

_LANGUAGES = ["ind"]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class LexIndo(datasets.GeneratorBasedBuilder):
    """This open-source lexicon consists of 2,000 common Indonesian words, with phoneme series attached"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "t2t"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} lexicon with source schema",
            schema="source",
            subset_id=_DATASETNAME,
        )
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} lexicon with SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        )
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        schema = self.config.schema
        if schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "word": datasets.Value("string"), "phoneme": datasets.Value("string")})
        else:
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            )
        ]

    def _generate_examples(self, filepath: Path):
        """Yields examples as (key, example) tuples."""

        try:
            with open(f"{filepath}/Indonesian_dic.txt", "r") as f:
                data = f.readlines()
        except FileNotFoundError:
            print("File not found. Please check the file path. Make sure Indonesian_dic.txt is in dest directory")
        except IOError:
            print("An error occurred while trying to read the file.")

        for idx, text in enumerate(data):
            word_i = text.split()[0]
            phoneme_i = " ".join(text.split()[1:])
            if self.config.schema == "source":
                example = {"id": str(idx), "word": word_i, "phoneme": phoneme_i}
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                example = {
                    "id": str(idx),
                    "text_1": word_i,
                    "text_2": phoneme_i,
                    "text_1_name": _LANGUAGES[-1],
                    "text_2_name": "phoneme",
                }
            else:
                raise ValueError(f"Invalid config: {self.config.name}")
            yield idx, example
