from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses

_CITATION = """\
@misc{beayelexicon2024,
  author    = {Lopo, Joanito Agili and Moeljadi, David and Cahyawijaya, Samuel and Aji, Alham Fikri and Sommerlot, 
  Carly J. and Jacob, June},
  title     = {Penyusunan Korpus Paralel Bahasa Indonesia–Bahasa Melayu Ambon, Melayu Kupang, Beaye, dan Uab Meto},
  year      = {2024},
  howpublished = {Online},
  url       = {https://github.com/joanitolopo/makalah-kongresxii},
  note      = {Manuscript in preparation},
}
"""

_DATASETNAME = "beaye_lexicon"
_DESCRIPTION = """The Beaye Lexicon is a lexicon resource encompassing translations between Indonesian, English, and 
Beaye words. Developed through a collaborative effort involving two native Beaye speakers and evaluated by linguistic 
experts, this lexicon comprises 984 Beaye vocabularies. The creation of the Beaye Lexicon marks the inaugural effort in 
documenting the previously unrecorded Beaye language."""

_HOMEPAGE = "https://github.com/joanitolopo/bhinneka-korpus/tree/main/lexicon"
_LICENSE = Licenses.APACHE_2_0.value
_URLS = "https://raw.githubusercontent.com/joanitolopo/bhinneka-korpus/main/lexicon"
_SUPPORTED_TASKS = []
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"

_LANGUAGES = ["ind", "day"]


def seacrowd_config_constructor(lang, schema, version):
    if lang == "":
        raise ValueError(f"Invalid lang {lang}")

    if schema != "source":
        raise ValueError(f"Invalid schema: {schema}")

    return SEACrowdConfig(
        name=f"{_DATASETNAME}_{lang}_{schema}",
        version=datasets.Version(version),
        description=f"beaye lexicon with {schema} schema for {lang} language",
        schema=schema,
        subset_id="beaye_lexicon",
    )


class BeayeLexicon(datasets.GeneratorBasedBuilder):
    """Beaye Lexicon is a lexicon resource encompassing translations between Indonesian, English, and Beaye words"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    BUILDER_CONFIGS = [seacrowd_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGUAGES]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_ind_source"

    def _info(self) -> datasets.DatasetInfo:
        schema = self.config.schema
        if schema == "source":
            features = datasets.Features({"id": datasets.Value("string"), "word": datasets.Value("string")})
        else:
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = Path(dl_manager.download(_URLS + "/lexicon.xlsx"))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                }
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        dfs = pd.read_excel(filepath, engine="openpyxl")
        lang = self.config.name.split("_")[2]
        text = dfs[lang]

        if self.config.schema == "source":
            for idx, word in enumerate(text.values):
                row = {"id": str(idx), "word": word}
                yield idx, row
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
