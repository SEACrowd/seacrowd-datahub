from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses

_CITATION = """\
@misc{lopo2024constructing,
      title={Constructing and Expanding Low-Resource and Underrepresented Parallel Datasets for Indonesian Local Languages}, 
      author={Joanito Agili Lopo and Radius Tanone},
      year={2024},
      eprint={2404.01009},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
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
_SEACROWD_VERSION = "2024.06.20"
_LOCAL = False

_LANGUAGES = ["ind", "day", "eng"]

class BeayeLexicon(datasets.GeneratorBasedBuilder):
    """Beaye Lexicon is a lexicon resource encompassing translations between Indonesian, English, and Beaye words"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = (
        [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"beaye lexicon with source schema for {lang} language",
                schema="source",
                subset_id="beaye_lexicon",
            )
            for lang in _LANGUAGES if lang != "eng"
        ]
        + [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_ext_{lang}_source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"beaye lexicon with source schema for extensive definiton of beaye language",
                schema="source",
                subset_id="beaye_lexicon",
            )
            for lang in _LANGUAGES if lang != "ind"
        ]
    )

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
        if "ext" in self.config.name.split("_"):
            data_dir = Path(dl_manager.download(_URLS + "/english.xlsx"))
        else:
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
        if "ext" in self.config.name.split("_"):
            lang = self.config.name.split("_")[3]
        else:
            lang = self.config.name.split("_")[2]

        text = dfs[lang]

        if self.config.schema == "source":
            for idx, word in enumerate(text.values):
                row = {"id": str(idx), "word": word}
                yield idx, row
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
