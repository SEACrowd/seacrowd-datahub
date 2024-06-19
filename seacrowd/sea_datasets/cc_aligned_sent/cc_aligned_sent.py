from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = r"""
@inproceedings{chaudhary-etal-2019-low,
    title = "Low-Resource Corpus Filtering Using Multilingual Sentence Embeddings",
    author = "Chaudhary, Vishrav  and
      Tang, Yuqing  and
      Guzm{\'a}n, Francisco  and
      Schwenk, Holger  and
      Koehn, Philipp",
    editor = "Bojar, Ond{\v{r}}ej  and
      Chatterjee, Rajen  and
      Federmann, Christian  and
      Fishel, Mark  and
      Graham, Yvette  and
      Haddow, Barry  and
      Huck, Matthias  and
      Yepes, Antonio Jimeno  and
      Koehn, Philipp  and
      Martins, Andr{\'e}  and
      Monz, Christof  and
      Negri, Matteo  and
      N{\'e}v{\'e}ol, Aur{\'e}lie  and
      Neves, Mariana  and
      Post, Matt  and
      Turchi, Marco  and
      Verspoor, Karin",
    booktitle = "Proceedings of the Fourth Conference on Machine Translation (Volume 3: Shared Task Papers, Day 2)",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5435",
    doi = "10.18653/v1/W19-5435",
    pages = "261--266",
}
"""

_LOCAL = False
_LANGUAGES = ["ind", "jav", "sun", "tha", "vie", "zlm", "lao", "khm", "mya", "ceb"]
_DATASETNAME = "cc_aligned_sent"
_DESCRIPTION = """\
This dataset contains the sentence pairs extracted from CC-Aligned document
pairs using similarity scores of LASER embeddings (minimum similarity 1.04,
sorted based on decreasing similarity score). It misses some languages not
covered by LASER.
"""

_HOMEPAGE = "https://www2.statmt.org/cc-aligned/"
_LICENSE = Licenses.UNKNOWN.value
_URL = "https://data.statmt.org/cc-aligned/sentence-aligned/"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_SUBSETS = ["id_ID", "jv_ID", "su_ID", "th_TH", "vi_VN", "ms_MY", "lo_LA", "km_KH", "my_MM", "cx_PH"]


class CCAlignedSentencesDataset(datasets.GeneratorBasedBuilder):
    """CC Aligned Sentences dataset by Chaudhary et al., (2019)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "t2t"

    # Add configurations for loading a dataset per language.
    dataset_names = sorted([f"{_DATASETNAME}_{subset}" for subset in _SUBSETS])
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
    first_subset = sorted(_SUBSETS)[0]
    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_{first_subset}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Source_Sentence": datasets.Value("string"),
                    "Target_Sentence": datasets.Value("string"),
                    "LASER_similarity": datasets.Value("float64"),
                }
            )

        if self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text_to_text.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        """Return SplitGenerators."""
        # Define some functions for parsing config and URL names
        def _split_at_n(text: str, n: int) -> Tuple[str, str]:
            """Split text on the n-th instance"""
            return ("_".join(text.split("_")[:n]), "_".join(text.split("_")[n:]))

        # Get URL. For cx_PH, the source and target languages are reversed
        _, subset = _split_at_n(_split_at_n(self.config.name, 5)[0], 3)
        (source_lang, target_lang) = (subset, "en_XX") if subset == "cx_PH" else ("en_XX", subset)
        url = _URL + f"{source_lang}-{target_lang}.tsv.xz"
        filepath = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                },
            )
        ]

    def _generate_examples(self, filepath: Path, source_lang: str, target_lang: str) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""
        with open(filepath, encoding="utf-8") as file:
            for idx, row in enumerate(file):
                text_1, text_2, score = row.strip().split("\t")
                if self.config.schema == "source":
                    example = {
                        "id": idx,
                        "Source_Sentence": text_1,
                        "Target_Sentence": text_2,
                        "LASER_similarity": float(score),
                    }
                if self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                    example = {
                        "id": idx,
                        "text_1": text_1,
                        "text_2": text_2,
                        "text_1_name": source_lang,
                        "text_2_name": target_lang,
                    }
                yield idx, example
