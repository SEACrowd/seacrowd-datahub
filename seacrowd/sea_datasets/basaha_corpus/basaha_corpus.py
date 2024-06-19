from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{imperial-kochmar-2023-basahacorpus,
    title = "{B}asaha{C}orpus: An Expanded Linguistic Resource for Readability Assessment in {C}entral {P}hilippine Languages",
    author = "Imperial, Joseph Marvin  and
      Kochmar, Ekaterina",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.388",
    doi = "10.18653/v1/2023.emnlp-main.388",
    pages = "6302--6309",
    }
"""

_DATASETNAME = "basaha_corpus"

_DESCRIPTION = """
BasahaCorpus contains short stories in four Central Philippine languages \
    (Minasbate, Rinconada, Kinaray-a, and Hiligaynon) for low-resource \
    readability assessment. Each dataset per language contains stories \
    distributed over the first three grade levels (L1, L2, and L3) in \
    the Philippine education context. The grade levels of the dataset \
    have been provided by an expert from Let's Read Asia.
"""
_HOMEPAGE = "https://github.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA"

_LANGUAGES = [
    "msb",
    "rin",
    "kar",
    "hil",
]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_LOCAL = False

_URLS = {
    # Minasbate, Rinconada, Kinaray-a, and Hiligaynon (from the _DESCRIPTION)
    "msb": "https://raw.githubusercontent.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA/main/data/features/min_features.csv",
    "rin": "https://raw.githubusercontent.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA/main/data/features/rin_features.csv",
    "kar": "https://raw.githubusercontent.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA/main/data/features/kar_features.csv",
    "hil": "https://raw.githubusercontent.com/imperialite/BasahaCorpus-HierarchicalCrosslingualARA/main/data/features/hil_features.csv",
}

_SUPPORTED_TASKS = [Tasks.READABILITY_ASSESSMENT]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class BasahaCorpusDataset(datasets.GeneratorBasedBuilder):
    """
    BasahaCorpus comprises short stories in four Central Philippine
    languages (Minasbate, Rinconada, Kinaray-a, and Hiligaynon)
    for low-resource readability assessment. Each language dataset
    includes stories from the first three grade levels (L1, L2, and L3)
    in the Philippine education context, as classified by an expert
    from Let's Read Asia.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [SEACrowdConfig(name=f"{_DATASETNAME}_{lang}_source", version=datasets.Version(_SOURCE_VERSION), description=f"{_DATASETNAME} source schema", schema="source", subset_id=f"{_DATASETNAME}_{lang}",) for lang in _LANGUAGES] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}_{lang}",
        )
        for lang in _LANGUAGES
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_msb_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "book_title": datasets.Value("string"),
                    "word_count": datasets.Value("int64"),
                    "sentence_count": datasets.Value("int64"),
                    "phrase_count_per_sentence": datasets.Value("float64"),
                    "average_word_len": datasets.Value("float64"),
                    "average_sentence_len": datasets.Value("float64"),
                    "average_syllable_count": datasets.Value("float64"),
                    "polysyll_count": datasets.Value("int64"),
                    "consonant_cluster_density": datasets.Value("float64"),
                    "v_density": datasets.Value("float64"),
                    "cv_density": datasets.Value("float64"),
                    "vc_density": datasets.Value("float64"),
                    "cvc_density": datasets.Value("float64"),
                    "vcc_density": datasets.Value("float64"),
                    "cvcc_density": datasets.Value("float64"),
                    "ccvc_density": datasets.Value("float64"),
                    "ccv_density": datasets.Value("float64"),
                    "ccvcc_density": datasets.Value("float64"),
                    "ccvccc_density": datasets.Value("float64"),
                    "tag_bigram_sim": datasets.Value("float64"),
                    "bik_bigram_sim": datasets.Value("float64"),
                    "ceb_bigram_sim": datasets.Value("float64"),
                    "hil_bigram_sim": datasets.Value("float64"),
                    "rin_bigram_sim": datasets.Value("float64"),
                    "min_bigram_sim": datasets.Value("float64"),
                    "kar_bigram_sim": datasets.Value("float64"),
                    "tag_trigram_sim": datasets.Value("float64"),
                    "bik_trigram_sim": datasets.Value("float64"),
                    "ceb_trigam_sim": datasets.Value("float64"),
                    "hil_trigam_sim": datasets.Value("float64"),
                    "rin_trigam_sim": datasets.Value("float64"),
                    "min_trigam_sim": datasets.Value("float64"),
                    "kar_trigam_sim": datasets.Value("float64"),
                    "grade_level": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(["1", "2", "3"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        lang = self.config.name.split("_")[2]

        if lang in _LANGUAGES:
            data_path = Path(dl_manager.download_and_extract(_URLS[lang]))
        else:
            data_path = [Path(dl_manager.download_and_extract(_URLS[lang])) for lang in _LANGUAGES]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = pd.read_csv(filepath, index_col=None)

        for index, row in df.iterrows():

            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == "seacrowd_text":

                example = {
                    "id": str(index),
                    "text": str(row["book_title"]),
                    "label": str(row["grade_level"]),
                }

            yield index, example
