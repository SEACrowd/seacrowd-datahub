import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = r"""
@inproceedings{kabra-etal-2023-multi,
    title = "Multi-lingual and Multi-cultural Figurative Language Understanding",
    author = "Kabra, Anubha  and
      Liu, Emmy  and
      Khanuja, Simran  and
      Aji, Alham Fikri  and
      Winata, Genta  and
      Cahyawijaya, Samuel  and
      Aremu, Anuoluwapo  and
      Ogayo, Perez  and
      Neubig, Graham",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.525",
    doi = "10.18653/v1/2023.findings-acl.525",
    pages = "8269--8284",
}
"""

_LOCAL = False
_LANGUAGES = ["ind", "jav", "sun"]
_DATASETNAME = "mabl"
_DESCRIPTION = r"""\
The MABL (Metaphors Across Borders and Languages) dataset is a collection of
6,366 figurative language expressions from seven languages, crafted to improve
multilingual models' understanding of figurative speech and its linguistic
variations. It was built by crowdsourcing native speakers to generate paired
metaphors that began with the same words but had different meanings, as well as
the literal interpretations of both phrases. Each expression was checked by
fluent speakers to ensure they were clear, appropriate, and followed the format,
discarding any that didn't meet these standards.
"""

_HOMEPAGE = "https://github.com/simran-khanuja/Multilingual-Fig-QA"
_LICENSE = Licenses.MIT.value
_URL = "https://raw.githubusercontent.com/simran-khanuja/Multilingual-Fig-QA/main/langdata/"

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


def iso3to2(lang: str) -> str:
    """Convert 3-letter ISO code to its 2-letter equivalent"""
    iso_map = {"ind": "id", "jav": "jv", "sun": "su"}
    return iso_map[lang]


class MABLDataset(datasets.GeneratorBasedBuilder):
    """MABL dataset by Liu et al (2023)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "qa"

    dataset_names = sorted([f"{_DATASETNAME}_{lang}" for lang in _LANGUAGES])
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

    # Add configuration that allows loading all languages at once.
    BUILDER_CONFIGS.extend(
        [
            # mabl_source
            SEACrowdConfig(
                name=f"{_DATASETNAME}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} source schema (all)",
                schema="source",
                subset_id=_DATASETNAME,
            ),
            # mabl_seacrowd_qa
            SEACrowdConfig(
                name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} SEACrowd schema (all)",
                schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
                subset_id=_DATASETNAME,
            ),
        ]
    )

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "startphrase": datasets.Value("string"),
                    "ending1": datasets.Value("string"),
                    "ending2": datasets.Value("string"),
                    "labels": datasets.Value("string"),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.qa_features
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        """Return SplitGenerators."""
        mabl_source_data = []
        languages = []

        lang = self.config.name.split("_")[1]
        if lang in _LANGUAGES:
            # Load data per language
            mabl_source_data.append(dl_manager.download_and_extract(_URL + f"{iso3to2(lang)}.csv"))
            languages.append(lang)
        else:
            # Load examples for all languages at once.
            # We run this block when mabl_source / mabl_seacrowd_qa was chosen.
            for lang in _LANGUAGES:
                mabl_source_data.append(dl_manager.download_and_extract(_URL + f"{iso3to2(lang)}.csv"))
                languages.append(lang)

        return [
            datasets.SplitGenerator(
                # The MABL paper mentions that due to the size of each subset,
                # they consider each split as a test set.
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": mabl_source_data,
                    "split": "test",
                    "languages": languages,
                },
            )
        ]

    def _generate_examples(self, filepaths: List[Path], split: str, languages: List[str]) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""

        startphrases = []
        endings1 = []
        endings2 = []
        labels = []

        for lang, filepath in zip(languages, filepaths):
            with open(filepath, encoding="utf-8") as f:
                csv_reader = csv.reader(f, delimiter=",")
                next(csv_reader, None)  # skip the headers
                for row in csv_reader:
                    # Unfortunately, the columns in the subfiles of the MABL
                    # dataset are inconsistent. For 'ind', it is [ending1,
                    # ending2, labels, startphrase].  But for 'jav' and 'sun',
                    # the labels and startphrase columns were switched.  Here,
                    # I'm just hard-coding the column names
                    if lang == "ind":
                        end1, end2, label, start = row
                    if lang == "jav" or lang == "sun":
                        end1, end2, start, label = row

                    startphrases.append(start)
                    endings1.append(end1)
                    endings2.append(end2)
                    labels.append(label)

        for idx, (start, end1, end2, label) in enumerate(zip(startphrases, endings1, endings2, labels)):
            if self.config.schema == "source":
                example = {
                    "id": str(idx),
                    "startphrase": start,
                    "ending1": end1,
                    "ending2": end2,
                    "labels": label,
                }
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                # Create QA-specific items
                choices = [end1, end2]
                answer = choices[int(label)]

                # MABL doesn't differentiate between question and context.
                # It only contains a startphrase. Given that, I put the
                # startphrase in question and kept the context blank.
                example = {
                    "id": str(idx),
                    "question_id": idx,
                    "document_id": idx,
                    "question": start,
                    "type": "multiple_choice",
                    "choices": choices,
                    "context": "",
                    "answer": [answer],
                    "meta": {},
                }

            yield idx, example
