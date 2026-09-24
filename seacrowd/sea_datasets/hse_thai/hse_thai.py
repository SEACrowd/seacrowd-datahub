import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@misc{HSEThaiCorpus,
  title        = {HSE Thai Corpus},
  author       = {Conneau, Alexis and Rinott, Ruty and Lample, Guillaume and Williams, Adina and Bowman, Samuel R. and Schwenk, Holger and Stoyanov, Veselin},
  organization = {HSE School of Linguistics},
  year         = {2024},
  howpublished = {\\url{http://web-corpora.net/ThaiCorpus/search}},
  note         = {Accessed on 2024-05-15}
}
"""

_DATASETNAME = "hse_thai"

_DESCRIPTION = """\
HSE Thai Corpus is a corpus of modern texts written in Thai language. The texts, containing in whole 50 million tokens,
were collected from various Thai websites (mostly news websites). To make it easier for non-Thai-speakers to comprehend and use texts in the corpus the researchers decided to separate words in each sentence with spaces.

The data for the corpus was collected by means of Scrapy. To tokenize texts the Pythai module was used. The text in this dataset is encoded in UTF-8.
This dataset contains text from two sources: Wikipedia and thaigov.go.th.

The former is licensed under a standard Wikipedia license, and the latter under an Open Government License for Thailand.
"""

_HOMEPAGE = "http://web-corpora.net/ThaiCorpus/search/"

_LANGUAGES = ["tha"]

_LICENSE = Licenses.OTHERS.value

_LOCAL = False

_URLS = "https://github.com/khelli07/hse-thai-for-seacrowd/raw/master/texts_tagged.zip"

_SUPPORTED_TASKS = [Tasks.LANGUAGE_IDENTIFICATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class HSEThaiDataset(datasets.GeneratorBasedBuilder):
    """Modern Thai corpus taken from http://web-corpora.net/ThaiCorpus/search/"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA_NAME = "text"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        features = schemas.text_features(_LANGUAGES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLS)
        data_dir = os.path.join(data_dir, "texts_tagged_200")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        base_path = filepath
        folders = os.listdir(base_path)

        id_ = 0
        for folder in folders:
            files = os.listdir(os.path.join(base_path, folder))
            for file in files[3:]:
                filepath = os.path.join(base_path, folder, file)
                if filepath.endswith(".xml"):
                    root = ET.parse(filepath).getroot()
                    sentences = root.findall(".//se")
                    for sentence in sentences:
                        words = sentence.findall("w")
                        article = " ".join([word.text for word in words])
                        id_ += 1
                        yield id_, {
                            "id": str(id_),
                            "text": article,
                            "label": "tha",
                        }
