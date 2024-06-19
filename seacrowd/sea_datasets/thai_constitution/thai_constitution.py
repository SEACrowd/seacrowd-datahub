from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (DEFAULT_SEACROWD_VIEW_NAME,
                                       DEFAULT_SOURCE_VIEW_NAME, Tasks, Licenses, TASK_TO_SCHEMA)

_DATASETNAME = "thai_constitution"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME

_CITATION = """\
@misc{
   thaiconstitution, 
   title={Thai Constitution Corpus}, 
   url={https://github.com/PyThaiNLP/Thai-constitution-corpus},
   author={Wannaphong Phatthiyaphaibun}
} 
"""

_LOCAL = False

_DESCRIPTION = """\
Thailand's constitutional archive since 1932
- Data collected from Office of the Council of State
- This project is part of the development plan PyThaiNLP
- The information collected in this text archive is in the public domain according to the Copyright Act 1994, Section 7 (The following are not considered copyrighted works under this Act: (1) daily news and Various facts which has the nature of being only news and not work in the literature department Science department or art department [...] (3) regulations, announcements, orders, clarifications and correspondence of ministries, bureaus, departments or any other government or local agencies [...])
"""

_HOMEPAGE = "https://github.com/PyThaiNLP/Thai-constitution-corpus/tree/master"

_LICENSE = Licenses.CC0_1_0.value

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SEACROWD_SCHEMA_NAME = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()
_LANGUAGES = ['tha']
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_URLS = [
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%9B%E0%B8%81%E0%B8%84%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%AD%E0%B8%B2%E0%B8%93%E0%B8%B2%E0%B8%88%E0%B8%B1%E0%B8%81%E0%B8%A3%20%E0%B8%9E%E0%B8%B8%E0%B8%97%E0%B8%98%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B8%B2%E0%B8%8A%202502.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%9B%E0%B8%81%E0%B8%84%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%AD%E0%B8%B2%E0%B8%93%E0%B8%B2%E0%B8%88%E0%B8%B1%E0%B8%81%E0%B8%A3%20%E0%B8%9E%E0%B8%B8%E0%B8%97%E0%B8%98%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B8%B2%E0%B8%8A%202515.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%9B%E0%B8%81%E0%B8%84%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%AD%E0%B8%B2%E0%B8%93%E0%B8%B2%E0%B8%88%E0%B8%B1%E0%B8%81%E0%B8%A3%20%E0%B8%9E%E0%B8%B8%E0%B8%97%E0%B8%98%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B8%B2%E0%B8%8A%202520.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%9B%E0%B8%81%E0%B8%84%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%AD%E0%B8%B2%E0%B8%93%E0%B8%B2%E0%B8%88%E0%B8%B1%E0%B8%81%E0%B8%A3%20%E0%B8%9E%E0%B8%B8%E0%B8%97%E0%B8%98%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B8%B2%E0%B8%8A%202534.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%9E%E0%B8%A3%E0%B8%B0%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%9A%E0%B8%B1%E0%B8%8D%E0%B8%8D%E0%B8%B1%E0%B8%95%E0%B8%B4%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%9B%E0%B8%81%E0%B8%84%E0%B8%A3%E0%B8%AD%E0%B8%87%E0%B9%81%E0%B8%9C%E0%B9%88%E0%B8%99%E0%B8%94%E0%B8%B4%E0%B8%99%E0%B8%AA%E0%B8%A2%E0%B8%B2%E0%B8%A1%E0%B8%8A%E0%B8%B1%E0%B9%88%E0%B8%A7%E0%B8%84%E0%B8%A3%E0%B8%B2%E0%B8%A72475.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2475.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2489.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2492.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2511.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2517.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2519.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2521.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2534.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2540.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2550.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D2560.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D%E0%B8%89%E0%B8%B0%E0%B8%9A%E0%B8%B1%E0%B8%9A%E0%B8%8A%E0%B8%B1%E0%B9%88%E0%B8%A7%E0%B8%84%E0%B8%A3%E0%B8%B2%E0%B8%A72490.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D%E0%B9%81%E0%B8%AB%E0%B9%88%E0%B8%87%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%AD%E0%B8%B2%E0%B8%93%E0%B8%B2%E0%B8%88%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B9%84%E0%B8%97%E0%B8%A2%20(%E0%B8%89%E0%B8%9A%E0%B8%B1%E0%B8%9A%E0%B8%8A%E0%B8%B1%E0%B9%88%E0%B8%A7%E0%B8%84%E0%B8%A3%E0%B8%B2%E0%B8%A7)%20%E0%B8%9E%E0%B8%B8%E0%B8%97%E0%B8%98%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B8%B2%E0%B8%8A%202549.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D%E0%B9%81%E0%B8%AB%E0%B9%88%E0%B8%87%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%AD%E0%B8%B2%E0%B8%93%E0%B8%B2%E0%B8%88%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B9%84%E0%B8%97%E0%B8%A2%20(%E0%B8%89%E0%B8%9A%E0%B8%B1%E0%B8%9A%E0%B8%8A%E0%B8%B1%E0%B9%88%E0%B8%A7%E0%B8%84%E0%B8%A3%E0%B8%B2%E0%B8%A7)%20%E0%B8%9E%E0%B8%B8%E0%B8%97%E0%B8%98%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B8%B2%E0%B8%8A%202557.txt",
    "https://raw.githubusercontent.com/PyThaiNLP/Thai-constitution-corpus/master/data/%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%98%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%99%E0%B8%B9%E0%B8%8D%E0%B9%81%E0%B8%AB%E0%B9%88%E0%B8%87%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%AD%E0%B8%B2%E0%B8%93%E0%B8%B2%E0%B8%88%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B9%84%E0%B8%97%E0%B8%A2%20%E0%B8%9E%E0%B8%B8%E0%B8%97%E0%B8%98%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B8%B2%E0%B8%8A%202475%20%E0%B9%81%E0%B8%81%E0%B9%89%E0%B9%84%E0%B8%82%E0%B9%80%E0%B8%9E%E0%B8%B4%E0%B9%88%E0%B8%A1%E0%B9%80%E0%B8%95%E0%B8%B4%E0%B8%A1%20%E0%B8%9E%E0%B8%B8%E0%B8%97%E0%B8%98%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%A3%E0%B8%B2%E0%B8%8A%202495.txt"
    ]


class ThaiConstitutionDataset(datasets.GeneratorBasedBuilder):
    """Thailand's constitutional archive since 1932"""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="thai_constitution_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="Thai constitution source schema",
            schema="source",
            subset_id="thai_constitution",
        ),
        SEACrowdConfig(
            name=f"thai_constitution_seacrowd_{_SEACROWD_SCHEMA_NAME}",
            version=datasets.Version(_SEACROWD_VERSION),
            description="Thai constitution SEACrowd schema",
            schema=f"seacrowd_{_SEACROWD_SCHEMA_NAME}",
            subset_id="thai_constitution",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.schema == f"seacrowd_{_SEACROWD_SCHEMA_NAME}":
            features = schemas.self_supervised_pretraining.features
        else:
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        filepaths = [Path(dl_manager.download(url)) for url in _URLS]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": filepaths},
            ),
        ]

    def _generate_examples(self, filepaths: List[Path]) -> Tuple[int, Dict]:
        counter = 0
        for path in filepaths:
            with open(path, encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == "":
                        continue

                    if self.config.schema == "source":
                        yield (
                            counter,
                            {
                                "id": str(counter),
                                "text": line.strip(),
                            },
                        )
                    elif self.config.schema == f"seacrowd_{_SEACROWD_SCHEMA_NAME}":
                        yield (
                            counter,
                            {
                                "id": str(counter),
                                "text": line.strip(),
                            },
                        )

                    counter += 1     
