# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PFSA-ID is an annotated corpus for Public Figure Statement Attribution in the Indonesian Language.
The annotation using the multi-class named entity recognition with 11 labels: PERSON, ROLE, AFFILIATION, PERSONCOREF, CUE, CUECOREF, STATEMENT, ISSUE, EVENT, DATETIME, and LOCATION and using the BILOU scheme as the representation of tokens.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{PURNOMOWP2022,
  title = {PFSA-ID: an annotated Indonesian corpus and baseline model of public figures statements attributions},
  journal = {Global Knowledge, Memory and Communication},
  volume = {ahead-of-print},
  pages = {ahead-of-print},
  year = {2022},
  issn = {2514-9342},
  doi = {https://doi.org/10.1108/GKMC-04-2022-0091},
  url = {https://www.emerald.com/insight/content/doi/10.1108/GKMC-04-2022-0091/full/html},
  author = {Yohanes Sigit {Purnomo W.P.} and Yogan Jaya Kumar and Nur Zareen Zulkarnain},
  keywords = {Indonesian corpus, Public figures, Statement attribution, News article, Baseline model, Named entity recognition},
  abstract = {Purpose By far, the corpus for the quotation extraction and quotation attribution tasks in Indonesian is still limited in quantity and depth. This study aims to develop an Indonesian corpus of public figure statements attributions and a
    baseline model for attribution extraction, so it will contribute to fostering research in information extraction for the Indonesian language. Design/methodology/approach The methodology is divided into corpus development and extraction model
    development. During corpus development, data were collected and annotated. The development of the extraction model entails feature extraction, the definition of the model architecture, parameter selection and configuration,
    model training and evaluation, as well as model selection. Findings The Indonesian corpus of public figure statements attribution achieved 90.06% agreement level between the annotator and experts and could serve as a gold standard corpus.
    Furthermore, the baseline model predicted most labels and achieved 82.026% F-score. Originality/value To the best of the authors’ knowledge, the resulting corpus is the first corpus for attribution of public figures’ statements in the Indonesian
    language, which makes it a significant step for research on attribution extraction in the language. The resulting corpus and the baseline model can be used as a benchmark for further research. Other researchers could follow the methods presented
    in this paper to develop a new corpus and baseline model for other languages.
    }
}

@article{PurnomoWP2024,
  title = {Extraction and attribution of public figures statements for journalism in Indonesia using deep learning},
  volume = {289},
  ISSN = {0950-7051},
  url = {http://dx.doi.org/10.1016/j.knosys.2024.111558},
  DOI = {10.1016/j.knosys.2024.111558},
  journal = {Knowledge-Based Systems},
  publisher = {Elsevier BV},
  author = {Purnomo W.P.,  Yohanes Sigit and Kumar,  Yogan Jaya and Zulkarnain,  Nur Zareen and Raza,  Basit},
  year = {2024},
  month = apr,
  pages = {111558}
}
"""

_DATASETNAME = "pfsa_id"

_DESCRIPTION = """\
PFSA-ID is an annotated corpus for Public Figure Statement Attribution in the Indonesian Language.

The annotation using the multi-class named entity recognition with 11 labels: PERSON, ROLE, AFFILIATION, PERSONCOREF, CUE, CUECOREF, STATEMENT, ISSUE, EVENT, DATETIME, and LOCATION and using the BILOU scheme as the representation of tokens.
"""

_HOMEPAGE = "https://github.com/sigit-purnomo/pfsa-id"

_LANGUAGES = ["ind"]

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value

_LOCAL = False

_URLS = {
    "pfsa_id": {
        "train": [
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id/train-60.txt",
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id/train-70.txt",
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id/train-80.txt",
        ],
        "dev": [
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id/dev-20.txt",
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id/dev-30.txt",
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id/dev-40.txt",
        ],
    },
    "pfsa_id_med": {
        "train": [
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id-med/train-dl-60-v2.txt",
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id-med/train-dl-70-v2.txt",
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id-med/train-dl-80-v2.txt",
        ],
        "dev": [
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id-med/dev-dl-20-v2.txt",
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id-med/dev-dl-30-v2.txt",
            "https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id-med/dev-dl-40-v2.txt",
        ],
    },
    "pfsa_id_test": {
        "train": ["https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id-test/train.txt"],
        "dev": ["https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id-test/dev.txt"],
        "test": ["https://raw.githubusercontent.com/sigit-purnomo/pfsa-id-dl/main/corpus/pfsa-id-test/test-data.txt"],
    },
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class PfsaIdDataset(datasets.GeneratorBasedBuilder):
    """PFSA-ID is an annotated corpus for Public Figure Statement Attribution in the Indonesian Language.

    The annotation using the multi-class named entity recognition with 11 labels: PERSON, ROLE, AFFILIATION, PERSONCOREF, CUE, CUECOREF, STATEMENT, ISSUE, EVENT, DATETIME, and LOCATION and using the BILOU scheme as the representation of tokens."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    LABEL_CLASSES = [
        "L-EVENT",
        "U-EVENT",
        "B-EVENT",
        "B-STATEMENT",
        "L-DATETIME",
        "I-ROLE",
        "B-DATETIME",
        "I-LOCATION",
        "I-STATEMENT",
        "B-PERSON",
        "U-AFFILIATION",
        "I-PERSONCOREF",
        "B-ROLE",
        "I-EVENT",
        "B-ISSUE",
        "B-AFFILIATION",
        "I-AFFILIATION",
        "U-PERSON",
        "U-ROLE",
        "B-LOCATION",
        "L-LOCATION",
        "L-ROLE",
        "U-CUE",
        "I-DATETIME",
        "L-PERSONCOREF",
        "L-AFFILIATION",
        "B-PERSONCOREF",
        "O",
        "I-ISSUE",
        "L-PERSON",
        "U-CUECOREF",
        "U-LOCATION",
        "L-STATEMENT",
        "U-PERSONCOREF",
        "I-PERSON",
        "L-ISSUE",
    ]

    SUBSETS = ["", "_med", "_test"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema for pfsa_id{subset} subset",
            schema="source",
            subset_id=f"{_DATASETNAME}{subset}",
        )
        for subset in SUBSETS
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}{subset}_seacrowd_seq_label",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd seq_label schema for pfsa_id{subset} subset",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}{subset}",
        )
        for subset in SUBSETS
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(datasets.ClassLabel(names=self.LABEL_CLASSES)),
                }
            )

        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(self.LABEL_CLASSES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[self.config.subset_id]
        data_dir = dl_manager.download_and_extract(urls)

        split_gen = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["dev"],
                },
            ),
        ]

        if self.config.subset_id == "pfsa_id_test":
            split_gen.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": data_dir["test"],
                    },
                )
            )

        return split_gen

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        i = 0
        for _, path in enumerate(filepath):
            tokens, labels = [], []
            lines = open(path, "r").readlines()

            for line in lines:
                if line == "\n":
                    yield i, {
                        "id": str(i),
                        "tokens": tokens,
                        "labels": labels,
                    }
                    i += 1
                    tokens = []
                    labels = []
                else:
                    token, label = line.split("\t")
                    tokens.append(token)
                    labels.append(label.strip("\n"))
