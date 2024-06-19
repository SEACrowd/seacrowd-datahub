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

"""\
This dataset is designed for named entity recognition (NER) tasks in the Bahasa Indonesia tourism domain. It contains labeled sequences of named entities, including locations, facilities, and tourism-related entities. The dataset is annotated with the following entity types:

    O (0)    : Non-entity or other words not falling into the specified categories.
    B-WIS (1): Beginning of a tourism-related entity.
    I-WIS (2): Continuation of a tourism-related entity.
    B-LOC (3): Beginning of a location entity.
    I-LOC (4): Continuation of a location entity.
    B-FAS (5): Beginning of a facility entity.
    I-FAS (6): Continuation of a facility entity.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@article{JLK,
    author = {Ahmad Hidayatullah and Muhammad Fakhri Despawida Aulia Putra and Adityo Permana Wibowo and Kartika Rizqi Nastiti},
    title = { Named Entity Recognition on Tourist Destinations Reviews in the Indonesian Language},
    journal = {Jurnal Linguistik Komputasional},
    volume = {6},
    number = {1},
    year = {2023},
    keywords = {},
    abstract = {To find information about tourist destinations, tourists usually search the reviews about the destinations they want to visit. However, many studies made it hard for them to see the desired information. Named Entity Recognition (NER) is one of the techniques to detect entities in a text. The objective of this research was to make a NER model using BiLSTM to detect and evaluate entities on tourism destination reviews. This research used 2010 reviews of several tourism destinations in Indonesia and chunked them into 116.564 tokens of words. Those tokens were labeled according to their categories: the name of the tourism destination, locations, and facilities. If the tokens could not be classified according to the existing categories, the tokens would be labeled as O (outside). The model has been tested and gives 94,3% as the maximum average of F1-Score.},
    issn = {2621-9336}, pages = {30--35},   doi = {10.26418/jlk.v6i1.89},
    url = {https://inacl.id/journal/index.php/jlk/article/view/89}
}
"""

_DATASETNAME = "indoner_tourism"

_DESCRIPTION = """\
This dataset is designed for named entity recognition (NER) tasks in the Bahasa Indonesia tourism domain. It contains labeled sequences of named entities, including locations, facilities, and tourism-related entities. The dataset is annotated with the following entity types:

    O (0)    : Non-entity or other words not falling into the specified categories.
    B-WIS (1): Beginning of a tourism-related entity.
    I-WIS (2): Continuation of a tourism-related entity.
    B-LOC (3): Beginning of a location entity.
    I-LOC (4): Continuation of a location entity.
    B-FAS (5): Beginning of a facility entity.
    I-FAS (6): Continuation of a facility entity.
"""

_HOMEPAGE = "https://github.com/fathanick/IndoNER-Tourism/tree/main"

_LANGUAGES = ['ind']  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.AFL_3_0.value 

_LOCAL = False

_URL = "https://raw.githubusercontent.com/fathanick/IndoNER-Tourism/main/ner_data.tsv"

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class IndoNERTourismDataset(datasets.GeneratorBasedBuilder):
    """\
This dataset is designed for named entity recognition (NER) tasks in the Bahasa Indonesia tourism domain. It contains labeled sequences of named entities, including locations, facilities, and tourism-related entities. The dataset is annotated with the following entity types:

    O (0)    : Non-entity or other words not falling into the specified categories.
    B-WIS (1): Beginning of a tourism-related entity.
    I-WIS (2): Continuation of a tourism-related entity.
    B-LOC (3): Beginning of a location entity.
    I-LOC (4): Continuation of a location entity.
    B-FAS (5): Beginning of a facility entity.
    I-FAS (6): Continuation of a facility entity.
"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="indoner_tourism source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="indoner_tourism SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    'tokens'  : datasets.Sequence(datasets.Value("string")),
                    'ner_tags': datasets.Sequence(
                                    datasets.ClassLabel(names=["O", "B-WIS", "I-WIS", "B-LOC", "I-LOC", "B-FAS", "I-FAS"])
                                ),
                }
            )
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label.features(["O", "B-WIS", "I-WIS", "B-LOC", "I-LOC", "B-FAS", "I-FAS"])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URL
        path = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        tokens = []
        ner_tags = []
        counter = 0
        with open(filepath, encoding="utf-8") as file:
            for line in file:
                # End of Sentence met
                if line.strip() == "":
                    if self.config.schema == "source":
                        yield counter, {'tokens': tokens, 'ner_tags': ner_tags}
                        counter += 1
                        tokens = []
                        ner_tags = []
                    elif self.config.schema == "seacrowd_seq_label":
                        yield counter, {'id': counter, 'tokens': tokens, 'labels': ner_tags}
                        counter += 1
                        tokens = []
                        ner_tags = []
                # Process until End of Sentence met
                elif len(line.split('\t')) == 2:
                    token, ner_tag = line.split('\t')
                    tokens.append(token.strip())
                    if ner_tag not in ["O", "B-WIS", "I-WIS", "B-LOC", "I-LOC", "B-FAS", "I-FAS"]:
                        if ner_tag[0] in ["B", "I"]:
                            if any(tag in ner_tag for tag in ["WIS", "LOC", "FAS"]):
                                if '_' in ner_tag:
                                    ner_tag = '-'.join(ner_tag.split('_'))
                    ner_tags.append(ner_tag.strip())
