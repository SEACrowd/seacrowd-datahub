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
A dataset of Indonesian News for Named-Entity Recognition task.
  This dataset re-annotated the dataset previously provided by Syaifudin & Nurwidyantoro (2016)
  (https://github.com/yusufsyaifudin/Indonesia-ner) with a more standardized NER tags.
  There are three subsets, namely train.txt, dev.txt, and test.txt.
  Each file consists of three columns which are Tokens, PoS Tag, and NER Tag respectively.
  The format is following CoNLL dataset. The NER tag use the IOB format.
  The PoS tag using UDPipe (http://ufal.mff.cuni.cz/udpipe),
  a pipeline for tokenization, tagging, lemmatization and dependency parsing
  whose model is trained on UD Treebanks.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{khairunnisa-etal-2020-towards,
    title = "Towards a Standardized Dataset on {I}ndonesian Named Entity Recognition",
    author = "Khairunnisa, Siti Oryza  and
      Imankulova, Aizhan  and
      Komachi, Mamoru",
    editor = "Shmueli, Boaz  and
      Huang, Yin Jou",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics
      and the 10th International Joint Conference on Natural Language Processing: Student Research Workshop",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.aacl-srw.10",
    pages = "64--71",
    abstract = "In recent years, named entity recognition (NER) tasks in the Indonesian language
    have undergone extensive development. There are only a few corpora for Indonesian NER;
    hence, recent Indonesian NER studies have used diverse datasets. Although an open dataset is available,
    it includes only approximately 2,000 sentences and contains inconsistent annotations,
    thereby preventing accurate training of NER models without reliance on pre-trained models.
    Therefore, we re-annotated the dataset and compared the two annotations{'} performance
    using the Bidirectional Long Short-Term Memory and Conditional Random Field (BiLSTM-CRF) approach.
    Fixing the annotation yielded a more consistent result for the organization tag and improved the prediction score
    by a large margin. Moreover, to take full advantage of pre-trained models, we compared different feature embeddings
    to determine their impact on the NER task for the Indonesian language.",
}
"""

_DATASETNAME = "idner_news_2k"

_DESCRIPTION = """\
A dataset of Indonesian News for Named-Entity Recognition task.
  This dataset re-annotated the dataset previously provided by Syaifudin & Nurwidyantoro (2016)
  (https://github.com/yusufsyaifudin/Indonesia-ner) with a more standardized NER tags.
  There are three subsets, namely train.txt, dev.txt, and test.txt.
  Each file consists of three columns which are Tokens, PoS Tag, and NER Tag respectively.
  The format is following CoNLL dataset. The NER tag use the IOB format.
  The PoS tag using UDPipe (http://ufal.mff.cuni.cz/udpipe),
  a pipeline for tokenization, tagging, lemmatization and dependency parsing
  whose model is trained on UD Treebanks.
"""

_HOMEPAGE = "https://github.com/khairunnisaor/idner-news-2k"

_LANGUAGES = ["ind"]

_LICENSE = Licenses.MIT.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "train": "https://raw.githubusercontent.com/khairunnisaor/idner-news-2k/main/train.txt",
        "dev": "https://raw.githubusercontent.com/khairunnisaor/idner-news-2k/main/dev.txt",
        "test": "https://raw.githubusercontent.com/khairunnisaor/idner-news-2k/main/test.txt",
    },
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class IdNerNews2kDataset(datasets.GeneratorBasedBuilder):
    """This dataset is designed for Named-Entity Recognition NLP task in Indonesian,
    consisting of train, dev, and test files in CoNLL format. The NER tag in IOB format."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "seq_label"

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
        NAMED_ENTITIES = ["B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER", "O"]
        POS_TAGS = ["PROPN", "AUX", "NUM", "NOUN", "ADP", "PRON", "VERB", "ADV", "ADJ", "PUNCT", "DET", "PART", "SCONJ", "CCONJ", "SYM", "X"]

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "pos_tags": datasets.Sequence(datasets.ClassLabel(names=POS_TAGS)),
                    "ner_tags": datasets.Sequence(datasets.ClassLabel(names=NAMED_ENTITIES)),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.seq_label.features(NAMED_ENTITIES)

        else:
            raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:

        urls = _URLS[_DATASETNAME]
        train_path = dl_manager.download_and_extract(urls["train"])
        dev_path = dl_manager.download_and_extract(urls["dev"])
        test_path = dl_manager.download_and_extract(urls["test"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_path,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dev_path,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        df = pd.read_csv(filepath, delimiter=" ", header=None, skip_blank_lines=False)
        if self.config.schema == "source":
            tokens, pos_tags, ner_tags = [], [], []

            for idx, row in df.iterrows():
                if pd.isnull(row[0]):
                    if tokens:
                        yield idx, {"id": idx, "tokens": tokens, "pos_tags": pos_tags, "ner_tags": ner_tags}
                        tokens, pos_tags, ner_tags = [], [], []
                else:
                    tokens.append(row[0])
                    pos_tags.append(row[1])
                    ner_tags.append(row[2])

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            tokens, ner_tags = [], []

            for idx, row in df.iterrows():
                if pd.isnull(row[0]):
                    if tokens:
                        yield idx, {"id": idx, "tokens": tokens, "labels": ner_tags}
                        tokens, ner_tags = [], []
                else:
                    tokens.append(row[0])
                    ner_tags.append(row[2])
        else:
            raise ValueError(f"Invalid config: {self.config.name}")