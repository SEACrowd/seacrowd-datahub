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
The khPOS Corpus (Khmer POS Corpus) is a 12,000 sentences (25,626 words) manually word segmented and POS tagged corpus 
developed for Khmer language NLP research and developments. We collected Khmer sentences from websites that include
various area such as economics, news, politics. Moreover it is also contained some student list and voter list of 
national election committee of Cambodia. The average number of words per sentence in the whole corpus is 10.75. 
Here, some symbols such as "។" (Khmer sign Khan), "៖" (Khmer sign Camnuc pii kuuh), "-", "?", "[", "]" etc. also 
counted as words. The shortest sentence contained only 1 word and longest sentence contained 169 words. This dataset contains
A validation set and a test set, each containing 1000 sentences.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@inproceedings{kyaw2017comparison,
  title={Comparison of Six POS Tagging Methods on 12K Sentences Khmer Language POS Tagged Corpus},
  author={Ye Kyaw Thu and Vichet Chea and Yoshinori Sagisaka},
  booktitle={Proceedings of the first Regional Conference on Optical character recognition and Natural language processing technologies for ASEAN languages (ONA 2017)},
  year={2017},
  month={December 7-8},
  address={Phnom Penh, Cambodia}
}
"""

_DATASETNAME = "khpos"

_DESCRIPTION = """\
The khPOS Corpus (Khmer POS Corpus) is a 12,000 sentences (25,626 words) manually word segmented and POS tagged corpus 
developed for Khmer language NLP research and developments. We collected Khmer sentences from websites that include
various area such as economics, news, politics. Moreover it is also contained some student list and voter list of 
national election committee of Cambodia. The average number of words per sentence in the whole corpus is 10.75. 
Here, some symbols such as "។" (Khmer sign Khan), "៖" (Khmer sign Camnuc pii kuuh), "-", "?", "[", "]" etc. also 
counted as words. The shortest sentence contained only 1 word and longest sentence contained 169 words. This dataset contains
A validation set and a test set, each containing 1000 sentences.
"""

_HOMEPAGE = "https://github.com/ye-kyaw-thu/khPOS/tree/master"

_LANGUAGES = ['khm']  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value 

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        'train': "https://raw.githubusercontent.com/ye-kyaw-thu/khPOS/master/corpus-draft-ver-1.0/data/after-replace/train.all2",
        'validation': "https://raw.githubusercontent.com/ye-kyaw-thu/khPOS/master/corpus-draft-ver-1.0/data/OPEN-TEST",
        'test': "https://raw.githubusercontent.com/ye-kyaw-thu/khPOS/master/corpus-draft-ver-1.0/data/CLOSE-TEST"
    }
}

_SUPPORTED_TASKS = [Tasks.POS_TAGGING]  

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class KhPOS(datasets.GeneratorBasedBuilder):
    """\
This datasets contain 12000 sentences (25626 words) for the Khmer language.
There are 24 POS tags and their description can be found at https://github.com/ye-kyaw-thu/khPOS/tree/master.
The used Khmer Tokenizer can be found in the above github repository as well. This dataset contains
A validation set and a test set, each containing 1000 sentences.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="khpos_source",
            version=SOURCE_VERSION,
            description="khpos source schema",
            schema="source",
            subset_id="khpos",
        ),
        SEACrowdConfig(
            name="khpos_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="khpos SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id="khpos",
        ),
    ]

    DEFAULT_CONFIG_NAME = "khpos_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({
                "id"      : datasets.Value("string"),
                "tokens"  : datasets.Sequence(datasets.Value("string")),
                #pos_tags follows order from corpus-draft-ver-1.0/data/after-replace/train.all2.tag.freq
                "pos_tags": datasets.Sequence(datasets.features.ClassLabel(
                    names = [
                        'AB', 'AUX', 'CC', 'CD',
                        'DBL', 'DT', 'ETC', 'IN',
                        'JJ', 'KAN', 'M', 'NN',
                        'PA', 'PN', 'PRO', 'QT',
                        'RB', 'RPN', 'SYM', 'UH',
                        'VB', 'VB_JJ', 'VCOM'
                    ]
                ))
            })
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label.features([
                        'AB', 'AUX', 'CC', 'CD',
                        'DBL', 'DT', 'ETC', 'IN',
                        'JJ', 'KAN', 'M', 'NN',
                        'PA', 'PN', 'PRO', 'QT',
                        'RB', 'RPN', 'SYM', 'UH',
                        'VB', 'VB_JJ', 'VCOM'
                    ])

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]['train']
        path = dl_manager.download_and_extract(urls)

        dev_url  = _URLS[_DATASETNAME]['validation']
        dev_path = dl_manager.download_and_extract(dev_url)

        test_url  = _URLS[_DATASETNAME]['test']
        test_path = dl_manager.download_and_extract(test_url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dev_path,
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_path,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding="utf-8") as file:
            counter = 0
            for line in file:
                if line.strip() != "":
                    groups = line.split(" ")
                    tokens = []
                    pos_tags = []
                    for group in groups:
                        token, pos_tag = group.split("/")
                        tokens.append(token)
                        pos_tags.append(pos_tag)
                    if self.config.schema == "source":
                        yield (
                            counter,
                            {
                                "id"      : str(counter),
                                "tokens"  : tokens,
                                "pos_tags": pos_tags
                            }
                        )
                        counter += 1
                    elif self.config.schema == "seacrowd_seq_label":
                        yield (
                            counter,
                            {
                                "id"    : str(counter),
                                "tokens": tokens,
                                "labels": pos_tags
                            }
                        )
                        counter += 1
