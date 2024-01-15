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
Thai NNER Dataset
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.common_parser import load_conll_data
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses


_CITATION = """\
@inproceedings{buaphet-etal-2022-thai,
    title = "{T}hai Nested Named Entity Recognition Corpus",
    author = "Buaphet, Weerayut  and
      Udomcharoenchaikit, Can  and
      Limkonchotiwat, Peerat  and
      Rutherford, Attapol  and
      Nutanong, Sarana",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.116",
    doi = "10.18653/v1/2022.findings-acl.116",
    pages = "1473--1486",
    abstract = "This paper presents the first Thai Nested Named Entity Recognition (N-NER) dataset. Thai N-NER consists of 264,798 mentions, 104 classes, and a maximum depth of 8 layers obtained from 4,894 documents in the domains of news articles and restaurant reviews. Our work, to the best of our knowledge, presents the largest non-English N-NER dataset and the first non-English one with fine-grained classes. To understand the new challenges our proposed dataset brings to the field, we conduct an experimental study on (i) cutting edge N-NER models with the state-of-the-art accuracy in English and (ii) baseline methods based on well-known language model architectures. From the experimental results, we obtained two key findings. First, all models produced poor F1 scores in the tail region of the class distribution. There is little or no performance improvement provided by these models with respect to the baseline methods with our Thai dataset. These findings suggest that further investigation is required to make a multilingual N-NER solution that works well across different languages.",
}
"""


_DATASETNAME = "ThaiNNER"


_DESCRIPTION = """\
Thai N-NER consists of 264,798 mentions, 104 classes, and a maximum depth of 8 layers obtained from 4,894 documents in the domains of news articles and restaurant reviews. 
To create the dataset, the authors gather 4,894 documents from two different domains: news articles and restaurant reviews. In particular, we obtain 4,396 news articles from Prachathai, a news website, and 498 restaurant reviews from Wongnai, a crowd-sourced restaurant review platform.

"""


_HOMEPAGE = "https://github.com/vistec-AI/Thai-NNER"


_LANGUAGES = ["tha"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)


_LICENSE = Licenses.CC_BY_SA_3_0.value


_LOCAL = False


_URLS = {
    "train": "https://drive.google.com/uc?export=download&id=1FNJ0Ylcq27SowSiMEIRzR--45Sz7dsGw",
    "test": "https://drive.google.com/uc?export=download&id=1FSM68Jh2Am4WByxrzzz4Cpou0J9Uok7Y"
}


_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class ThaiNnerDataset(datasets.GeneratorBasedBuilder):
    """Thai Nested Named Entity Recognition (N-NER) dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    label_classes = ['O', 'S-org:religious', 'E-tv_show', 'I-soi', 'E-last', 'E-fund', 'B-country', 'B-port', 'B-airport', 'I-title', 'E-stock_exchange', 'I-restaurant', 'I-station', 'I-org:political', 'I-animal_species', 'I-index', 'E-latitude', 'B-law', 'S-middle', 'I-product:food', 'E-animal_species', 'I-goverment', 'S-city', 'E-longtitude', 'I-song', 'I-award', 'E-event:others', 'B-jargon', 'S-facility:other', 'S-money', 'I-stadium', 'E-space', 'S-animate', 'E-station', 'S-percent', 'E-country', 'I-year', 'S-district', 'E-org:religious', 'E-continent', 'I-food:ingredient', 'I-person', 'B-electronics', 'B-museum', 'B-product:drug', 'B-month', 'B-loc:others', 'B-org:religious', 'S-season', 'I-continent', 'E-norp:political', 'E-airport', 'B-money', 'S-role', 'B-cardinal', 'B-film', 'B-hotel', 'B-unit', 'E-energy', 'I-jargon', 'I-date', 'S-norp:others', 'S-country', 'E-film', 'I-energy', 'B-norp:political', 'E-index', 'I-concert', 'I-sciname', 'I-product:drug', 'I-org:other', 'S-animal_species', 'I-hospital', 'B-nationality', 'S-law', 'S-disease', 'I-latitude', 'E-award', 'B-periodic', 'I-roadname', 'S-jargon', 'E-city', 'S-person', 'S-date', 'E-goverment', 'B-stadium', 'S-religion', 'B-address', 'E-band', 'S-continent', 'I-fold', 'E-stadium', 'S-band', 'E-middle', 'B-org:edu', 'I-distance', 'I-island', 'S-hospital', 'S-loc:others', 'S-book', 'I-war', 'I-norp:political', 'E-title', 'S-war', 'S-org:edu', 'I-nationality', 'E-restaurant', 'S-province', 'E-war', 'I-quantity', 'S-orgcorp', 'E-person', 'E-day', 'E-disease', 'I-last', 'S-product:drug', 'S-space', 'E-product:drug', 'I-country', 'E-weight', 'B-fold', 'E-sciname', 'E-building', 'E-duration', 'B-longtitude', 'E-religion', 'S-soi', 'E-woa', 'B-date', 'I-city', 'I-role', 'E-org:edu', 'I-norp:others', 'B-title', 'I-namemod', 'E-speed', 'S-nicknametitle', 'E-mult', 'E-army', 'I-day', 'S-duration', 'B-time', 'I-rel', 'B-language', 'B-ocean', 'I-woa', 'E-animate', 'I-facility:other', 'E-psudoname', 'E-hotel', 'I-bridge', 'I-port', 'S-address', 'B-war', 'E-mountian', 'B-psudoname', 'I-sports_team', 'B-last', 'B-animate', 'B-latitude', 'E-cardinal', 'S-roadname', 'I-sub_district', 'E-food:ingredient', 'E-nickname', 'E-port', 'B-district', 'B-sports_team', 'E-org:political', 'I-loc:others', 'E-ocean', 'E-month', 'S-bridge', 'B-fund', 'I-money', 'I-state', 'S-unit', 'B-distance', 'B-food:ingredient', 'E-unit', 'S-namemod', 'S-airport', 'I-cardinal', 'S-sub_district', 'S-day', 'B-woa', 'S-army', 'B-tv_show', 'E-bridge', 'E-loc:others', 'S-sciname', 'I-law', 'B-space', 'B-religion', 'I-film', 'E-quantity', 'S-org:other', 'I-sports_event', 'I-event:others', 'S-firstname', 'I-band', 'I-weight', 'E-song', 'I-book', 'E-sports_event', 'S-weapon', 'I-space', 'B-speed', 'B-event:others', 'E-soi', 'I-org:edu', 'E-product:food', 'S-year', 'B-concert', 'B-province', 'B-sciname', 'I-vehicle', 'E-percent', 'I-percent', 'E-org:other', 'I-orgcorp', 'B-org:other', 'E-year', 'I-media', 'S-food:ingredient', 'I-electronics', 'E-hospital', 'I-psudoname', 'I-month', 'E-distance', 'B-duration', 'S-nationality', 'I-firstname', 'B-namemod', 'S-building', 'S-psudoname', 'B-restaurant', 'I-org:religious', 'B-vehicle', 'B-sports_event', 'B-book', 'B-temperature', 'E-norp:others', 'E-time', 'B-org:political', 'E-state', 'I-periodic', 'S-tv_show', 'S-product:food', 'S-postcode', 'S-mult', 'I-disease', 'S-month', 'B-weight', 'B-station', 'E-book', 'E-sub_district', 'E-orgcorp', 'S-fund', 'I-nickname', 'B-band', 'S-stock_exchange', 'S-natural_disaster', 'S-electronics', 'E-sports_team', 'E-museum', 'I-fund', 'B-river', 'B-rel', 'I-unit', 'I-time', 'S-mountian', 'S-port', 'I-temperature', 'I-airport', 'S-god', 'B-year', 'I-army', 'B-index', 'B-natural_disaster', 'E-roadname', 'E-namemod', 'S-game', 'I-hotel', 'B-quantity', 'B-person', 'B-mountian', 'S-river', 'E-vehicle', 'S-media', 'S-sports_team', 'I-river', 'I-mountian', 'B-hospital', 'S-state', 'B-mult', 'S-film', 'B-soi', 'E-role', 'S-org:political', 'S-station', 'E-jargon', 'S-last', 'S-ocean', 'S-quantity', 'S-award', 'E-law', 'I-middle', 'E-firstname', 'E-facility:other', 'I-address', 'I-religion', 'S-time', 'E-periodic', 'S-vehicle', 'B-nickname', 'S-fold', 'B-role', 'E-rel', 'S-goverment', 'I-district', 'B-song', 'E-money', 'E-island', 'I-duration', 'E-electronics', 'B-sub_district', 'E-language', 'E-temperature', 'B-building', 'S-sports_event', 'E-concert', 'B-percent', 'S-island', 'I-museum', 'B-island', 'B-media', 'S-language', 'B-stock_exchange', 'S-title', 'E-natural_disaster', 'B-animal_species', 'B-facility:other', 'B-state', 'E-address', 'I-animate', 'S-stadium', 'I-province', 'B-city', 'B-army', 'I-weapon', 'B-middle', 'S-cardinal', 'E-date', 'S-event:others', 'S-periodic', 'I-speed', 'B-day', 'B-goverment', 'E-media', 'B-norp:others', 'S-nickname', 'E-district', 'I-natural_disaster', 'E-fold', 'I-language', 'S-rel', 'B-product:food', 'B-bridge', 'I-longtitude', 'B-orgcorp', 'B-award', 'E-nationality', 'E-weapon', 'E-province', 'I-tv_show', 'B-energy', 'I-building', 'B-roadname', 'B-continent', 'B-firstname', 'I-stock_exchange', 'S-norp:political', 'S-weight', 'S-restaurant', 'B-weapon', 'S-song', 'B-disease', 'E-river']
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="thai_nner_source",
            version=SOURCE_VERSION,
            description=" Thai NNER source schema",
            schema="source",
            subset_id="thai_nner",
        ),
        SEACrowdConfig(
            name="thai_nner_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="Thai NNER SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id="thai_nner",
        ),
    ]

    DEFAULT_CONFIG_NAME = "thai_nner_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"index": datasets.Value("string"), "tokens": [datasets.Value("string")],
                                          "ner_tag": [datasets.Value("string")]})

        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(self.label_classes)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        train_path = Path(dl_manager.download_and_extract(_URLS["train"]))
        test_path = Path(dl_manager.download_and_extract(_URLS["test"]))


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
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
            )
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        def _custom_conll_loader(file_path):
            # Read file
            data = open(file_path, "r", encoding="utf-8").readlines()

            # Prepare buffer
            dataset = []
            sentence, seq_label = [], []
            for line in data:
                if len(line.strip()) > 0:
                    # Handle lines with a single field
                    if "\t" in line:
                        token, label = line[:-1].split("\t")
                    else:
                        token, label = line.strip(), "O"  # Default label if not provided

                    sentence.append(token)
                    seq_label.append(label)
                else:
                    dataset.append({"sentence": sentence, "label": seq_label})
                    sentence = []
                    seq_label = []
            return dataset

        conll_dataset = _custom_conll_loader(file_path=filepath)

        if self.config.schema == "source":
            for index, row in enumerate(conll_dataset):
                ex = {"index": str(index), "tokens": row["sentence"], "ner_tag": row["label"]}
                yield index, ex
        elif self.config.schema == "seacrowd_seq_label":
            for index, row in enumerate(conll_dataset):
                ex = {"id": str(index), "tokens": row["sentence"], "labels": row["label"]}
                yield index, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")

# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


if __name__ == "__main__":
    datasets.load_dataset(__file__)