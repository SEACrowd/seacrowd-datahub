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
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re

import datasets
from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@article{sornlertlamvanich1999building,
  title={Building a Thai part-of-speech tagged corpus (ORCHID)},
  author={Sornlertlamvanich, Virach and Takahashi, Naoto and Isahara, Hitoshi},
  journal={Journal of the Acoustical Society of Japan (E)},
  volume={20},
  number={3},
  pages={189--198},
  year={1999},
  publisher={Acoustical Society of Japan}
}
"""

_DATASETNAME = "orchid_pos"

_DESCRIPTION = """\
The ORCHID corpus is a Thai part-of-speech (POS) tagged dataset, resulting from a collaboration between\
Japan's Communications Research Laboratory (CRL) and Thailand's National Electronics and Computer Technology\
Center (NECTEC). It is structured at three levels: paragraph, sentence, and word. The dataset incorporates a\
unique tagset designed for use in multi-lingual machine translation projects, and is tailored to address the\
challenges of Thai text, which lacks explicit word and sentence boundaries, punctuation, and inflection.\
This dataset includes text information along with numbering for retrieval, and employs a probabilistic trigram\
model for word segmentation and POS tagging. The ORCHID corpus is specifically structured to reduce ambiguity in\
POS assignments, making it a valuable resource for Thai language processing and computational linguistics research.
"""

_HOMEPAGE = "https://github.com/wannaphong/corpus_mirror/releases/tag/orchid-v1.0"

_LANGUAGES = ["tha"]

_LICENSE = Licenses.CC_BY_NC_SA_3_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://github.com/wannaphong/corpus_mirror/releases/download/orchid-v1.0/orchid97.crp.utf",
}

_SUPPORTED_TASKS = [Tasks.POS_TAGGING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"


class OrchidPOS(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "TTitle": datasets.Value("string"),
                    "ETitle": datasets.Value("string"),
                    "TAuthor": datasets.Value("string"),
                    "EAuthor": datasets.Value("string"),
                    "TInbook": datasets.Value("string"),
                    "EInbook": datasets.Value("string"),
                    "TPublisher": datasets.Value("string"),
                    "EPublisher": datasets.Value("string"),
                    "Page": datasets.Value("string"),
                    "Year": datasets.Value("string"),
                    "File": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_seq_label":
            label_names = [
                "NPRP",
                "NCNM",
                "NONM",
                "NLBL",
                "NCMN",
                "NTTL",
                "PPRS",
                "PDMN",
                "PNTR",
                "PREL",
                "VACT",
                "VSTA",
                "VATT",
                "XVBM",
                "XVAM",
                "XVMM",
                "XVBB",
                "XVAE",
                "DDAN",
                "DDAC",
                "DDBQ",
                "DDAQ",
                "DIAC",
                "DIBQ",
                "DIAQ",
                "DCNM",
                "DONM",
                "ADVN",
                "ADVI",
                "ADVP",
                "ADVS",
                "CNIT",
                "CLTV",
                "CMTR",
                "CFQC",
                "CVBL",
                "JCRG",
                "JCMP",
                "JSBR",
                "RPRE",
                "INT",
                "FIXN",
                "FIXV",
                "EAFF",
                "EITT",
                "NEG",
                "PUNC",
            ]
            features = schemas.seq_label_features(label_names)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, ""),
                    "split": "train",
                },
            )
        ]

    def _get_tokens_labels(self, paragraphs):
        tokens = []
        labels = []
        for paragraph in paragraphs:
            sentences = re.split(r"#\d+\n", paragraph)
            for sentence in sentences[1:]:
                token_pos_pairs = sentence.split("//")[1]
                for token_pos_pair in token_pos_pairs.split("\n")[1:-1]:
                    tokens.append(token_pos_pair.split("/")[0])
                    labels.append(token_pos_pair.split("/")[1])

        return tokens, labels

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        file_content = open(filepath, "r").read()
        texts = file_content.split("%TTitle:")

        if self.config.schema == "source":
            idx = 0
            for text in texts[1:]:
                parts = text.split("%")
                example = {
                    "TTitle": parts[0],
                    "ETitle": ":".join(parts[1].split(":")[1:]),
                    "TAuthor": ":".join(parts[2].split(":")[1:]),
                    "EAuthor": ":".join(parts[3].split(":")[1:]),
                    "TInbook": ":".join(parts[4].split(":")[1:]),
                    "EInbook": ":".join(parts[5].split(":")[1:]),
                    "TPublisher": ":".join(parts[6].split(":")[1:]),
                    "EPublisher": ":".join(parts[7].split(":")[1:]),
                    "Page": ":".join(parts[8].split(":")[1:]),
                    "Year": ":".join(parts[9].split(":")[1:]),
                    "File": ":".join(parts[10].split(":")[1:]),
                }
                yield idx, example
                idx += 1

        elif self.config.schema == "seacrowd_seq_label":
            idx = 0
            for text in texts[1:]:
                parts = text.split("%")
                last_part = parts[-1]
                tokens, labels = self._get_tokens_labels(re.split(r"#P\d+\n", last_part)[1:])
                example = {
                    "id": idx,
                    "tokens": tokens,
                    "labels": labels,
                }
                yield idx, example
                idx += 1
