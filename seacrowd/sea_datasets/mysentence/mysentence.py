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

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{Aung_Kyaw Thu_Hlaing_2023, place={Nonthaburi, Thailand}, title={mySentence: Sentence Segmentation for Myanmar Language
using Neural Machine Translation Approach}, volume={9}, url={https://ph05.tci-thaijo.org/index.php/JIIST/article/view/87}, abstractNote=
{&amp;lt;p&amp;gt;&amp;amp;nbsp;A sentence is an independent unit which is a string of complete words containing valuable information of the text.
In informal Myanmar Language, for which most of NLP applications like Automatic Speech Recognition (ASR) are used, there is no predefined rule to
mark the end of sentence. In this paper, we contributed the first corpus for Myanmar Sentence Segmentation and proposed the first systematic study
with Machine Learning based Sequence Tagging as baseline and Neural Machine Translation approach. Before conducting the experiments, we prepared two
types of data - one containing only sentences and the other containing both sentences and paragraphs. We trained each model on both types of data and
evaluated the results on both types of test data. The accuracies were measured in terms of Bilingual Evaluation Understudy (BLEU) and character n-gram
F-score (CHRF ++) scores. Word Error Rate (WER) was also used for the detailed study of error analysis. The experimental results show that Sequence-to-Sequence
architecture based Neural Machine Translation approach with the best BLEU score (99.78), which is trained on both sentence-level and paragraph-level data, achieved better CHRF
++ scores (+18.4) and (+16.7) than best results of such machine learning models on both test data.&amp;lt;/p&amp;gt;}, number={October}, journal={Journal of Intelligent Informatics
and Smart Technology}, author={Aung, Thura and Kyaw Thu , Ye and Hlaing , Zar Zar}, year={2023}, month={Nov.}, pages={e001} }
"""

_DATASETNAME = "mysentence"
_DESCRIPTION = """\
mySentence is a corpus with a total size of around 55K for Myanmar sentence segmentation. In formal Burmese (Myanmar language), sentences are grammatically structured
and typically end with the "။" pote-ma symbol. However, informal language, more commonly used in daily conversations due to its natural flow, does not always follow predefined
rules for ending sentences, making it challenging for machines to identify sentence boundaries. In this corpus, each token of the sentences and paragraphs is tagged from start to finish.
"""

_HOMEPAGE = "https://github.com/ye-kyaw-thu/mySentence"

_LANGUAGES = ["mya"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value  # example: Licenses.MIT.value, Licenses.CC_BY_NC_SA_4_0.value, Licenses.UNLICENSE.value, Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "sent": {
        "train": "https://raw.githubusercontent.com/ye-kyaw-thu/mySentence/main/ver1.0/data/data-sent/sent_tagged/train.tagged",
        "valid": "https://raw.githubusercontent.com/ye-kyaw-thu/mySentence/main/ver1.0/data/data-sent/sent_tagged/valid.tagged",
        "test": "https://raw.githubusercontent.com/ye-kyaw-thu/mySentence/main/ver1.0/data/data-sent/sent_tagged/test.tagged",
    },
    "sent+para": {
        "train": "https://raw.githubusercontent.com/ye-kyaw-thu/mySentence/main/ver1.0/data/data-sent+para/sent+para_tagged/train.tagged",
        "valid": "https://raw.githubusercontent.com/ye-kyaw-thu/mySentence/main/ver1.0/data/data-sent+para/sent+para_tagged/valid.tagged",
        "test": "https://raw.githubusercontent.com/ye-kyaw-thu/mySentence/main/ver1.0/data/data-sent+para/sent+para_tagged/test.tagged",
    },
}

_SUPPORTED_TASKS = [Tasks.POS_TAGGING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class mysentenceDataset(datasets.GeneratorBasedBuilder):
    """mySentence is a corpus with a total size of around 55K for Myanmar sentence segmentation. In formal Burmese (Myanmar language), sentences are grammatically
    structured and typically end with the "။" pote-ma symbol. However, informal language, more commonly used in daily conversations due to its natural flow, does not
    always follow predefined rules for ending sentences, making it challenging for machines to identify sentence boundaries. In this corpus, each token of the sentences and paragraphs is tagged from start to finish."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    # print(schemas.seq_label)
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="mysentence_source",
            version=SOURCE_VERSION,
            description="sentences source schema",
            schema="source",
            subset_id="mysentence",
        ),
        SEACrowdConfig(
            name="mysentence_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="sentences SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id="mysentence",
        ),
        SEACrowdConfig(
            name="mysentence+paragraphs_source",
            version=SOURCE_VERSION,
            description="sentences para source schema",
            schema="source",
            subset_id="mysentence+paragraphs",
        ),
        SEACrowdConfig(
            name="mysentence+paragraphs_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="sentence para SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id="mysentence+paragraphs",
        ),
    ]

    DEFAULT_CONFIG_NAME = "mysentence_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(datasets.Value("string")),
                }
            )
        else:
            features = schemas.seq_label_features(["B", "O", "N", "E"])
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # B (Begin), O (Other), N (Next), and E (End)
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.subset_id == "mysentence":
            DATA_URL_ = _URLS["sent"]
        elif self.config.subset_id == "mysentence+paragraphs":
            DATA_URL_ = _URLS["sent+para"]
        else:
            raise ValueError(f"No related dataset id for {self.config.subset_id}")

        data_dir = dl_manager.download_and_extract(DATA_URL_)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_dir["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir["test"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["valid"],
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:

        with open(filepath, "r") as filein:
            examples = [line.strip("\n").split(" ") for line in filein.readlines()]
            for eid, exam in enumerate(examples):
                tokens = []
                pos = []
                for tok_chunk in exam:
                    tok_ = tok_chunk.split("/")
                    tokens.append(tok_[0])
                    pos.append(tok_[1])
                yield eid, {"id": str(eid), "tokens": tokens, "labels": pos}
