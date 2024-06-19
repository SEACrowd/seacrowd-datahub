# coding=utf-8
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{Aung_Kyaw_Thu_Hlaing_2023,
    title        = {{mySentence: Sentence Segmentation for Myanmar Language using Neural Machine Translation Approach}},
    author       = {Aung, Thura and Kyaw Thu , Ye and Hlaing , Zar Zar},
    year         = 2023,
    month        = {Nov.},
    journal      = {Journal of Intelligent Informatics and Smart Technology},
    volume       = 9,
    number       = {October},
    pages        = {e001},
    url          = {https://ph05.tci-thaijo.org/index.php/JIIST/article/view/87},
    place        = {Nonthaburi, Thailand},
    abstract     = {In the informal Myanmar language, for which most NLP applications are used, there is no predefined rule to mark the end of the sentence. Therefore, in this paper, we contributed the first Myanmar sentence segmentation corpus and systemat ically experimented with twelve neural sequence labeling architectures trained and tested on both sentence and sentence+paragraph data. The word LSTM + Softmax achieved the highest accuracy of 99.95{\%} while trained and tested on sentence-only data and 97.40{\%} while trained and tested on sentence + paragraph data.}
}
@inproceedings{10.1007/978-3-031-36886-8_24,
    title        = {{Neural Sequence Labeling Based Sentence Segmentation for Myanmar Language}},
    author       = {Thu, Ye Kyaw and Aung, Thura and Supnithi, Thepchai},
    year         = 2023,
    booktitle    = {The 12th Conference on Information Technology and Its Applications},
    publisher    = {Springer Nature Switzerland},
    address      = {Cham},
    pages        = {285--296},
    isbn         = {978-3-031-36886-8},
    editor       = {Nguyen, Ngoc Thanh and Le-Minh, Hoa and Huynh, Cong-Phap and Nguyen, Quang-Vu},
    abstract     = {In the informal Myanmar language, for which most NLP applications are used, there is no predefined rule to mark the end of the sentence. Therefore, in this paper, we contributed the first Myanmar sentence segmentation corpus and systemat ically experimented with twelve neural sequence labeling architectures trained and tested on both sentence and sentence+paragraph data. The word LSTM + Softmax achieved the highest accuracy of 99.95{\%} while trained and tested on sentence-only data and 97.40{\%} while trained and tested on sentence + paragraph data.}
}
"""

_DATASETNAME = "mysentence"
_DESCRIPTION = """\
mySentence is a corpus with a total size of around 55K for Myanmar sentence segmentation. In formal Burmese (Myanmar language), sentences are grammatically structured
and typically end with the "။" pote-ma symbol. However, informal language, more commonly used in daily conversations due to its natural flow, does not always follow predefined
rules for ending sentences, making it challenging for machines to identify sentence boundaries. In this corpus, each token of the sentences and paragraphs is tagged from start to finish.
"""

_HOMEPAGE = "https://github.com/ye-kyaw-thu/mySentence"
_LANGUAGES = ["mya"]
_LICENSE = Licenses.CC_BY_NC_SA_4_0.value
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
_SEACROWD_VERSION = "2024.06.20"


class MysentenceDataset(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=_DESCRIPTION,
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="sentences SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_and_paragraphs_source",
            version=SOURCE_VERSION,
            description="sentences para source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_and_paragraphs",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_and_paragraphs_seacrowd_seq_label",
            version=SEACROWD_VERSION,
            description="sentence para SEACrowd schema",
            schema="seacrowd_seq_label",
            subset_id=f"{_DATASETNAME}_and_paragraphs",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

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
        if self.config.subset_id == f"{_DATASETNAME}":
            DATA_URL_ = _URLS["sent"]
        elif self.config.subset_id == f"{_DATASETNAME}_and_paragraphs":
            DATA_URL_ = _URLS["sent+para"]
        else:
            raise ValueError(f"No related dataset id for {self.config.subset_id}")

        data_dir = dl_manager.download_and_extract(DATA_URL_)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
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
