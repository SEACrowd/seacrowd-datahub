from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{miranda-2023-developing,
    title = {Developing a Named Entity Recognition Dataset for Tagalog},
    author = "Miranda, Lester James Validad",
    booktitle = "Proceedings of the First Workshop for Southeast Asian Language Processing (SEALP),"
    month = nov,
    year = 2023,
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
"""

_LOCAL = False
_LANGUAGES = ["tgl"]
_DATASETNAME = "tlunified_ner"
_DESCRIPTION = """\
This dataset contains the annotated TLUnified corpora from Cruz and Cheng
(2021). It is a curated sample of around 7,000 documents for the named entity
recognition (NER) task. The majority of the corpus are news reports in Tagalog,
resembling the domain of the original ConLL 2003. There are three entity types:
Person (PER), Organization (ORG), and Location (LOC).
"""

_HOMEPAGE = "https://huggingface.co/ljvmiranda921/tlunified-ner"
_LICENSE = Licenses.GPL_3_0.value
_URLS = {
    "train": "https://huggingface.co/datasets/ljvmiranda921/tlunified-ner/resolve/main/corpus/iob/train.iob",
    "dev": "https://huggingface.co/datasets/ljvmiranda921/tlunified-ner/resolve/main/corpus/iob/dev.iob",
    "test": "https://huggingface.co/datasets/ljvmiranda921/tlunified-ner/resolve/main/corpus/iob/test.iob",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class TLUnifiedNERDataset(datasets.GeneratorBasedBuilder):
    """Tagalog Named Entity Recognition dataset from https://huggingface.co/ljvmiranda921/tlunified-ner"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "seq_label"
    LABEL_CLASSES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.features.ClassLabel(names=self.LABEL_CLASSES)),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.seq_label_features(self.LABEL_CLASSES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_files = {
            "train": Path(dl_manager.download_and_extract(_URLS["train"])),
            "dev": Path(dl_manager.download_and_extract(_URLS["dev"])),
            "test": Path(dl_manager.download_and_extract(_URLS["test"])),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_files["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_files["dev"], "split": "dev"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_files["test"], "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""
        # The only difference between the source schema and the seacrowd seq_label schema is the dictionary keys.
        # The implementation is the same.
        label_key = "ner_tags" if self.config.schema == "source" else "labels"
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            label_key: ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # TLUnified-NER iob are separated by \t
                    token, ner_tag = line.split("\t")
                    tokens.append(token)
                    ner_tags.append(ner_tag.rstrip())
            # Last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    label_key: ner_tags,
                }
