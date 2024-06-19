import csv
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses
from seacrowd.utils import schemas

_CITATION = """\
@mastersthesis{chumpolsathien_2020,
    title={Using Knowledge Distillation from Keyword Extraction to Improve the Informativeness of Neural Cross-lingual Summarization},
    author={Chumpolsathien, Nakhun},
    year={2020},
    school={Beijing Institute of Technology}
"""

_DATASETNAME = "thai_sum"

_DESCRIPTION = """
We present ThaiSum, a large-scale corpus for Thai text summarization obtained from several online news websites namely Thairath, ThaiPBS, Prachathai, and The Standard. This dataset consists of over 350,000 article and summary pairs written by journalists.
"""

_HOMEPAGE = "https://github.com/nakhunchumpolsathien/ThaiSum"

_LICENSE = Licenses.MIT.value

_LANGUAGES = ["tha"]

_LOCAL = False

_URLS = {
    "train": "https://nakhun-chumpolsathien.oss-us-west-1.aliyuncs.com/thaisum/thaisum.csv",
    "val": "https://nakhun-chumpolsathien.oss-us-west-1.aliyuncs.com/thaisum/validation_set.csv",
    "test": "https://nakhun-chumpolsathien.oss-us-west-1.aliyuncs.com/thaisum/test_set.csv",
}

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class ThaiSumDataset(datasets.GeneratorBasedBuilder):
    """
    Sequence-to-sequence (Seq2Seq) models have shown great achievement in text summarization.
    However, Seq2Seq model often requires large-scale training data to achieve effective results.
    Although many impressive advancements in text summarization field have been made,
    most of summarization studies focus on resource-rich languages.
    The progress of Thai text summarization is still far behind.
    The dearth of large-scale dataset keeps Thai text summarization in its infancy.
    As far as our knowledge goes, there is not a large-scale dataset for Thai text summarization available anywhere.
    Thus, we present ThaiSum, a large-scale corpus for Thai text summarization
    obtained from several online news websites namely Thairath, ThaiPBS, Prachathai, and The Standard.
    This dataset consists of over 350,000 article and summary pairs written by journalists.
    We evaluate the performance of various existing summarization models on ThaiSum dataset and analyse
    the characteristic of the dataset to present its difficulties.
    """

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
            name=f"{_DATASETNAME}_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"title": datasets.Value("string"), "body": datasets.Value("string"), "summary": datasets.Value("string"), "type": datasets.Value("string"), "tags": datasets.Value("string"), "url": datasets.Value("string")})
        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        path_dict = dl_manager.download_and_extract(_URLS)
        train_path, val_path, test_path = path_dict["train"], path_dict["val"], path_dict["test"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": val_path,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        csv.field_size_limit(int(1000000))
        with open(filepath, encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # skip header
            if self.config.schema == "source":
                for id_, row in enumerate(csv_reader):
                    yield id_, {
                        "title": row[0],
                        "body": row[1],
                        "summary": row[2],
                        "type": row[3],
                        "tags": row[4],
                        "url": row[5],
                    }
            elif self.config.schema == "seacrowd_t2t":
                for id_, row in enumerate(csv_reader):
                    yield id_, {
                        "id": str(id_),
                        "text_1": row[1],
                        "text_2": row[2],
                        "text_1_name": "document",
                        "text_2_name": "summary",
                    }
