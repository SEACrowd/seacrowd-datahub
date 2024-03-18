from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks

_CITATION = """\
@misc{fransiska2023utilizing,
      title={Utilizing Weak Supervision To Generate Indonesian Conservation Dataset},
      author={Mega Fransiska and Diah Pitaloka and Saripudin and Satrio Putra and Lintang Sutawika},
      year={2023},
      eprint={2310.11258},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "mongabay"

_DESCRIPTION = """\
Conservation dataset that was collected from mongabay.co.id contains
 topic-classification task (multi-label format) and sentiment classification.
 The dataset consists of 31 important topics that are commonly found in
 Indonesian conservation articles or general news, and each article can
 belong to more than one topic. After gathering topics for each article,
 each article will be classified into one of author's sentiments
 (positive, neutral, negative) based on related topics.
"""

_HOMEPAGE = ""

_LICENSE = "The Unlicense (unlicense)"

_URLS = {"mongabay-tag-classification": "https://huggingface.co/datasets/Datasaur/Mongabay-tags-classification", "mongabay-sentiment-classification": "https://huggingface.co/datasets/Datasaur/Mongabay-sentiment-classification"}

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "1.0.0"

_SUPPORTED_TASKS = [Tasks.PARAPHRASING]

_LANGUAGES = ["ind"]


class Mongabay(datasets.GeneratorBasedBuilder):
    """mongabay is a dataset sourced from mongabay.co.id's Indonesian articles from 2012-2023. Each article is chunked to maximum 512 tokens to ease experiment process"""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="mongabay-tag-classification_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="mongabay-tag-classification source schema",
            schema="source",
            subset_id="mongabay-tag-classification",
        ),
        SEACrowdConfig(
            name="mongabay-tag-classification_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description="mongabay-tag-classification SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id="mongabay-tag-classification",
        ),
        SEACrowdConfig(
            name="mongabay-sentiment-classification_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="mongabay-sentiment-classification source schema",
            schema="source",
            subset_id="mongabay-sentiment-classification",
        ),
        SEACrowdConfig(
            name="mongabay-sentiment-classification_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description="mongabay-sentiment-classification SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id="mongabay-sentiment-classification",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            if "mongabay-sentiment-classification" in self.config.name:
                features = datasets.Features(
                    {
                        "text": datasets.Value("string"),
                        "tags": datasets.Value("string"),
                        "label": datasets.Value("string"),
                    }
                )
            elif "mongabay-tag-classification" in self.config.name:
                features = datasets.Features(
                    {
                        "text": datasets.Value("string"),
                        "label": datasets.Value("string"),
                    }
                )
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
        name = self.config.name.replace("_" + self.config.schema, "")
        url = _URLS[name]
        filename = "/".join(url.split("/")[-2:])

        output = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filename": filename,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filename": filename,
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filename": filename,
                    "split": "test",
                },
            ),
        ]

        return output

    def _generate_examples(self, filename: Path, split: str) -> Tuple[int, Dict]:
        """Yield examples as tuples of idx, (text, tags[optional], and label)"""

        try:
            dataset = datasets.load_dataset(filename)[split]

            if self.config.schema == "source":
                for idx, row in enumerate(dataset):
                    yield idx, row

            elif self.config.schema == "seacrowd_t2t":
                for idx, row in enumerate(dataset):
                    sample = {"id": str(idx), "text_1": row["text"], "text_2": row["label"], "text_1_name": "text", "text_2_name": "weak_label"}
                    yield idx, sample
        except datasets.exceptions.DatasetGenerationError as e:
            print(e)
