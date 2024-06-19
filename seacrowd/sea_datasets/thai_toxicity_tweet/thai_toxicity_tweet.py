from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, TASK_TO_SCHEMA, Tasks

_CITATION = """\
@inproceedings{sirihattasak2018annotation,
  title     = {Annotation and Classification of Toxicity for Thai Twitter},
  author    = {Sirihattasak, Sugan and Komachi, Mamoru and Ishikawa, Hiroshi},
  year      = {2018},
  booktitle = {Proceedings of LREC 2018 Workshop and the 2nd Workshop on Text Analytics for Cybersecurity and Online Safety (TA-COS’18)},
  address   = {Miyazaki, Japan},
}
"""

_LOCAL = False
_LANGUAGES = ["tha"]
_DATASETNAME = "thai_toxicity_tweet"
_DESCRIPTION = """\
The Thai Toxicity Tweet Corpus contains 3,300 tweets (506 tweets with texts missing) annotated by humans with guidelines
including a 44-word dictionary. The author acquired 2,027 toxic and 1,273 non-toxic tweets, which were
manually labeled by three annotators. Toxicity is defined by the author as any message conveying harmful,
damaging, or negative intent, in accordance with their defined criteria for toxicity.
"""

_HOMEPAGE = "https://github.com/tmu-nlp/ThaiToxicityTweetCorpus"
_LICENSE = Licenses.CC_BY_NC_4_0.value
_URL = "https://huggingface.co/datasets/thai_toxicity_tweet"

_SUPPORTED_TASKS = [Tasks.ABUSIVE_LANGUAGE_PREDICTION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ThaiToxicityTweetsDataset(datasets.GeneratorBasedBuilder):
    """Dataset of Thai tweets annotated for toxicity."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEACROWD_SCHEMA = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()
    CLASS_LABELS = [0, 1]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA}",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "tweet_id": datasets.Value("string"),
                    "tweet_text": datasets.Value("string"),
                    "toxic_votes": datasets.Value("int32"),
                    "nontoxic_votes": datasets.Value("int32"),
                    "is_toxic": datasets.ClassLabel(names=self.CLASS_LABELS),
                }
            )

        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA}":
            features = schemas.text_features(label_names=self.CLASS_LABELS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # dl_manager not used since dataloader uses HF `load_dataset`
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train"},
            ),
        ]

    def _load_hf_data_from_remote(self) -> datasets.DatasetDict:
        """Load dataset from HuggingFace."""
        HF_REMOTE_REF = "/".join(_URL.split("/")[-1:])
        _hf_dataset_source = datasets.load_dataset(HF_REMOTE_REF, split="train")
        return _hf_dataset_source

    def _generate_examples(self, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = self._load_hf_data_from_remote()
        index = 0
        for row in data:
            if row["tweet_text"] in ("TWEET_NOT_FOUND", ""):
                continue
            if self.config.schema == "source":
                example = row
            elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA}":
                example = {
                    "id": str(index),
                    "text": row["tweet_text"],
                    "label": row["is_toxic"],
                }
            yield index, example
            index += 1
