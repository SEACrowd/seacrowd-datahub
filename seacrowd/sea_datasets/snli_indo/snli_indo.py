"""
SNLI Indo is derived from the SNLI corpus, where the premise and hypothesis sentences are translated directly from English to Indonesian using the Google Cloud Translation API. The SNLI corpus is divided into three sets, namely train, development, and test set. The translation process is applied to all the premise and hypothesis sentences in all the three sets. This ensures that the number of sentence pairs obtained is the same as the original SNLI dataset, namely 570k sentence pairs. A filtering process is carried out to remove incomplete sentence pairs and those with a gold label `-`. As a result, 569,027 sentence pairs are obtained.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import jsonlines

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{suwija2023snli,
  author       = "Suwija Putra, I Made 
        and Siahaan, Daniel 
        and Saikhu, Ahmad",
  title        = "SNLI Indo: A recognizing textual entailment dataset in Indonesian derived from the Stanford Natural Language Inference dataset"
  year         = "2024",
  journal      = "Data in Brief",
  volume       = "52",
  pages        = "109998",
  publisher    = "Elsevier",
  doi          = "https://doi.org/10.1016/j.dib.2023.109998",
  url          = "https://www.sciencedirect.com/science/article/pii/S2352340923010284",
}
"""

_DATASETNAME = "snli_indo"

_DESCRIPTION = """\
The SNLI Indo dataset is derived from the SNLI corpus by translating each premise and hypothesis sentence from English to Indonesia via the Google Cloud Translation API. Premise sentences are crawled image captions from Flickr, and hypothesis sentences are manually created through crowdsourcing. Five annotators are assigned per sentence pair to label the inference relationship as entailment (true), contradiction (false) or neutral (undetermined).
"""

_HOMEPAGE = "https://data.mendeley.com/datasets/k4tjhzs2gd/1"

_LANGUAGES = ["ind"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: {
        "train": "https://data.mendeley.com/public-files/datasets/k4tjhzs2gd/files/ee45b2bb-e2ea-47b7-bec4-b6653c467d27/file_downloaded",
        "val": "https://data.mendeley.com/public-files/datasets/k4tjhzs2gd/files/5e47db3c-ea84-4c73-9a2f-bfd57b4e2c05/file_downloaded",
        "test": "https://data.mendeley.com/public-files/datasets/k4tjhzs2gd/files/23aff85c-ff72-48b6-aba1-c1dd5dac216b/file_downloaded",
    }
}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class SNLIDataset(datasets.GeneratorBasedBuilder):
    """SNLI Indo is derived from the SNLI corpus, where the premise and hypothesis sentences are translated directly from English to Indonesian using the Google Cloud Translation API. This dataset contains ~570k annotated sentence pairs."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="snli_indo_source",
            version=SOURCE_VERSION,
            description="SNLI Indo source schema",
            schema="source",
            subset_id="snli_indo",
        ),
        SEACrowdConfig(
            name="snli_indo_seacrowd_pairs",
            version=SEACROWD_VERSION,
            description="SNLI Indo SEACrowd schema",
            schema="seacrowd_pairs",
            subset_id="snli_indo",
        ),
    ]

    DEFAULT_CONFIG_NAME = "snli_source"
    labels = ["kontradiksi", "keterlibatan", "netral"] # ["contradiction", "entailment", "neutral" ]

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=self.labels),
                }
            )

        elif self.config.schema == "seacrowd_pairs":
            features = schemas.pairs_features(self.labels)

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
                    "filepath": data_dir["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["val"],
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with jsonlines.open(filepath) as f:
                i = -1
                for example in f.iter():
                    i += 1
                    yield str(i), {
                        "premise": example["kalimat1"],
                        "hypothesis": example["kalimat2"],
                        "label": example["label emas"],
                    }

        elif self.config.schema == "seacrowd_pairs":
            with jsonlines.open(filepath) as f:
                i = -1
                for example in f.iter():
                    i += 1
                    yield str(i), {
                        "id": str(i),
                        "text_1": example["kalimat1"],
                        "text_2": example["kalimat2"],
                        "label": example["label emas"],
                    }
