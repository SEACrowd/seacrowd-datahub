from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{
    mukherjee-etal-2023-global,
    title = "{G}lobal {V}oices, Local Biases: Socio-Cultural Prejudices across Languages",
    author = "Mukherjee, Anjishnu and Raj, Chahat and Zhu, Ziwei and Anastasopoulos, Antonios",
    editor = "Bouamor, Houda and Pino, Juan and Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.981",
    doi = "10.18653/v1/2023.emnlp-main.981",
    pages = "15828--15845",
    abstract = "Human biases are ubiquitous but not uniform: disparities exist across linguistic, cultural, and societal borders. As large amounts of recent literature suggest, language models (LMs) trained on human data can reflect and often amplify the effects of these social biases. However, the vast majority of existing studies on bias are heavily skewed towards Western and European languages. In this work, we scale the Word Embedding Association Test (WEAT) to 24 languages, enabling broader studies and yielding interesting findings about LM bias. We additionally enhance this data with culturally relevant information for each language, capturing local contexts on a global scale. Further, to encompass more widely prevalent societal biases, we examine new bias dimensions across toxicity, ableism, and more. Moreover, we delve deeper into the Indian linguistic landscape, conducting a comprehensive regional bias analysis across six prevalent Indian languages. Finally, we highlight the significance of these social biases and the new dimensions through an extensive comparison of embedding methods, reinforcing the need to address them in pursuit of more equitable language models."
}
"""

_DATASETNAME = "weathub"

_DESCRIPTION = """\
    WEATHub is a dataset containing 24 languages. It contains words organized into groups of (target1, target2, attribute1, attribute2) to measure the association target1:target2 :: attribute1:attribute2. For example target1 can be insects, target2 can be flowers. And we might be trying to measure whether we find insects or flowers pleasant or unpleasant. The measurement of word associations is quantified using the WEAT metric from their paper. It is a metric that calculates an effect size (Cohen's d) and also provides a p-value (to measure statistical significance of the results). In their paper, they use word embeddings from language models to perform these tests and understand biased associations in language models across different languages.
"""

_HOMEPAGE = "https://huggingface.co/datasets/iamshnoo/WEATHub"

_LANGUAGES = ["tha", "tgl", "vie", "cmn", "eng"]

_LICENSE = Licenses.CC_BY_4_0.value

_LOCAL = False

_URLS = {
    "original_weat": "https://huggingface.co/datasets/iamshnoo/WEATHub/resolve/main/data/original_weat-00000-of-00001-7e85a83920777d87.parquet",
    "new_human_biases": "https://huggingface.co/datasets/iamshnoo/WEATHub/resolve/main/data/new_human_biases-00000-of-00001-557150d939938864.parquet",
    "india_specific_biases": "https://huggingface.co/datasets/iamshnoo/WEATHub/resolve/main/data/india_specific_biases-00000-of-00001-a2b8ed74878afc1b.parquet",
}

_SUPPORTED_TASKS = [Tasks.WORD_LIST]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class WEATHubDataset(datasets.GeneratorBasedBuilder):
    """WEATHub is a dataset containing 24 languages. It contains words organized into groups of (target1, target2, attribute1, attribute2) to measure the association target1:target2 :: attribute1:attribute2. This dataset corresponds to the data described in the paper "Global Voices, Local Biases: Socio-Cultural Prejudices across Languages" accepted to EMNLP 2023."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)
    SEA_LANGUAGES = set(["th", "tl", "vi", "zh", "en"])

    subsets = [
        f"{_DATASETNAME}.original_weat",
        f"{_DATASETNAME}.new_human_biases",
        f"{_DATASETNAME}.india_specific_biases",
    ]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{sub}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{sub} source schema",
            schema="source",
            subset_id=sub,
        )
        for sub in subsets
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}.original_weat_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "language": datasets.Value("string"),
                    "weat": datasets.Value("string"),
                    "attr1.category": datasets.Value("string"),
                    "attr1.examples": [datasets.Value("string")],
                    "attr2.category": datasets.Value("string"),
                    "attr2.examples": [datasets.Value("string")],
                    "targ1.category": datasets.Value("string"),
                    "targ1.examples": [datasets.Value("string")],
                    "targ2.category": datasets.Value("string"),
                    "targ2.examples": [datasets.Value("string")],
                }
            )
        elif "seacrowd" in self.config.schema:
            raise NotImplementedError("No seacrowd schema for word list tasks")
        else:
            raise ValueError("Invalid schema name")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS
        data_dir = {split: dl_manager.download_and_extract(url) for split, url in urls.items()}

        subset = "_".join(self.config.name.split(".")[-1].split("_")[:-1])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir[subset],
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            df = pd.read_parquet(filepath)
            for i, example in df.iterrows():
                if example["language"] not in self.SEA_LANGUAGES:
                    continue
                yield str(i), {
                    "language": example["language"],
                    "weat": example["weat"],
                    "attr1.category": example["attr1.category"],
                    "attr1.examples": example["attr1.examples"],
                    "attr2.category": example["attr2.category"],
                    "attr2.examples": example["attr2.examples"],
                    "targ1.category": example["targ1.category"],
                    "targ1.examples": example["targ1.examples"],
                    "targ2.category": example["targ2.category"],
                    "targ2.examples": example["targ2.examples"],
                }
