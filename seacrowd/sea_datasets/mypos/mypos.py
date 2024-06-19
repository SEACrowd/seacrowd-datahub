from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@inproceedings{hlaing-2020-myanmar,
  author={Hlaing, Zar Zar and Thu, Ye Kyaw and Wai, Myat Myo Nwe and Supnithi, Thepchai and Netisopakul, Ponrudee},
  booktitle={2020 15th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP)},
  title={Myanmar POS Resource Extension Effects on Automatic Tagging Methods},
  year={2020},
  pages={1-6},
  doi={10.1109/iSAI-NLP51646.2020.9376835}}
@inproceedings{htike2017comparison,
  title={Comparison of six POS tagging methods on 10K sentences Myanmar language (Burmese) POS tagged corpus},
  author={Htike, Khin War War and Thu, Ye Kyaw and Zuping Zhang, Win Pa Pa and Sagisaka, Yoshinori and Iwahashi, Naoto},
  booktitle={Proceedings of the CICLING},
  year={2017}
}
"""

_LOCAL = False
_LANGUAGES = ["mya"]
_DATASETNAME = "mypos"
_DESCRIPTION = """\
This version of the myPOS corpus extends the original myPOS corpus from
11,000 to 43,196 Burmese sentences by adding data from the ASEAN MT NECTEC
corpus and two developed parallel corpora (Myanmar-Chinese and
Myanmar-Korean). The original 11,000 sentences were collected from Wikipedia
and includes various topics such as economics, history, news, politics and
philosophy. The format used in the corpus is word/POS-tag, and the pipe
delimiter "
"""

_HOMEPAGE = "https://github.com/ye-kyaw-thu/myPOS"
_LICENSE = Licenses.CC_BY_NC_SA_4_0.value
_URL = "https://raw.githubusercontent.com/ye-kyaw-thu/myPOS/master/corpus-ver-3.0/corpus/mypos-ver.3.0.txt"

_SUPPORTED_TASKS = [Tasks.POS_TAGGING]
_SOURCE_VERSION = "3.0.0"
_SEACROWD_VERSION = "2024.06.20"


class MyPOSDataset(datasets.GeneratorBasedBuilder):
    """MyPOS dataset from https://github.com/ye-kyaw-thu/myPOS"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "seq_label"
    # Reference: https://github.com/ye-kyaw-thu/myPOS/tree/master#pos-tags
    LABEL_CLASSES = ["abb", "adj", "adv", "conj", "fw", "int", "n", "num", "part", "ppm", "pron", "punc", "sb", "tn", "v"]

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
        # No specific schema from source, we will just reuse the seacrowd schema
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
        data_file = Path(dl_manager.download_and_extract(_URL))
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_file})]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yield examples as (key, example) tuples"""

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):

            line = line.rstrip("\n")
            tags = self._tokenize(line)

            split_token = [tag.split("/") for tag in tags if tag]
            tokens = [split[0] for split in split_token]
            labels = [split[1] for split in split_token]
            example = {"id": str(idx), "tokens": tokens, "labels": labels}

            yield idx, example

    def _tokenize(self, sentence: str) -> List[str]:
        """Tokenize Myanmar text

        From the README: https://github.com/ye-kyaw-thu/myPOS/tree/master#word-segmentation
        Important things to point out:
        - Words composed of single or multiple syllables are usually not separated by white space.
        - There are no clear rules for using spaces in Myanmar language.
        - The authors used six rules for word segmentation
        """
        final_tokens = []

        # Segment via spaces (c.f. "Spaces are used for easier reading and generally put between phrases")
        init_tokens = sentence.split(" ")
        # Segment breakpoints ('|' pipe character) for compount words
        for token in init_tokens:
            final_tokens.extend(token.split("|"))

        return final_tokens
