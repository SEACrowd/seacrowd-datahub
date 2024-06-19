from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@inproceedings{chenchen2017statistical,
    author    = {Ding Chenchen and
                Chea Vichet and
                Pa Win Pa and
                Utiyama Masao and
                Sumita Eiichiro},
    title     = {Statistical Romanization for Abugida Scripts: Data and Experiment on Khmer and Burmese},
    booktitle = {Proceedings of the 23rd Annual Conference of the Association for Natural Language Processing,
                {NLP2017}, Tsukuba, Japan, 13-17 March 2017},
    year      = {2017},
    url       = {https://www.anlp.jp/proceedings/annual_meeting/2017/pdf_dir/P5-7.pdf},
}
"""

_DATASETNAME = "burmese_romanize"
_DESCRIPTION = """\
This dataset consists of 2,335 Burmese names from real university students and faculty,
public figures, and minorities from Myanmar. Each entry includes the original name in
Burmese script, its corresponding Romanization, and the aligned Burmese and Latin
graphemes.
"""

_HOMEPAGE = "http://www.nlpresearch-ucsy.edu.mm/NLP_UCSY/name-db.html"
_LANGUAGES = ["mya"]
_LICENSE = Licenses.CC_BY_NC_SA_4_0.value
_URLS = "http://www.nlpresearch-ucsy.edu.mm/NLP_UCSY/myanmaroma.zip"

_SUPPORTED_TASKS = [Tasks.TRANSLITERATION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"

_LOCAL = False


class BurmeseRomanizeDataset(datasets.GeneratorBasedBuilder):
    """Romanization of names in Burmese script"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "t2t"

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "original": datasets.Value("string"),
                    "romanized": datasets.Value("string"),
                    "aligned_graphemes": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data_dir = Path(dl_manager.download_and_extract(_URLS)) / "myanmaroma"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir / "myanmaroma.txt",
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        df = pd.read_csv(filepath, sep=" \|\|\| ", engine='python', header=None, names=["ori", "roman", "seg"])
        if self.config.schema == "source":
            for i, row in df.iterrows():
                yield i, {
                    "original": row["ori"],
                    "romanized": row["roman"],
                    "aligned_graphemes": row["seg"].strip().split(),
                }
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            for i, row in df.iterrows():
                yield i, {
                    "id": str(i),
                    "text_1": row["ori"],
                    "text_2": row["roman"],
                    "text_1_name": "original",
                    "text_2_name": "romanized",
                }