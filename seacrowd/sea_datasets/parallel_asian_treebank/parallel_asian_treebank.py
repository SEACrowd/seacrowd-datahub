import itertools
from pathlib import Path
from typing import List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_DATASETNAME = "parallel_asian_treebank"

_LANGUAGES = ["khm", "lao", "mya", "ind", "fil", "zlm", "tha", "vie"]
_LANGUAGES_TO_FILENAME_LANGUAGE_CODE = {
    "khm": "khm",
    "lao": "lo",
    "mya": "my",
    "ind": "id",
    "fil": "fil",
    "zlm": "ms",
    "tha": "th",
    "vie": "vi",
    "eng": "en",
    "hin": "hi",
    "jpn": "ja",
    "zho": "zh",
}
_LOCAL = False
_CITATION = """\
@inproceedings{riza2016introduction,
  title={Introduction of the asian language treebank},
  author={Riza, Hammam and Purwoadi, Michael and Uliniansyah, Teduh and Ti, Aw Ai and Aljunied, Sharifah Mahani and Mai, Luong Chi and Thang, Vu Tat and Thai, Nguyen Phuong and Chea, Vichet and Sam, Sethserey and others},
  booktitle={2016 Conference of The Oriental Chapter of International Committee for Coordination and Standardization of Speech Databases and Assessment Techniques (O-COCOSDA)},
  pages={1--6},
  year={2016},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT.
It was first conducted by NICT and UCSY as described in Ye Kyaw Thu, Win Pa Pa, Masao Utiyama, Andrew Finch and Eiichiro Sumita (2016).
Then, it was developed under ASEAN IVO.
The process of building ALT began with sampling about 20,000 sentences from English Wikinews, and then these sentences were translated into the other languages.
ALT now has 13 languages: Bengali, English, Filipino, Hindi, Bahasa Indonesia, Japanese, Khmer, Lao, Malay, Myanmar (Burmese), Thai, Vietnamese, Chinese (Simplified Chinese).
"""

_HOMEPAGE = "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/"

_LICENSE = Licenses.CC_BY_4_0.value

_URLS = {
    "data": "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/ALT-Parallel-Corpus-20191206.zip",
    "train": "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-train.txt",
    "dev": "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-dev.txt",
    "test": "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/URL-test.txt",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ParallelAsianTreebankDataset(datasets.GeneratorBasedBuilder):
    """The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT"""

    BUILDER_CONFIGS = []
    lang_combinations = list(itertools.combinations(_LANGUAGES_TO_FILENAME_LANGUAGE_CODE.keys(), 2))
    for lang_a, lang_b in lang_combinations:
        if lang_a not in _LANGUAGES and lang_b not in _LANGUAGES:
            # Don't create a subset if both languages are not from SEA
            pass
        else:
            BUILDER_CONFIGS.append(
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{lang_a}_{lang_b}_source",
                    version=_SOURCE_VERSION,
                    description=f"{_DATASETNAME} source schema",
                    schema="source",
                    subset_id=f"{_DATASETNAME}_{lang_a}_{lang_b}_source",
                )
            )
            BUILDER_CONFIGS.append(
                SEACrowdConfig(
                    name=f"{_DATASETNAME}_{lang_a}_{lang_b}_seacrowd_t2t",
                    version=_SOURCE_VERSION,
                    description=f"{_DATASETNAME} seacrowd schema",
                    schema="seacrowd_t2t",
                    subset_id=f"{_DATASETNAME}_{lang_a}_{lang_b}_seacrowd_t2t",
                )
            )

    def _info(self):
        # The features are the same for both source and seacrowd
        features = schemas.text2text_features
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:

        def _split_at_n(text: str, n: int) -> Tuple[str, str]:
            """Split text on the n-th instance"""
            return ("_".join(text.split("_")[:n]), "_".join(text.split("_")[n:]))

        _, subset = _split_at_n(self.config.subset_id, 3)
        lang_pair, _ = _split_at_n(subset, 2)
        lang_a, lang_b = lang_pair.split("_")

        data_dir = Path(dl_manager.download_and_extract(_URLS["data"])) / "ALT-Parallel-Corpus-20191206"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "lang_a": lang_a, "lang_b": lang_b, "split_file": dl_manager.download(_URLS["train"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": data_dir, "lang_a": lang_a, "lang_b": lang_b, "split_file": dl_manager.download(_URLS["test"])},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_dir": data_dir, "lang_a": lang_a, "lang_b": lang_b, "split_file": dl_manager.download(_URLS["dev"])},
            ),
        ]

    def _generate_examples(self, data_dir: Path, lang_a: str, lang_b: str, split_file: str):

        def _get_texts(lang: str) -> pd.DataFrame:
            with open(data_dir / f"data_{_LANGUAGES_TO_FILENAME_LANGUAGE_CODE[lang]}.txt", "r") as f:
                rows = [line.strip().split("\t") for line in f.readlines()]

            url_id = [row[0].split(".")[1] for row in rows]
            sent_id = [row[0].split(".")[-1] for row in rows]
            text = []
            for row in rows:
                # There are rows with an empty text, but they are still tagged with an ID
                # so we keep them and just pass an empty string.
                sent = row[1] if len(row) > 1 else ""
                text.append(sent)


            df = pd.DataFrame({"url_id": url_id, "sent_id": sent_id, "text": text})
            return df

        with open(split_file, "r") as f:
            url_texts = [line.strip() for line in f.readlines()]
            # Get valid URLs for the split
            urlids_for_current_split = [row.split("\t")[0].split(".")[1] for row in url_texts]

        lang_a_df = _get_texts(lang_a)
        lang_b_df = _get_texts(lang_b)

        for idx, urlid in enumerate(urlids_for_current_split):
            lang_a_df_split = lang_a_df[lang_a_df["url_id"] == urlid]
            lang_b_df_split = lang_b_df[lang_b_df["url_id"] == urlid]

            if len(lang_a_df_split) == 0 or len(lang_b_df_split) == 0:
                # Sometimes, not all languages have values for a specific ID
                pass
            else:
                text_a = " ".join(lang_a_df_split["text"].to_list())
                text_b = " ".join(lang_b_df_split["text"].to_list())

                # Same schema for both source and SEACrowd
                yield idx, {
                    "id": idx,
                    "text_1": text_a,
                    "text_2": text_b,
                    "text_1_name": lang_a,
                    "text_2_name": lang_b,
                }
