from itertools import combinations
from typing import List

import datasets

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
}
_LOCAL = True
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

_URL = "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/ALT-Parallel-Corpus-20191206.zip"

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class ParallelAsianTreebank(datasets.GeneratorBasedBuilder):
    """The ALT project aims to advance the state-of-the-art Asian natural language processing (NLP) techniques through the open collaboration for developing and using ALT"""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=_SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_t2t",
            version=_SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "text_1_name": datasets.Value("string"),
                    "text_2_name": datasets.Value("string"),
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
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": data_dir, "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_dir": data_dir, "split": "dev"},
            ),
        ]

    def _generate_examples(self, data_dir: str, split: str):

        if self.config.schema not in ["source", "seacrowd_t2t"]:
            raise ValueError(f"Invalid config: {self.config.name}")

        mapping_data = {}

        for language in _LANGUAGES:
            lines = open(f"{data_dir}/data_{_LANGUAGES_TO_FILENAME_LANGUAGE_CODE[language]}.txt.{split}", "r").readlines()

            for line in lines:
                id, sentence = line.split("\t")
                sentence = sentence.rsplit()

                if id not in mapping_data:
                    mapping_data[id] = {}

                mapping_data[id][language] = sentence

        combination_languages = list(combinations(_LANGUAGES, 2))

        i = 0

        for id in mapping_data:
            for each_pair in combination_languages:
                if each_pair[0] in mapping_data[id] and each_pair[1] in mapping_data[id]:
                    yield i, {
                        "id": f"{id}-{each_pair[0]}-{each_pair[1]}",
                        "text_1": mapping_data[id][each_pair[0]],
                        "text_2": mapping_data[id][each_pair[1]],
                        "text_1_name": each_pair[0],
                        "text_2_name": each_pair[1],
                    }

                    i += 1

                    yield i, {
                        "id": f"{id}-{each_pair[1]}-{each_pair[0]}",
                        "text_1": mapping_data[id][each_pair[1]],
                        "text_2": mapping_data[id][each_pair[0]],
                        "text_1_name": each_pair[1],
                        "text_2_name": each_pair[0],
                    }

                    i += 1
