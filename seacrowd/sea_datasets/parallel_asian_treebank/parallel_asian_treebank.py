from pathlib import Path
from typing import List
from itertools import combinations

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_DATASETNAME = "parallel_asian_treebank"

_LANGUAGES = ["khm", "lao", "mya", "ind", "fil", "zlm", "tha", "vie"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_MAPPING_LANGUAGES = { "khm": "khm", "lao": "lo", "mya": "my", "ind": "id", "fil": "fil", "zlm": "ms", "tha": "th", "vie": "vi", }
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
            description=f"{_DATASETNAME} Nusantara schema",
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
        data_path = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path
                },
            )
        ]

    def _generate_examples(self, filepath: Path):

        mapping_datas = {}

        for language in _LANGUAGES:
            datas = open(f"{filepath}/ALT-Parallel-Corpus-20191206/data_{_MAPPING_LANGUAGES[language]}.txt", "r").readlines()

            for line in datas:
                id, sentence = line.split("\t")
                sentence, _ = sentence.split("\n")

                if id not in mapping_datas:
                    mapping_datas[id] = {}

                if language not in mapping_datas[id]:
                    mapping_datas[id][language] = {}

                mapping_datas[id][language] = sentence
        
        combination_languages = list(combinations(_LANGUAGES, 2))

        if self.config.schema == "source" or self.config.schema == "seacrowd_t2t":
            i = 0
            for id in mapping_datas:
                for each_pair in combination_languages:
                    if each_pair[0] in mapping_datas[id] and each_pair[1] in mapping_datas[id]:
                        yield i, {
                            "id": f"{id}-{each_pair[0]}-{each_pair[1]}",
                            "text_1": mapping_datas[id][each_pair[0]],
                            "text_2": mapping_datas[id][each_pair[1]],
                            "text_1_name": each_pair[0],
                            "text_2_name": each_pair[1],
                        }

                        i+=1

                        yield i, {
                            "id": f"{id}-{each_pair[1]}-{each_pair[0]}",
                            "text_1": mapping_datas[id][each_pair[1]],
                            "text_2": mapping_datas[id][each_pair[0]],
                            "text_1_name": each_pair[1],
                            "text_2_name": each_pair[0],
                        }

                        i+=1

        else:
            raise ValueError(f"Invalid config: {self.config.name}")
