# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{longpre-etal-2021-mkqa,
    title = "{MKQA}: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering",
    author = "Longpre, Shayne  and
      Lu, Yi  and
      Daiber, Joachim",
    editor = "Roark, Brian  and
      Nenkova, Ani",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "9",
    year = "2021",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2021.tacl-1.82",
    doi = "10.1162/tacl_a_00433",
    pages = "1389--1406",
}
"""

_DATASETNAME = "mkqa"

_DESCRIPTION = """\
Multilingual Knowledge Questions and Answers (MKQA), an open-domain question answering evaluation set comprising 10k question-answer pairs aligned across 26 typologically diverse languages (260k question-answer pairs in total)
"""

_HOMEPAGE = "https://github.com/apple/ml-mkqa"

_LICENSE = Licenses.CC_BY_SA_3_0.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_LANGUAGES = [
    "khm",
    "zsm",
    "tha",
    "vie",
]  # follows the convention of 3-letter code as suggested since NusaCrowd.


class MKQADataset(datasets.GeneratorBasedBuilder):
    """
    MKQA, an open-domain question answering evaluation set comprising 10k question-answer pairs
    aligned across 26 typologically diverse languages (260k question-answer pairs in total).
    The goal of this dataset is to provide a challenging benchmark for question answering quality
    across a wide set of languages.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    _ANS_TYPES = [
        "binary",
        "date",
        "entity",
        "long_answer",
        "number",
        "number_with_unit",
        "short_phrase",
        "unanswerable",
    ]

    _SOURCE_LANGUAGES = [
        "km",
        "ms",
        "th",
        "vi",
        # Filtered out:
        # "ar", "da", "de", "en", "es", "fi", "fr", "he", "hu", "it", "ja", "ko",
        # "nl", "no", "pl", "pt", "ru", "sv", "tr", "zh_cn", "zh_hk", "zh_tw",
    ]

    _LANG_3TO2 = {
        "khm": "km",
        "zsm": "ms",
        "tha": "th",
        "vie": "vi",
    }

    BUILDER_CONFIGS = [
        *[
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset_lang}{'_' if subset_lang else ''}source",
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} source schema",
                schema="source",
                subset_id=f"{_DATASETNAME}_{subset_lang}",
            )
            for subset_lang in ["", *_LANGUAGES]
        ],
        *[
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{subset_lang}{'_' if subset_lang else ''}seacrowd_qa",
                version=datasets.Version(_SEACROWD_VERSION),
                description=f"{_DATASETNAME} SEACrowd schema",
                schema="seacrowd_qa",
                subset_id=f"{_DATASETNAME}_{subset_lang}",
            )
            for subset_lang in ["", *_LANGUAGES]
        ],
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        lang = self.config.subset_id.rsplit("_", 1)[-1]
        lang = self._LANG_3TO2.get(lang, lang)

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "query": datasets.Value("string"),
                    "answers": {
                        cur_lang: [
                            {
                                "type": datasets.ClassLabel(names=self._ANS_TYPES),
                                "entity": datasets.Value("string"),
                                "text": datasets.Value("string"),
                                "aliases": [datasets.Value("string")],
                            }
                        ]
                        for cur_lang in ([lang] if lang else self._SOURCE_LANGUAGES)
                    },
                    "queries": {cur_lang: datasets.Value("string") for cur_lang in ([lang] if lang else self._SOURCE_LANGUAGES)},
                    "example_id": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_qa":
            features = schemas.qa_features
            features["meta"]["answer_entity"] = datasets.Sequence(datasets.Value("string"))
            features["meta"]["answer_aliases"] = datasets.Sequence(datasets.Sequence(datasets.Value("string")))
            features["meta"]["answer_type"] = datasets.Sequence(datasets.ClassLabel(names=self._ANS_TYPES))

        else:  # schema not found! should NOT reach here ...
            raise NotImplementedError()

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
        data_path = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_path},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        lang = self.config.subset_id.rsplit("_", 1)[-1]
        lang = self._LANG_3TO2.get(lang, lang)

        datas = []
        with open(filepath, "r", encoding="utf8") as ipt:
            for cur in map(json.loads, ipt):
                cur["example_id"] = str(cur["example_id"])
                for key in ["answers", "queries"]:
                    cur[key] = {k: v for k, v in cur[key].items() if k in ([lang] if lang else self._SOURCE_LANGUAGES)}
                datas.append(cur)

        if self.config.schema == "source":
            for cur in datas:
                for anslist in cur["answers"].values():
                    for ans in anslist:
                        ans.setdefault("entity", "")
                        ans.setdefault("aliases", [])
                yield int(cur["example_id"]), cur

        elif self.config.schema == "seacrowd_qa":
            for cur in datas:
                for cur_lang in [lang] if lang else map(lambda k: self._LANG_3TO2.get(k, k), _LANGUAGES):
                    ret = {
                        "id": f'{cur["example_id"]}_{cur_lang}',
                        "question_id": cur["example_id"],
                        "document_id": "",
                        "question": cur["queries"][cur_lang],
                        "type": "open_domain",
                        "choices": [],
                        "context": "",
                        "answer": [ans.get("text", None) for ans in cur["answers"][cur_lang]],
                        "meta": {f"answer_{k}": [ans.get(k, None) for ans in cur["answers"][cur_lang]] for k in ["entity", "aliases", "type"]},
                    }
                    ret["meta"]["answer_aliases"] = list(map(lambda a: [] if a is None else a, ret["meta"]["answer_aliases"]))
                    yield ret["id"], ret
