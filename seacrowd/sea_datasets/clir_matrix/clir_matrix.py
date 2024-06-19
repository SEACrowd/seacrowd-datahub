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

from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses

_CITATION = """\
@inproceedings{sun-duh-2020-clirmatrix,
    title = "{CLIRM}atrix: A massively large collection of bilingual and multilingual datasets for Cross-Lingual Information Retrieval",
    author = "Sun, Shuo  and
      Duh, Kevin",
    editor = "Webber, Bonnie  and
      Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.340",
    doi = "10.18653/v1/2020.emnlp-main.340",
    pages = "4160--4170",
}
"""

_DATASETNAME = "clir_matrix"

_DESCRIPTION = """\
A massively large collection of bilingual and multilingual datasets for Cross-Lingual Information Retrieval extracted automatically from Wikipedia.
CLIRMatrix (Cross-Lingual Information Retrieval Matrix) comprises:
  (1) BI-139, a bilingual dataset of queries in one language matched with relevant documents in another language for 139x138=19,182 language pairs, and
  (2) MULTI-8, a multilingual dataset of queries and documents jointly aligned in 8 different languages.

Only (1) BI-139 has languages covered in SEACROWD.
"""

_HOMEPAGE = "https://github.com/ssun32/CLIRMatrix"

_LANGUAGES = ["tgl", "ilo", "min", "jav", "sun", "ceb", "vie", "tha"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_CLIR_LANG = {
    "tgl": "tl",
    "jav": "jv",
    "sun": "su",
    "vie": "vi",
    "tha": "th",
    "ilo": "ilo",
    "min": "min",
    "ceb": "ceb",
}
_URLS = {
    ds: {
        split: {(lque, ldoc): (f"https://www.cs.jhu.edu/~shuosun/clirmatrix/data/BI-139/{ds}/{_CLIR_LANG[lque]}/" f"{_CLIR_LANG[lque]}.{_CLIR_LANG[ldoc]}.{split}{'.base' if ds == 'base' else ''}.jl.gz") for lque, ldoc in permutations(_LANGUAGES, 2)}
        for split in ["train", "dev", "test1", "test2"]
    }
    for ds in ["base", "full"]
} | {"docs": {ldoc: f"https://www.cs.jhu.edu/~shuosun/clirmatrix/data/DOCS/{_CLIR_LANG[ldoc]}.tsv.gz" for ldoc in _LANGUAGES}}

_SUPPORTED_TASKS = []

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class CLIRMatrixDataset(datasets.GeneratorBasedBuilder):
    """Cross-Lingual Information Retrieval dataset of 49 million unique queries and 34 billion triplets."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        *[
            SEACrowdConfig(
                name=f"{_DATASETNAME}{subset}_source",  # refers to the `base` split in the original paper.
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} source schema",
                schema="source",
                subset_id=f"{_DATASETNAME}{subset}",
            )
            for subset in [f"{'_' if lque else ''}{lque}{'_' if ldoc else ''}{ldoc}" for lque, ldoc in [("", ""), *permutations(_LANGUAGES, 2)]]
        ],
        *[
            SEACrowdConfig(
                name=f"{_DATASETNAME}{subset}_full_source",  # refers to the `full` split in the original paper.
                version=datasets.Version(_SOURCE_VERSION),
                description=f"{_DATASETNAME} full subset source schema",
                schema="source",
                subset_id=f"{_DATASETNAME}{subset}_full",
            )
            for subset in [f"{'_' if lque else ''}{lque}{'_' if ldoc else ''}{ldoc}" for lque, ldoc in [("", ""), *permutations(_LANGUAGES, 2)]]
        ],
        # source-only dataloader
        # SEACrowdConfig(
        #     name=f"{_DATASETNAME}_seacrowd_pairs",
        #     version=SEACROWD_VERSION,
        #     description=f"{_DATASETNAME} SEACrowd schema",
        #     schema="seacrowd_pairs",
        #     subset_id=f"{_DATASETNAME}",
        # ),
        # SEACrowdConfig(
        #     name=f"{_DATASETNAME}_full_seacrowd_pairs",
        #     version=SEACROWD_VERSION,
        #     description=f"{_DATASETNAME} full subset SEACrowd schema",
        #     schema="seacrowd_pairs",
        #     subset_id=f"{_DATASETNAME}_full",
        # ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "src_id": datasets.Value("string"),
                    "src_query": datasets.Value("string"),
                    "tgt_results": [
                        {
                            "doc_id": datasets.Value("string"),
                            "score": datasets.Value("int32"),
                            "doc_text": datasets.Value("string"),
                        }
                    ],
                    "lang_query": datasets.Value("string"),
                    "lang_doc": datasets.Value("string"),
                }
            )

        # elif self.config.schema == "seacrowd_[seacrowdschema_name]":
        # source_only, skipping this.
        else:
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

        subset_id = self.config.subset_id.split("_")

        urls = _URLS["full" if subset_id[-1] == "full" else "base"]
        urls_doc = _URLS["docs"]

        # filter subset direction
        if len(subset_id) > 3:
            lque, ldoc = subset_id[2:4]
            urls = {split: {(lque, ldoc): v[(lque, ldoc)]} for split, v in urls.items()}
            urls_doc = {ldoc: urls_doc[ldoc]}

        data_paths = dl_manager.download_and_extract(urls)
        doc_paths = dl_manager.download_and_extract(urls_doc)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_paths["train"], "doc_paths": doc_paths},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_paths["test1"], "doc_paths": doc_paths},
            ),
            datasets.SplitGenerator(
                name="test2",  # just supplementary test sets for users to use in whatever way they want # just supplementary test sets for users to use in whatever way they want
                gen_kwargs={"filepath": data_paths["test2"], "doc_paths": doc_paths},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_paths["dev"], "doc_paths": doc_paths},
            ),
        ]

    def _generate_examples(self, filepath: Dict[Tuple, Path], doc_paths: Dict[str, Path]) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        docs_id2txt = {}
        for ldoc, p in doc_paths.items():
            docs_id2txt[ldoc] = pd.read_csv(p, sep="\t", dtype=str, header=None).set_index(0).iloc[:, 0]

        if self.config.schema == "source":
            for (lque, ldoc), fp in filepath.items():
                df = pd.read_json(fp, orient="records", lines=True)
                not_found = set()
                for idx, row in df.iterrows():
                    ret = row.to_dict()
                    for doc_id, score in ret["tgt_results"]:
                        if doc_id not in docs_id2txt[ldoc]:
                            not_found.add(doc_id)
                    ret["lang_query"] = lque
                    ret["lang_doc"] = ldoc
                    ret["tgt_results"] = [
                        {
                            "doc_id": doc_id,
                            "score": score,
                            "doc_text": docs_id2txt[ldoc].get(doc_id, ""),
                            # many doc_id discrepancy, i.e. not found in the tab-separated document files, in particular for Sundanese (sun);
                        }
                        for doc_id, score in ret["tgt_results"]
                    ]
                    yield f"{lque}_{ldoc}_{idx}", ret

        # source-only dataloader, skipping seacrowd schema.
        # elif self.config.schema == "seacrowd_[seacrowd_schema_name]":
