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

"""
The LTI LangID corpus is a dataset for language identification.
The most recent version, v5, contains training data for 1266 languages, and some (possibly very tiny) amount of text for a total of 1706 languages.
"""
import os
import subprocess
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{brown2014non,
  title={Non-linear mapping for improved identification of 1300+ languages},
  author={Brown, Ralf D},
  booktitle={Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={627--632},
  year={2014}
}
"""

_DATASETNAME = "lti_langid_corpus"

_DESCRIPTION = """\
The LTI LangID corpus is a dataset for language identification.
The most recent version, v5, contains training data for 1266 languages, and some (possibly very tiny) amount of text for a total of 1706 languages.
This dataloader can only be executed in a BASH environment at the moment. (See https://github.com/SEACrowd/seacrowd-datahub/pull/405)
"""

_HOMEPAGE = "https://www.cs.cmu.edu/~ralf/langid.html"

_LANGUAGES = [
    "ifa",
    "ace",
    "btm",
    "mqj",
    "mbs",
    "nbq",
    "gor",
    "zyp",
    "tbl",
    "kjp",
    "kmk",
    "kqe",
    "ptu",
    "blw",
    "ceb",
    "prf",
    "yva",
    "zlm",
    "bps",
    "tdt",
    "mya",
    "dgc",
    "lus",
    "wrs",
    "abx",
    "rgu",
    "aaz",
    "agn",
    "ccp",
    "cmn",
    "jav",
    "obo",
    "due",
    "msk",
    "xsb",
    "syb",
    "ind",
    "lbk",
    "min",
    "smk",
    "att",
    "nod",
    "tdj",
    "atb",
    "atd",
    "pag",
    "hvn",
    "ksc",
    "lao",
    "kkl",
    "lti",
    "dao",
    "cgc",
    "tbk",
    "gdg",
    "amk",
    "mbd",
    "clu",
    "msb",
    "sbl",
    "cek",
    "khm",
    "yue",
    "sgb",
    "beu",
    "eip",
    "ifu",
    "mnw",
    "suc",
    "bgs",
    "pam",
    "ebk",
    "eng",
    "nfa",
    "cbk",
    "ify",
    "csy",
    "heg",
    "shn",
    "mta",
    "mbt",
    "lex",
    "tha",
    "mmn",
    "sun",
    "vie",
    "llg",
    "xnn",
    "txq",
    "bcl",
    "kje",
    "kne",
    "san",
    "hlt",
    "kyu",
    "bkd",
    "duo",
    "tet",
    "ury",
    "yka",
    "bjn",
    "tiy",
    "ivv",
    "agt",
    "ban",
    "blz",
    "mbi",
    "ilo",
    "mkn",
    "isd",
    "cth",
    "bpr",
    "por",
    "mbb",
    "tgl",
    "msm",
    "ivb",
    "tam",
    "plw",
    "alp",
    "row",
]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URL = "http://sourceforge.net/projects/la-strings/files/Language-Data/LTI-LangID-rel5.txz/download"

_SUPPORTED_TASKS = [Tasks.LANGUAGE_IDENTIFICATION]

_SOURCE_VERSION = "5.0.0"
_SEACROWD_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class LTILangIDDataset(datasets.GeneratorBasedBuilder):
    """LTI LangID corpus is a dataset for language identification for 1266 languages."""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"text": datasets.Value("string"), "language": datasets.Value("string")})

        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(label_names=_LANGUAGES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(data_dir, "MIL-TALE", "5")

        datasplits_dir = os.path.join(data_dir, "datasplits")
        if not Path(datasplits_dir).exists() or not Path(os.path.join(datasplits_dir, "DONE")).exists():
            # Run provided install.sh to generate train/dev/test splits in "datasplits" folder
            subprocess.call([os.path.join(data_dir, "code", "install.sh"), datasplits_dir])
            with open(os.path.join(datasplits_dir, "DONE"), "w"):
                pass

        train_filepaths = []
        dev_filepaths = []
        test_filepaths = []

        dataset_dir = os.path.join(data_dir, "datasplits")

        for lang_id in _LANGUAGES:
            train_filepaths.append(
                (
                    lang_id,
                    glob(os.path.join(dataset_dir, "train", f"{lang_id}*")),
                )
            )
            dev_filepaths.append(
                (
                    lang_id,
                    glob(os.path.join(dataset_dir, "devtest", f"{lang_id}*")),
                )
            )
            test_filepaths.append(
                (
                    lang_id,
                    glob(os.path.join(dataset_dir, "test", f"{lang_id}*")),
                )
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "all_filepaths": train_filepaths,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "all_filepaths": dev_filepaths,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "all_filepaths": test_filepaths,
                },
            ),
        ]

    def _generate_examples(self, all_filepaths: List[Path]) -> Tuple[int, Dict]:

        if self.config.schema == "source":
            key = 0
            for lang_id, filepaths in all_filepaths:
                for filepath in filepaths:
                    try:
                        for line in open(filepath):
                            text = line.strip().replace("\u200d", " ").replace("\u200b", " ")
                            yield key, {"text": text, "language": lang_id}
                            key += 1
                    except UnicodeDecodeError:
                        continue

        elif self.config.schema == "seacrowd_text":
            key = 0
            for lang_id, filepaths in all_filepaths:
                for filepath in filepaths:
                    try:
                        for line in open(filepath):
                            text = line.strip().replace("\u200d", " ").replace("\u200b", " ")
                            yield key, {"id": f"{filepath.split('/')[-1]}_{key}", "text": text, "label": lang_id}
                            key += 1
                    except UnicodeDecodeError:
                        continue
