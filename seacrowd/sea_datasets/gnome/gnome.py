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

from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import requests

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import SCHEMA_TO_FEATURES, TASK_TO_SCHEMA, Licenses, Tasks

_CITATION = r"""\
@inproceedings{tiedemann-2012-parallel,
    title = "Parallel Data, Tools and Interfaces in {OPUS}",
    author = {Tiedemann, J{\"o}rg},
    editor = "Calzolari, Nicoletta  and
        Choukri, Khalid  and
        Declerck, Thierry  and
        Do{\u{g}}an, Mehmet U{\u{g}}ur  and
        Maegaard, Bente  and
        Mariani, Joseph  and
        Moreno, Asuncion  and
        Odijk, Jan  and
        Piperidis, Stelios",
    booktitle = "Proceedings of the Eighth International Conference on Language
    Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf",
    pages = "2214--2218",
    abstract = "This paper presents the current status of OPUS, a growing
    language resource of parallel corpora and related tools. The focus in OPUS
    is to provide freely available data sets in various formats together with
    basic annotation to be useful for applications in computational linguistics,
    translation studies and cross-linguistic corpus studies. In this paper, we
    report about new data sets and their features, additional annotation tools
    and models provided from the website and essential interfaces and on-line
    services included in the project.",
}
"""

_DATASETNAME = "gnome"

_DESCRIPTION = """\
A parallel corpus of GNOME localization files, which contains the interface text
in the GNU Network Object Model Environment (GNOME) and published by GNOME
translation teams. Text in this dataset is relatively short and technical.
"""

_HOMEPAGE = "https://opus.nlpl.eu/GNOME/corpus/version/GNOME"

_LANGUAGES = ["eng", "vie", "mya", "ind", "tha", "tgl", "zlm", "lao"]
_SUBSETS = ["en", "vi", "my", "id", "th", "tl", "ms", "lo"]
_SUBSET_PAIRS = [(src, tgt) for src in _SUBSETS for tgt in _SUBSETS if src != tgt]

_LICENSE = Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    "api": "http://opus.nlpl.eu/opusapi/?source={src_lang}&target={tgt_lang}&corpus=GNOME&version=v1",
    "data": "https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/{lang_pair}.txt.zip",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]
_SEACROWD_SCHEMA = f"seacrowd_{TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()}"  # t2t

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class GnomeDataset(datasets.GeneratorBasedBuilder):
    """A parallel corpus of GNOME localization files"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = []
    for subset in _SUBSET_PAIRS:
        lang_pair = f"{subset[0]}-{subset[1]}"
        BUILDER_CONFIGS += [
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang_pair}_source",
                version=SOURCE_VERSION,
                description=f"{_DATASETNAME} {lang_pair} source schema",
                schema="source",
                subset_id=lang_pair,
            ),
            SEACrowdConfig(
                name=f"{_DATASETNAME}_{lang_pair}_{_SEACROWD_SCHEMA}",
                version=SEACROWD_VERSION,
                description=f"{_DATASETNAME} {lang_pair} SEACrowd schema",
                schema=_SEACROWD_SCHEMA,
                subset_id=lang_pair,
            ),
        ]

    DEFAULT_CONFIG_NAME = (
        f"{_DATASETNAME}_{_SUBSET_PAIRS[0][0]}-{_SUBSET_PAIRS[0][1]}_source"
    )

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "source": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            )
        elif self.config.schema == _SEACROWD_SCHEMA:
            features = SCHEMA_TO_FEATURES[
                TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]]
            ]  # text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        src_lang, tgt_lang = self.config.subset_id.split("-")
        api_url = _URLS["api"].format(src_lang=src_lang, tgt_lang=tgt_lang)
        data_url = None

        response = requests.get(api_url, timeout=10)
        if response:
            corpora = response.json()["corpora"]
            for corpus in corpora:
                if ".txt.zip" in corpus["url"]:
                    data_url = corpus["url"]
                    break
        else:
            raise requests.exceptions.HTTPError(
                f"Non-success status code: {response.status_code}"
            )

        if not data_url:
            raise ValueError(f"No suitable corpus found, check {api_url}")
        else:
            lang_pair = data_url.split("/")[-1].split(".")[0]
            data_dir = Path(dl_manager.download_and_extract(data_url))
            src_file = data_dir / f"GNOME.{lang_pair}.{src_lang}"
            tgt_file = data_dir / f"GNOME.{lang_pair}.{tgt_lang}"

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "src_file": src_file,
                        "tgt_file": tgt_file,
                    },
                ),
            ]

    def _generate_examples(self, src_file: Path, tgt_file: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with open(src_file, "r", encoding="utf-8") as src_f, open(
            tgt_file, "r", encoding="utf-8"
        ) as tgt_f:
            for idx, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
                if self.config.schema == "source":
                    yield idx, {"source": src_line.strip(), "target": tgt_line.strip()}
                elif self.config.schema == _SEACROWD_SCHEMA:
                    yield idx, {
                        "id": str(idx),
                        "text_1": src_line.strip(),
                        "text_2": tgt_line.strip(),
                        "text_1_name": f"source ({src_file.name.split('.')[-1]})",
                        "text_2_name": f"target ({tgt_file.name.split('.')[-1]})",
                    }
