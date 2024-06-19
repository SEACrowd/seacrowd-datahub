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
The dataset consists of individual news articles, each corresponding to a unique URL at the
Thai government website (https://www.thaigov.go.th/). The dataset structure is as follows: a topic header is
followed by the content of the news article, which is then succeeded by a blank line and the source URL
"""
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import jsonlines

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{,
  author    = {PyThaiNLP},
  title     = {thaigov-v2-corpus},
  journal   = {},
  volume    = {},
  year      = {2023},
  url       = {https://github.com/PyThaiNLP/thaigov-v2-corpus/tree/master},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "thaigov"

_DESCRIPTION = """\
This dataset is a corpus from ThaiGov.
"""

_HOMEPAGE = "https://github.com/PyThaiNLP/thaigov-v2-corpus/tree/master/data"

_LANGUAGES = ["tha"]

_LICENSE = Licenses.PDDL.value

_LOCAL = False


_URLS = {
    _DATASETNAME: "https://github.com/PyThaiNLP/thaigov-v2-corpus/archive/refs/heads/master.zip",
}

_SUPPORTED_TASKS = [Tasks.SUMMARIZATION]

_SOURCE_VERSION = "2.0.0"

_SEACROWD_VERSION = "2024.06.20"


class NewDataset(datasets.GeneratorBasedBuilder):
    """This dataset is a corpus from ThaiGov, can be used for summarization tasks."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="thaigov_source",
            version=SOURCE_VERSION,
            description="thaigov source schema",
            schema="source",
            subset_id="thaigov",
        ),
        SEACrowdConfig(
            name="thaigov_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description="thaigov SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id="thaigov",
        ),
    ]

    DEFAULT_CONFIG_NAME = "thaigov_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "src": datasets.Value("string"),
                    "tgt": datasets.Value("string"),
                    "url": datasets.Value("string"),
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
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        # Since the data is stored based on date extracted, it will follow the pattern data/year/month/day/{article_names}.txt
        list_all_txt_files = list(glob.glob(os.path.join(data_dir, "thaigov-v2-corpus-master", "data", "*", "*", "*", "*.txt")))
        all_data = []
        counter = 0
        for i in list_all_txt_files:
            d = self._read_file(i)
            all_data.append({"id": counter, "src": d["context"], "tgt": d["title"], "url": d["url"]})
            counter += 1

        self._write_jsonl(data_dir + "/train.jsonl", all_data)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            i = 0
            with jsonlines.open(filepath) as f:
                for each_data in f.iter():
                    ex = {
                        "id": each_data["id"],
                        "src": each_data["src"],
                        "tgt": each_data["tgt"],
                        "url": each_data["url"],
                    }
                    yield i, ex
                    i += 1

        elif self.config.schema == "seacrowd_t2t":
            i = 0
            with jsonlines.open(filepath) as f:
                for each_data in f.iter():
                    ex = {"id": each_data["id"], "text_1": each_data["src"], "text_2": each_data["tgt"], "text_1_name": "input_document", "text_2_name": "output_summary"}
                    yield i, ex
                    i += 1

    def _read_file(self, path):
        text = {"title": "", "context": "", "url": ""}
        page_view_line = 0
        with open(path, "r", encoding="utf-8-sig") as f:
            for n, line in enumerate(f):
                line = line.strip()
                if n == 0:  # title line
                    text["title"] = line.strip()
                else:
                    if line:
                        if re.match(r"^[\d,]+$", line):
                            page_view_line = n
                            continue
                        if line == "พิมพ์" or page_view_line and page_view_line < n:  # skip 'print'
                            continue
                        if re.match(r"^ที่มา : http", line):
                            text["url"] = line.strip().split(" ")[-1]
                        else:
                            text["context"] += line.strip().replace("\xa0", "") + "\n"
        return text

    def _write_jsonl(self, filepath, values):
        with jsonlines.open(filepath, "w") as writer:
            for line in values:
                writer.write(line)
