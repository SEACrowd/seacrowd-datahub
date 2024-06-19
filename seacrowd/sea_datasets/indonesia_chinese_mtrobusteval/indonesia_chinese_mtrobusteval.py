from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import jsonlines
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@article{,
  author    = {supryzhu},
  title     = {Indonesia-Chinese-MTRobustEval},
  journal   = {None},
  volume    = {None},
  year      = {2023},
  url       = {https://github.com/supryzhu/Indonesia-Chinese-MTRobustEval},
  doi       = {None},
  biburl    = {None},
  bibsource = {None}
}
"""


_DATASETNAME = "indonesia_chinese_mtrobusteval"

_DESCRIPTION = """\
The dataset is curated for the purpose of evaluating the robustness of Neural Machine Translation (NMT) towards natural occuring noise
(typo, slang, code switching, etc.). The dataset is crawled from Twitter, then pre-processed to obtain sentences with noise.
The dataset consists of a thousand noisy sentences. The dataset is translated into Chinese manually as the benchmark for evaluating the robustness of NMT.
"""

_HOMEPAGE = "https://github.com/supryzhu/Indonesia-Chinese-MTRobustEval"

_LANGUAGES = ["ind", "cmn"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)


_LICENSE = Licenses.MIT.value  # example: Licenses.MIT.value, Licenses.CC_BY_NC_SA_4_0.value, Licenses.UNLICENSE.value, Licenses.UNKNOWN.value

_LOCAL = False

_URLS = {
    _DATASETNAME: "https://github.com/supryzhu/Indonesia-Chinese-MTRobustEval/raw/main/data/Indonesia-Chinese.xlsx",
}

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class IndonesiaChineseMtRobustEval(datasets.GeneratorBasedBuilder):
    """The dataset consists of a thousand noisy sentences. The dataset is translated into Chinese manually as the benchmark for evaluating the robustness of NMT."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description="indonesia_chinese_mtrobusteval source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_t2t",
            version=SEACROWD_VERSION,
            description="indonesia_chinese_mtrobusteval SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "src": datasets.Value("string"),
                    "tgt": datasets.Value("string"),
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
        file_path = dl_manager.download(urls)
        df = pd.read_excel(file_path)
        src = df["Indonesia"].tolist()
        tgt = df["Chinese"].tolist()
        results = []
        for i, item in enumerate(src):
            results.append({"id": str(i), "src": item, "tgt": tgt[i]})
        self._write_jsonl(file_path + ".jsonl", results)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": file_path + ".jsonl",
                    "split": "train",
                },
            )
        ]


    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        if self.config.schema == "source":
            i = 0
            with jsonlines.open(filepath) as f:
                for each_data in f.iter():
                    ex = {
                        "id": each_data["id"],
                        "src": each_data["src"],
                        "tgt": each_data["tgt"],
                    }
                    yield i, ex
                    i += 1

        elif self.config.schema == "seacrowd_t2t":
            i = 0
            with jsonlines.open(filepath) as f:
                for each_data in f.iter():
                    ex = {"id": each_data["id"], "text_1": each_data["src"], "text_2": each_data["tgt"], "text_1_name": "ind", "text_2_name": "cmn"}
                    yield i, ex
                    i += 1

    def _write_jsonl(self, filepath, values):
        with jsonlines.open(filepath, "w") as writer:
            for line in values:
                writer.write(line)

