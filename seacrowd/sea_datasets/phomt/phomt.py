import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{PhoMT,
title     = {{PhoMT: A High-Quality and Large-Scale Benchmark Dataset for Vietnamese-English Machine Translation}},
author    = {Long Doan and Linh The Nguyen and Nguyen Luong Tran and Thai Hoang and Dat Quoc Nguyen},
booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
year      = {2021},
pages     = {4495--4503}
}
"""

_DATASETNAME = "phomt"


_DESCRIPTION = """\
PhoMT is a high-quality and large-scale Vietnamese-English parallel dataset of 3.02M sentence pairs, which is 2.9M
pairs larger than the benchmark Vietnamese-English machine translation corpus IWSLT15. This is the first large-scale
Vietnamese-English machine translation study.
"""

_LANGUAGES = ["vie", "eng"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = True

_HOMEPAGE = "https://github.com/VinAIResearch/PhoMT"

_LICENSE = Licenses.MIT.value

_SUPPORTED_TASKS = [Tasks.MACHINE_TRANSLATION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

MAP_LANG = {"eng": "en", "vie": "vi"}


def seacrowd_config_constructor(src_lang, tgt_lang, schema, version):
    if src_lang == "" or tgt_lang == "":
        raise ValueError(f"Invalid src_lang {src_lang} or tgt_lang {tgt_lang}")

    if schema not in ["source", "seacrowd_t2t"]:
        raise ValueError(f"Invalid schema: {schema}")

    return SEACrowdConfig(
        name="phomt_{src}_{tgt}_{schema}".format(src=src_lang, tgt=tgt_lang, schema=schema),
        version=datasets.Version(version),
        description="phomt schema for {schema} from {src} to {tgt}".format(schema=schema, src=src_lang, tgt=tgt_lang),
        schema=schema,
        subset_id="phomt_{src}_{tgt}".format(src=src_lang, tgt=tgt_lang),
    )


class PhoMT(datasets.GeneratorBasedBuilder):
    """
    PhoMT is a high-quality and large-scale Vietnamese-English parallel dataset of 3.02M sentence pairs, which is
    2.9M pairs larger than the benchmark Vietnamese-English machine translation corpus IWSLT15.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        seacrowd_config_constructor("eng", "vie", "source", _SOURCE_VERSION),
        seacrowd_config_constructor("eng", "vie", "seacrowd_t2t", _SEACROWD_VERSION),
    ]

    DEFAULT_CONFIG_NAME = "phomt_eng_vie_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema in ("source", "seacrowd_t2t"):
            features = schemas.text2text_features
        else:
            raise ValueError(f"Invalid config schema: {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "detokenization", "train", "train.{lang}")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "detokenization", "dev", "dev.{lang}")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "detokenization", "test", "test.{lang}")},
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        config_names_split = self.config.name.split("_")
        src_lang = config_names_split[1]
        tgt_lang = config_names_split[2]

        src_path = filepath.format(lang=MAP_LANG[src_lang])
        tgt_path = filepath.format(lang=MAP_LANG[tgt_lang])

        with open(src_path, "r", encoding="utf8") as f:
            src_lines = f.readlines()
        with open(tgt_path, "r", encoding="utf8") as f:
            tgt_lines = f.readlines()

        if self.config.schema in ("source", "seacrowd_t2t"):
            for idx, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
                ex = {
                    "id": str(idx),
                    "text_1": src_line.strip(),
                    "text_2": tgt_line.strip(),
                    "text_1_name": src_lang,
                    "text_2_name": tgt_lang,
                }
                yield idx, ex

        else:
            raise NotImplementedError(f"Schema '{self.config.schema}' is not defined.")
