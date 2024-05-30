# coding=utf-8


from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{asgari2020unisent,
title={UniSent: Universal Adaptable Sentiment Lexica for 1000+ Languages},
author={Asgari, Ehsaneddin and Braune, Fabienne and Ringlstetter, Christoph and Mofrad, Mohammad RK},
booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC-2020)},
year={2020},
organization={European Language Resources Association (ELRA)}
}
"""
_DATASETNAME = "unisent"
_DESCRIPTION = """\
UniSent is a universal sentiment lexica for 1000+ languages.
To build UniSent, the authors use a massively parallel Bible
corpus to project sentiment information from English to other
languages for sentiment analysis on Twitter data. 183 of 1404
languages are spoken in Southeast Asia
"""
URL = "https://raw.githubusercontent.com/ehsanasgari/UniSent/master/unisent_lexica_v1/{}_unisent_lexicon.txt"
_HOMEPAGE = "https://github.com/ehsanasgari/UniSent"
_LANGUAGES = [
    "aaz",
    "abx",
    "ace",
    "agn",
    "agt",
    "ahk",
    "akb",
    "alj",
    "alp",
    "amk",
    "aoz",
    "atb",
    "atd",
    "att",
    "ban",
    "bbc",
    "bcl",
    "bgr",
    "bgs",
    "bgz",
    "bhp",
    "bkd",
    "bku",
    "blw",
    "blz",
    "bpr",
    "bps",
    "bru",
    "btd",
    "bth",
    "bto",
    "bts",
    "btx",
    "bug",
    "bvz",
    "bzi",
    "cbk",
    "ceb",
    "cfm",
    "cgc",
    "clu",
    "cmo",
    "cnh",
    "cnw",
    "csy",
    "ctd",
    "czt",
    "dgc",
    "dtp",
    "due",
    "duo",
    "ebk",
    "fil",
    "gbi",
    "gor",
    "heg",
    "hil",
    "hnj",
    "hnn",
    "hvn",
    "iba",
    "ifa",
    "ifb",
    "ifk",
    "ifu",
    "ify",
    "ilo",
    "ind",
    "iry",
    "isd",
    "itv",
    "ium",
    "ivb",
    "ivv",
    "jav",
    "jra",
    "kac",
    "khm",
    "kix",
    "kje",
    "kmk",
    "kne",
    "kqe",
    "krj",
    "ksc",
    "ksw",
    "kxm",
    "lao",
    "lbk",
    "lew",
    "lex",
    "lhi",
    "lhu",
    "ljp",
    "lus",
    "mad",
    "mak",
    "mbb",
    "mbd",
    "mbf",
    "mbi",
    "mbs",
    "mbt",
    "mej",
    "mkn",
    "mnb",
    "mog",
    "mqj",
    "mqy",
    "mrw",
    "msb",
    "msk",
    "msm",
    "mta",
    "mtg",
    "mtj",
    "mvp",
    "mwq",
    "mwv",
    "mya",
    "nbe",
    "nfa",
    "nia",
    "nij",
    "nlc",
    "npy",
    "obo",
    "pag",
    "pam",
    "plw",
    "pmf",
    "pne",
    "ppk",
    "prf",
    "prk",
    "ptu",
    "pww",
    "sas",
    "sbl",
    "sda",
    "sgb",
    "smk",
    "sml",
    "sun",
    "sxn",
    "szb",
    "tbl",
    "tby",
    "tcz",
    "tdt",
    "tgl",
    "tha",
    "tih",
    "tlb",
    "twu",
    "urk",
    "vie",
    "war",
    "whk",
    "wrs",
    "xbr",
    "yli",
    "yva",
    "zom",
    "zyp",
    "pse",
    "mnx",
    "mmn",
    "lsi",
    "hlt",
    "gdg",
    "bnj",
    "acn",
]

_LICENSE = Licenses.CC_BY_NC_ND_4_0.value  # cc-by-nc-nd-4.0
_LOCAL = False
_URLS = {
    _DATASETNAME: {lang: URL.format(lang) for lang in _LANGUAGES},
}

_SUPPORTED_TASKS = [Tasks.SENTIMENT_ANALYSIS]

_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


class UniSentDataset(datasets.GeneratorBasedBuilder):
    LABELS = ["NEGATIVE", "POSITIVE"]

    BUILDER_CONFIGS = [SEACrowdConfig(name=f"{_DATASETNAME}_{lang}_source", version=datasets.Version(_SOURCE_VERSION), description=_DESCRIPTION, schema="source", subset_id=f"{_DATASETNAME}_{lang}",) for lang in _LANGUAGES] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{lang}_seacrowd_text",
            version=datasets.Version(_SEACROWD_VERSION),
            description=_DESCRIPTION,
            schema="seacrowd_text",
            subset_id=f"{_DATASETNAME}_{lang}",
        )
        for lang in _LANGUAGES
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        features = None
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "word": datasets.Value("string"),
                    "lexicon": datasets.Value("string"),
                }
            )
        elif self.config.schema == "seacrowd_text":
            features = schemas.text_features(label_names=self.LABELS)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        lang = self.config.subset_id.split("_")[-1]
        urls = _URLS[_DATASETNAME][lang]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        with open(filepath, "r", encoding="utf-8") as filein:
            data_instances = [inst.strip("\n").split("\t") for inst in filein.readlines()]

        for di_idx, data_instance in enumerate(data_instances):
            word, lexicon = data_instance
            if self.config.schema == "source":
                yield di_idx, {"word": word, "lexicon": lexicon}
            elif self.config.schema == "seacrowd_text":
                yield di_idx, {"id": di_idx, "text": word, "label": self.LABELS[self._clip_label(int(lexicon))]}

    @staticmethod
    def _clip_label(label: int) -> int:
        """
        Original labels are -1, +1.
        Clip the label to 0 or 1 to get right index.
        """
        return 0 if int(label) < 0 else 1
