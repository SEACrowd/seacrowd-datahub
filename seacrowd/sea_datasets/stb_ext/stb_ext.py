import io

import conllu
import datasets

from seacrowd.utils.common_parser import load_ud_data_as_seacrowd_kb
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils import schemas
from seacrowd.utils.constants import DEFAULT_SEACROWD_VIEW_NAME, DEFAULT_SOURCE_VIEW_NAME, Licenses, Tasks

_DATASETNAME = "stb_ext"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME

_LANGUAGES = ["eng"]
_LOCAL = False
_CITATION = """\
@article{wang2019genesis,
title={From genesis to creole language: Transfer learning for singlish universal dependencies parsing and POS tagging},
author={Wang, Hongmin and Yang, Jie and Zhang, Yue},
journal={ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)},
volume={19},
number={1},
pages={1--29},
year={2019},
publisher={ACM New York, NY, USA}
}
"""

_DESCRIPTION = """\
We adopt the Universal Dependencies protocol for constructing the Singlish dependency treebank, both as a new resource
for the low-resource languages and to facilitate knowledge transfer from English. Briefly, the STB-EXT dataset offers
a 3-times larger training set, while keeping the same dev and test sets from STB-ACL. We provide treebanks with both
gold-standard as well as automatically generated POS tags.
"""

_HOMEPAGE = "https://github.com/wanghm92/Sing_Par/tree/master/TALLIP19_dataset/treebank"

_LICENSE = Licenses.MIT.value

_PREFIX = "https://raw.githubusercontent.com/wanghm92/Sing_Par/master/TALLIP19_dataset/treebank/"
_URLS = {
    "gold_pos": {
        "train": _PREFIX + "gold_pos/train.ext.conll",
    },
    "en_ud_autopos": {"train": _PREFIX + "en-ud-autopos/en-ud-train.conllu.autoupos", "validation": _PREFIX + "en-ud-autopos/en-ud-dev.conllu.ann.auto.epoch24.upos", "test": _PREFIX + "en-ud-autopos/en-ud-test.conllu.ann.auto.epoch24.upos"},
    "auto_pos_multiview": {
        "train": _PREFIX + "auto_pos/multiview/train.autopos.multiview.conll",
        "validation": _PREFIX + "auto_pos/multiview/dev.autopos.multiview.conll",
        "test": _PREFIX + "auto_pos/multiview/test.autopos.multiview.conll",
    },
    "auto_pos_stack": {
        "train": _PREFIX + "auto_pos/stack/train.autopos.stack.conll",
        "validation": _PREFIX + "auto_pos/stack/dev.autopos.stack.conll",
        "test": _PREFIX + "auto_pos/stack/test.autopos.stack.conll",
    },
}
_POSTAGS = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "root"]
_SUPPORTED_TASKS = [Tasks.POS_TAGGING, Tasks.DEPENDENCY_PARSING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


def config_constructor(subset_id, schema, version):
    return SEACrowdConfig(name=f"{_DATASETNAME}_{subset_id}_{schema}",
                          version=datasets.Version(version), description=_DESCRIPTION,
                          schema=schema, subset_id=subset_id)


class StbExtDataset(datasets.GeneratorBasedBuilder):
    """This is a seacrowd dataloader for the STB-EXT dataset, which offers a 3-times larger training set, while keeping
    the same dev and test sets from STB-ACL. It provides treebanks with both gold-standard and automatically generated POS tags."""

    BUILDER_CONFIGS = [
        # source
        config_constructor(subset_id="auto_pos_stack", schema="source", version=_SOURCE_VERSION),
        config_constructor(subset_id="auto_pos_multiview", schema="source", version=_SOURCE_VERSION),
        config_constructor(subset_id="en_ud_autopos", schema="source", version=_SOURCE_VERSION),
        config_constructor(subset_id="gold_pos", schema="source", version=_SOURCE_VERSION),
        # seq_label
        config_constructor(subset_id="auto_pos_stack", schema="seacrowd_seq_label", version=_SEACROWD_VERSION),
        config_constructor(subset_id="auto_pos_multiview", schema="seacrowd_seq_label", version=_SEACROWD_VERSION),
        config_constructor(subset_id="en_ud_autopos", schema="seacrowd_seq_label", version=_SEACROWD_VERSION),
        config_constructor(subset_id="gold_pos", schema="seacrowd_seq_label", version=_SEACROWD_VERSION),
        # dependency parsing
        config_constructor(subset_id="auto_pos_stack", schema="seacrowd_kb", version=_SEACROWD_VERSION),
        config_constructor(subset_id="auto_pos_multiview", schema="seacrowd_kb", version=_SEACROWD_VERSION),
        config_constructor(subset_id="en_ud_autopos", schema="seacrowd_kb", version=_SEACROWD_VERSION),
        config_constructor(subset_id="gold_pos", schema="seacrowd_kb", version=_SEACROWD_VERSION),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_gold_pos_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    # metadata
                    "sent_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_en": datasets.Value("string"),
                    # tokens
                    "id": [datasets.Value("string")],
                    "form": [datasets.Value("string")],
                    "lemma": [datasets.Value("string")],
                    "upos": [datasets.Value("string")],
                    "xpos": [datasets.Value("string")],
                    "feats": [datasets.Value("string")],
                    "head": [datasets.Value("string")],
                    "deprel": [datasets.Value("string")],
                    "deps": [datasets.Value("string")],
                    "misc": [datasets.Value("string")],
                }
            )
        elif self.config.schema == "seacrowd_seq_label":
            features = schemas.seq_label_features(label_names=_POSTAGS)
        elif self.config.schema == "seacrowd_kb":
            features = schemas.kb_features
        else:
            raise ValueError(f"Invalid config: {self.config.schema}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """ "return splitGenerators"""
        urls = _URLS[self.config.subset_id]
        downloaded_files = dl_manager.download_and_extract(urls)
        splits = []
        if "train" in downloaded_files:
            splits.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}))
        if "validation" in downloaded_files:
            splits.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["validation"]}))
        if "test" in downloaded_files:
            splits.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}))
        return splits

    def _generate_examples(self, filepath):
        def process_buffer(TextIO):
            BOM = "\ufeff"
            buffer = io.StringIO()
            for line in TextIO:
                line = line.replace(BOM, "") if BOM in line else line
                buffer.write(line)
            buffer.seek(0)
            return buffer

        with open(filepath, "r", encoding="utf-8") as data_file:
            tokenlist = list(conllu.parse_incr(process_buffer(data_file)))
            data_instances = []
            for idx, sent in enumerate(tokenlist):
                idx = sent.metadata["sent_id"] if "sent_id" in sent.metadata else idx
                tokens = [token["form"] for token in sent]
                txt = sent.metadata["text"] if "text" in sent.metadata else " ".join(tokens)
                example = {
                    # meta
                    "sent_id": str(idx),
                    "text": txt,
                    "text_en": txt,
                    # tokens
                    "id": [token["id"] for token in sent],
                    "form": [token["form"] for token in sent],
                    "lemma": [token["lemma"] for token in sent],
                    "upos": [token["upos"] for token in sent],
                    "xpos": [token["xpos"] for token in sent],
                    "feats": [str(token["feats"]) for token in sent],
                    "head": [str(token["head"]) for token in sent],
                    "deprel": [str(token["deprel"]) for token in sent],
                    "deps": [str(token["deps"]) for token in sent],
                    "misc": [str(token["misc"]) for token in sent]
                }
                data_instances.append(example)

            if self.config.schema == "source":
                pass
            if self.config.schema == "seacrowd_seq_label":
                data_instances = list(
                    map(
                        lambda d: {
                            "id": d["sent_id"],
                            "tokens": d["form"],
                            "labels": d["upos"],
                        },
                        data_instances,
                    )
                )
            if self.config.schema == "seacrowd_kb":
                data_instances = load_ud_data_as_seacrowd_kb(filepath, data_instances)
            for key, exam in enumerate(data_instances):
                yield key, exam
