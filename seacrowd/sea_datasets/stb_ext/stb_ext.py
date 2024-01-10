import io

import conllu
import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (DEFAULT_SEACROWD_VIEW_NAME,
                                      DEFAULT_SOURCE_VIEW_NAME, Tasks, Licenses)

_DATASETNAME = "stb_ext"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME

_LANGUAGES = ["sg"]
_LOCAL = False
_CITATION = """\
@article{10.1145/3321128,
author = {Wang, Hongmin and Yang, Jie and Zhang, Yue},
title = {From Genesis to Creole Language: Transfer Learning for Singlish Universal Dependencies Parsing and POS Tagging},
year = {2019},
issue_date = {January 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {19},
number = {1},
issn = {2375-4699},
url = {https://doi.org/10.1145/3321128},
doi = {10.1145/3321128},
abstract = {Singlish can be interesting to the computational linguistics community both linguistically, as a major
 low-resource creole based on English, and computationally, for information extraction and sentiment analysis of
 regional social media. In our conference paper, Wang et al. (2017), we investigated part-of-speech (POS) tagging and
 dependency parsing for Singlish by constructing a treebank under the Universal Dependencies scheme and successfully
 used neural stacking models to integrate English syntactic knowledge for boosting Singlish POS tagging and dependency
 parsing, achieving the state-of-the-art accuracies of 89.50% and 84.47% for Singlish POS tagging and dependency,
 respectively. In this work, we substantially extend Wang et al. (2017) by enlarging the Singlish treebank to more
 than triple the size and with much more diversity in topics, as well as further exploring neural multi-task models
 for integrating English syntactic knowledge. Results show that the enlarged treebank has achieved significant
 relative error reduction of 45.8% and 15.5% on the base model, 27% and 10% on the neural multi-task model, and
 21% and 15% on the neural stacking model for POS tagging and dependency parsing, respectively. Moreover, the
 state-of-the-art Singlish POS tagging and dependency parsing accuracies have been improved to 91.16% and 85.57%,
 respectively. We make our treebanks and models available for further research.},
journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
month = {may},
articleno = {1},
numpages = {29},
keywords = {part-of-speech tagging, creole language, transfer learning, Singlish, neural stacking, Dependency parsing,
universal dependencies, multi-task network}
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
_STB_DATASETS = {
    "gold_pos": {
        "train": _PREFIX + "gold_pos/train.ext.conll",
    },
    "en_ud_autopos": {"train": _PREFIX + "en-ud-autopos/en-ud-train.conllu.autoupos",
                      "validation": _PREFIX + "en-ud-autopos/en-ud-dev.conllu.ann.auto.epoch24.upos",
                      "test": _PREFIX + "en-ud-autopos/en-ud-test.conllu.ann.auto.epoch24.upos"},
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
_POSTAGS = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
                               "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "root"]
_SUPPORTED_TASKS = [Tasks.POS_TAGGING, Tasks.DEPENDENCY_PARSING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "1.0.0"


def config_constructor(subset_id, schema, version):
    return SEACrowdConfig(
        name=f"{_DATASETNAME}_{subset_id}_{schema}",
        version=datasets.Version(version),
        description=_DESCRIPTION,
        schema=schema,
        subset_id=subset_id
    )


class StbExtDataset(datasets.GeneratorBasedBuilder):
    """This is a seacrowd dataloader for the STB-EXT dataset, which offers a 3-times larger training set, while keeping
    the same dev and test sets from STB-ACL. It provides treebanks with both gold-standard and automatically generated POS tags."""

    BUILDER_CONFIGS = [
        config_constructor(subset_id="auto_pos_stack", schema="source", version=_SOURCE_VERSION),
        config_constructor(subset_id="auto_pos_multiview", schema="source", version=_SOURCE_VERSION),
        config_constructor(subset_id="en_ud_autopos", schema="source", version=_SOURCE_VERSION),
        config_constructor(subset_id="gold_pos", schema="source", version=_SOURCE_VERSION)
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_gold_pos_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "idx": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "lemmas": datasets.Sequence(datasets.Value("string")),
                    "upos": datasets.Sequence(datasets.features.ClassLabel(names=_POSTAGS)),
                    "xpos": datasets.Sequence(datasets.Value("string")),
                    "feats": datasets.Sequence(datasets.Value("string")),
                    "head": datasets.Sequence(datasets.Value("string")),
                    "deprel": datasets.Sequence(datasets.Value("string")),
                    "deps": datasets.Sequence(datasets.Value("string")),
                    "misc": datasets.Sequence(datasets.Value("string")),
                }
            )
        elif self.config.schema == "seacrowd_seq_label":
            pass
        elif self.config.schema == "seacrowd_kb":
            pass
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
        urls = _STB_DATASETS[self.config.name.subset_id]
        downloaded_files = dl_manager.download_and_extract(urls)
        splits = []
        if "train" in downloaded_files:
            splits.append(
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}))
        if "validation" in downloaded_files:
            splits.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                                  gen_kwargs={"filepath": downloaded_files["validation"]}))
        if "test" in downloaded_files:
            splits.append(
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}))
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
            for idx, sent in enumerate(tokenlist):
                idx = sent.metadata["sent_id"] if "sent_id" in sent.metadata else idx
                tokens = [token["form"] for token in sent]
                txt = sent.metadata["text"] if "text" in sent.metadata else " ".join(tokens)
                if self.config.schema == "source":
                    yield idx, {
                        "idx": str(idx),
                        "text": txt,
                        "tokens": [token["form"] for token in sent],
                        "lemmas": [token["lemma"] for token in sent],
                        "upos": [token["upos"] for token in sent],
                        "xpos": [token["xpos"] for token in sent],
                        "feats": [str(token["feats"]) for token in sent],
                        "head": [str(token["head"]) for token in sent],
                        "deprel": [str(token["deprel"]) for token in sent],
                        "deps": [str(token["deps"]) for token in sent],
                        "misc": [str(token["misc"]) for token in sent],
                    }
                if self.config.schema == "seacrowd_seq_label":
                    pass
                if self.config.schema == "seacrowd_kb":
                    pass