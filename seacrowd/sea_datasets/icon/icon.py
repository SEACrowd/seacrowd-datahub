import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
import nltk
from nltk import Tree
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (DEFAULT_SEACROWD_VIEW_NAME,
                                      DEFAULT_SOURCE_VIEW_NAME, Licenses,
                                      Tasks)

_DATASETNAME = "icon"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME
_CITATION = """\
@inproceedings{lim2023icon,
  title={ICON: Building a Large-Scale Benchmark Constituency Treebank for the Indonesian Language},
  author={Lim, Ee Suan and Leong, Wei Qi and Nguyen, Ngan Thanh and Adhista, Dea and Kng, Wei Ming and Tjh, William Chandra and Purwarianti, Ayu},
  booktitle={Proceedings of the 21st International Workshop on Treebanks and Linguistic Theories (TLT, GURT/SyntaxFest 2023)},
  pages={37--53},
  year={2023}
}
"""

_DESCRIPTION = """\
ICON (Indonesian CONstituency treebank) is a large-scale high-quality constituency treebank (10000 sentences)
for the Indonesian language, sourced from Wikipedia and news data from Tempo, spanning the period from 1971 to 2016.
The annotation guidelines were formulated with the Penn Treebank POS tagging and bracketing guidelines as a reference,
with additional adaptations to account for the characteristics of the Indonesian language.
"""

_HOMEPAGE = "https://github.com/aisingapore/seacorenlp-data/tree/main/id/constituency"

_LICENSE = Licenses.CC_BY_NC_SA_4_0.value
_LANGUAGES = ["ind"]
_LOCAL = False
_URLS = {
    "train": "https://raw.githubusercontent.com/aisingapore/seacorenlp-data/main/id/constituency/train.txt",
    "validation": "https://raw.githubusercontent.com/aisingapore/seacorenlp-data/main/id/constituency/dev.txt",
    "test": "https://raw.githubusercontent.com/aisingapore/seacorenlp-data/main/id/constituency/test.txt",
}

_SUPPORTED_TASKS = [Tasks.CONSTITUENCY_PARSING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class ICONDataset(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        SEACrowdConfig(name=f"{_DATASETNAME}_source", version=datasets.Version(_SOURCE_VERSION), description=_DESCRIPTION, schema="source", subset_id=f"{_DATASETNAME}"),
        SEACrowdConfig(name=f"{_DATASETNAME}_seacrowd_tree", version=datasets.Version(_SEACROWD_VERSION), description=_DESCRIPTION, schema="seacrowd_tree", subset_id=f"{_DATASETNAME}"),
    ]

    DEFAULT_CONFIG_NAME = "icon_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),  # index
                    "tree": datasets.Value("string"),  # nltk.tree
                    "sentence": datasets.Value("string"),  # bracketed sentence tree
                    "words": datasets.Sequence(datasets.Value("string")),  # words
                    "POS": datasets.Sequence(datasets.Value("string")),  # pos-tags
                }
            )
        elif self.config.schema == "seacrowd_tree":
            features = schemas.tree_features

        else:
            raise ValueError(f"Invalid config: {self.config.name}")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:

        train_txt = Path(dl_manager.download_and_extract(_URLS["train"]))
        dev_txt = Path(dl_manager.download_and_extract(_URLS["validation"]))
        test_txt = Path(dl_manager.download_and_extract(_URLS["test"]))

        data_dir = {
            "train": train_txt,
            "validation": dev_txt,
            "test": test_txt,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["validation"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        trees = nltk_load_trees(filepath)
        if self.config.schema == "source":
            for idx, tree in enumerate(trees):
                ex = {"index": str(idx), "tree": tree.tree, "words": tree.words, "sentence": tree.bra_sent, "POS": [itm[1] for itm in tree.pos()]}
                yield idx, ex
        if self.config.schema == "seacrowd_tree":
            for idx, tree in enumerate(trees):
                ex = get_node_char_indices_with_ids(tree.tree, str(idx))
                yield idx, ex


class BaseInputExample(ABC):
    """Parser input for a single sentence (abstract interface)."""

    words: List[str]
    space_after: List[bool]
    tree: Optional[nltk.Tree]

    @abstractmethod
    def leaves(self) -> Optional[List[str]]:
        """Returns leaves to use in the parse tree."""
        pass

    @abstractmethod
    def pos(self) -> Optional[List[Tuple[str, str]]]:
        """Returns a list of (leaf, part-of-speech tag) tuples."""
        pass


@dataclasses.dataclass
class ParsingExample(BaseInputExample):
    """A single parse tree and sentence."""

    words: List[str]
    bra_sent: str
    tree: Optional[nltk.Tree] = None
    _pos: Optional[List[Tuple[str, str]]] = None

    def leaves(self) -> Optional[List[str]]:
        return self.tree.leaves() if self.tree else None

    def pos(self) -> Optional[List[Tuple[str, str]]]:
        return self.tree.pos() if self.tree else self._pos

    def without_gold_annotations(self) -> "ParsingExample":
        return dataclasses.replace(self, tree=None, _pos=self.pos())


def nltk_load_trees(const_path: str) -> List[ParsingExample]:
    reader = BracketParseCorpusReader("", [const_path])
    trees = reader.parsed_sents()
    with open(const_path, "r") as filein:
        bracketed_sentences = [itm.strip() for itm in filein.readlines()]
    sents = [tree.leaves() for tree in trees]
    assert len(trees) == len(sents) == len(bracketed_sentences), f"Number Mismatched:  {len(trees)} vs {len(bracketed_sentences)}"
    treebank = [ParsingExample(tree=tree, words=words, bra_sent=bra_sent) for tree, bra_sent, words, in zip(trees, bracketed_sentences, sents)]
    for example in treebank:
        assert len(example.words) == len(example.leaves()), "Token count mismatch."
    return treebank


def get_node_char_indices_with_ids(tree, sent_id):
    def traverse_tree(subtree, start_index):
        nonlocal node_id
        current_id = node_id
        node_id += 1
        node_text = " ".join(subtree.leaves())
        end_index = start_index + len(node_text)

        # Record the current node
        node_data = {
            "id": f"{sent_id}_{current_id}",
            "type": subtree.label(),
            "text": node_text,
            "offsets": [start_index, end_index],
            "subnodes": [],
        }
        node_indices.append(node_data)

        for child in subtree:
            if isinstance(child, Tree):
                child_id = traverse_tree(child, start_index)
                node_data["subnodes"].append(child_id)
                start_index += len(" ".join(child.leaves())) + 1
        return f"{sent_id}_{current_id}"

    node_indices = []
    node_id = 0
    traverse_tree(tree, 0)
    sentence = " ".join(tree.leaves())
    passage = {"id": "p" + sent_id, "type": None, "text": tree.leaves(), "offsets": [0, len(sentence)]}
    return {"id": "s" + sent_id, "passage": passage, "nodes": node_indices}
