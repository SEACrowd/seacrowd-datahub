import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
import nltk
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

# from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import (DEFAULT_SEACROWD_VIEW_NAME,
                                      DEFAULT_SOURCE_VIEW_NAME, Tasks)

_DATASETNAME = "icon"
_SOURCE_VIEW_NAME = DEFAULT_SOURCE_VIEW_NAME
_UNIFIED_VIEW_NAME = DEFAULT_SEACROWD_VIEW_NAME

_LANGUAGES = ["ind"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)
_LOCAL = False
_CITATION = """\
@inproceedings{suan-lim-etal-2023-icon,
    title = "{ICON}: Building a Large-Scale Benchmark Constituency Treebank for the {I}ndonesian Language",
    author = "Suan Lim, Ee  and
      Qi Leong, Wei  and
      Thanh Nguyen, Ngan  and
      Adhista, Dea  and
      Ming Kng, Wei  and
      Chandra Tjh, William  and
      Purwarianti, Ayu",
    editor = {Dakota, Daniel  and
      Evang, Kilian  and
      K{\"u}bler, Sandra  and
      Levin, Lori},
    booktitle = "Proceedings of the 21st International Workshop on Treebanks and Linguistic Theories (TLT, GURT/SyntaxFest 2023)",
    month = mar,
    year = "2023",
    address = "Washington, D.C.",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.tlt-1.5",
    pages = "37--53",
    abstract = "Constituency parsing is an important task of informing how words are combined to form sentences.
    While constituency parsing in English has seen significant progress in the last few years, tools for constituency
    parsing in Indonesian remain few and far between. In this work, we publish ICON (Indonesian CONstituency treebank),
    the hitherto largest publicly-available manually-annotated benchmark constituency treebank for the Indonesian
    language with a size of 10,000 sentences and approximately 124,000 constituents and 182,000 tokens, which can
    support the training of state-of-the-art transformer-based models. We establish strong baselines on the ICON
    dataset using the Berkeley Neural Parser with transformer-based pre-trained embeddings, with the best performance
    of 88.85{%} F1 score coming from our own version of SpanBERT (IndoSpanBERT). We further analyze the predictions
    made by our best-performing model to reveal certain idiosyncrasies in the Indonesian language that pose challenges
    for constituency parsing.",
」
"""

_DESCRIPTION = """\
ICON (Indonesian CONstituency treebank) is a large-scale high-quality constituency treebank (10000 sentences)
for the Indonesian language, sourced from Wikipedia and news data from Tempo, spanning the period from 1971 to 2016.
The annotation guidelines were formulated with the Penn Treebank POS tagging and bracketing guidelines as a reference,
with additional adaptations to account for the characteristics of the Indonesian language.
"""

_HOMEPAGE = "https://github.com/aisingapore/seacorenlp-data/tree/main/id/constituency"

_LICENSE = "Creative Commons Attribution Share Alike 4.0 (cc-by-sa-4.0)"

_URLS = {
    "train": "https://raw.githubusercontent.com/aisingapore/seacorenlp-data/main/id/constituency/train.txt",
    "validation": "https://raw.githubusercontent.com/aisingapore/seacorenlp-data/main/id/constituency/dev.txt",
    "test": "https://raw.githubusercontent.com/aisingapore/seacorenlp-data/main/id/constituency/test.txt",
}

_SUPPORTED_TASKS = [Tasks.POS_TAGGING, Tasks.CONSTITUENCY_PARSING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "0.1.1"


class ICON(datasets.GeneratorBasedBuilder):
    """ICON (Indonesian CONstituency treebank) is a large-scale high-quality constituency treebank (10000 sentences)
    for the Indonesian language, sourced from Wikipedia and news data from Tempo, spanning the period from 1971 to 2016."""

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name="icon_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="ICON Constituency Tree for Indonesian Language in nltk.tree format",
            schema="source",
        ),
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
        if self.config.schema == "source":
            trees = nltk_load_trees(filepath)
            for i, tree in enumerate(trees):
                # id, tree
                ex = {"index": str(i), "tree": tree.tree, "words": tree.words, "sentence": tree.bra_sent, "POS": [itm[1] for itm in tree.pos()]}
                yield i, ex


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
