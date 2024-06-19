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
MIRACL is a multilingual dataset for ad hoc retrieval across 18 languages that 
collectively encompass over three billion native speakers around the world. 
This resource is designed to support monolingual retrieval tasks, where the 
queries and the corpora are in the same language. In total, we have gathered 
over 726k high-quality relevance judgments for 78k queries over Wikipedia in 
these languages, where all annotations have been performed by native speakers. 
MIRACL covers Indonesian and Thai languages
"""

from typing import Dict, List, Tuple

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

from collections import defaultdict

_CITATION = """\
    @article{10.1162/tacl_a_00595,
    title        = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
    author       = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
    year         = 2023,
    month        = {09},
    journal      = {Transactions of the Association for Computational Linguistics},
    volume       = 11,
    pages        = {1114--1131},
    doi          = {10.1162/tacl\_a\_00595},
    issn         = {2307-387X},
    url          = {https://doi.org/10.1162/tacl\%5Fa\%5F00595},
    abstract     = {{MIRACL is a multilingual dataset for ad hoc retrieval across 18 languages that collectively encompass over three billion native speakers around the world. This resource is designed to support monolingual retrieval tasks, where the queries and the corpora are in the same language. In total, we have gathered over 726k high-quality relevance judgments for 78k queries over Wikipedia in these languages, where all annotations have been performed by native speakers hired by our team. MIRACL covers languages that are both typologically close as well as distant from 10 language families and 13 sub-families, associated with varying amounts of publicly available resources. Extensive automatic heuristic verification and manual assessments were performed during the annotation process to control data quality. In total, MIRACL represents an investment of around five person-years of human annotator effort. Our goal is to spur research on improving retrieval across a continuum of languages, thus enhancing information access capabilities for diverse populations around the world, particularly those that have traditionally been underserved. MIRACL is available at http://miracl.ai/.}},
    eprint       = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf}
}
"""


_DATASETNAME = "miracl"

_DESCRIPTION = """\
MIRACL is a multilingual dataset for ad hoc retrieval across 18 languages that collectively encompass over three billion native speakers around the world. This resource is designed to support monolingual retrieval tasks, where the queries and the corpora are in the same language. In total, we have gathered over 726k high-quality relevance judgments for 78k queries over Wikipedia in these languages, where all annotations have been performed by native speakers. MIRACL covers Indonesian and Thai languages. Before using this dataloader, please accept the acknowledgement at https://huggingface.co/datasets/miracl/miracl and use huggingface-cli login for authentication.
"""

_HOMEPAGE = "https://project-miracl.github.io/"

_LANGUAGES = ["ind", "tha"]

_LICENSE = Licenses.APACHE_2_0.value

_LANGUAGE_MAP = {
    "id": "Thai", 
    "th": "Indonesian"
}

_URLS = {_DATASETNAME: {lang: {} for lang in _LANGUAGE_MAP}}

for lang in _LANGUAGE_MAP:
    _URLS[_DATASETNAME][lang]['train'] = [
        f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-train.tsv',
        f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-train.tsv',
    ]

    _URLS[_DATASETNAME][lang]['dev'] = [
        f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv',
        f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv',
    ]
    
    _URLS[_DATASETNAME][lang]['testB'] =[
        f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-test-b.tsv',
    ]

    _URLS[_DATASETNAME][lang]['testA'] = [
        f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-test-a.tsv',
    ]


_SUPPORTED_TASKS = [Tasks.TEXT_RETRIEVAL]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"

_LOCAL = False


def load_topic(fn):

    qid2topic = {}
    with open(fn, encoding="utf-8") as f:
        for line in f:
            qid, topic = line.strip().split('\t')
            qid2topic[qid] = topic
    return qid2topic


def load_qrels(fn):
    if fn is None:
        return None

    qrels = defaultdict(dict)
    with open(fn, encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split('\t')
            qrels[qid][docid] = int(rel)
    return qrels

def seacrowd_config_constructor(lang, schema, version):
    if lang not in _LANGUAGE_MAP:
        raise ValueError(f"Invalid lang {lang}")

    if schema != "source" and schema != "seacrowd_pairs":
        raise ValueError(f"Invalid schema: {schema}")

    return SEACrowdConfig(
        name="miracl_{lang}_{schema}".format(lang=lang, schema=schema),
        version=datasets.Version(version),
        description="MIRACL {schema} schema for {lang} language".format(lang=_LANGUAGE_MAP[lang], schema=schema),
        schema=schema,
        subset_id="miracl_{lang}".format(lang=lang),
    )

class Miracl(datasets.GeneratorBasedBuilder):
    """MIRACL is a multilingual retrieval dataset that focuses on search across 18 different languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [seacrowd_config_constructor(lang, "source", _SOURCE_VERSION) for lang in _LANGUAGE_MAP] + [seacrowd_config_constructor(lang, "seacrowd_pairs", _SEACROWD_VERSION) for lang in _LANGUAGE_MAP]

    DEFAULT_CONFIG_NAME = None

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({
                'query_id': datasets.Value('string'),
                'query': datasets.Value('string'),

                'positive_passages': [{
                    'docid': datasets.Value('string'),
                    'text': datasets.Value('string'), 'title': datasets.Value('string')
                }],
                'negative_passages': [{
                    'docid': datasets.Value('string'),
                    'text': datasets.Value('string'), 'title': datasets.Value('string'),
                }],
            })
        elif self.config.schema == "seacrowd_pairs":
            features = schemas.pairs_features(["pos", "neg", "none"])
            
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
        lang = self.config.name.split("_")[1]
        downloaded_files = dl_manager.download_and_extract(urls[lang])

        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "filepaths": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name="dev",
                gen_kwargs={
                    "filepaths": downloaded_files["dev"],
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name="testA",
                gen_kwargs={
                    "filepaths": downloaded_files["testA"],
                    "split": "testA",
                },
            ),
            datasets.SplitGenerator(
                name="testB",
                gen_kwargs={
                    "filepaths": downloaded_files["testB"],
                    "split": "testB",
                },
            )
        ]


    def _generate_examples(self, filepaths: List[str], split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        lang = self.config.name.split("_")[1]

        # the following code except for seacrowd_pairs is taken from the original MIRACL 
        # dataloader implementation
        # https://huggingface.co/datasets/miracl/miracl
        miracl_corpus = datasets.load_dataset('miracl/miracl-corpus', lang)['train']
        docid2doc = {doc['docid']: (doc['title'], doc['text']) for doc in miracl_corpus}
        topic_fn, qrel_fn = (filepaths) if len(filepaths) == 2 else (filepaths[0], None)
        qid2topic = load_topic(topic_fn)
        qrels = load_qrels(qrel_fn)

        if self.config.schema == "source":
            for qid in qid2topic:
                data = {}
                data['query_id'] = qid
                data['query'] = qid2topic[qid]
                
                pos_docids = [docid for docid, rel in qrels[qid].items() if rel == 1] if qrels is not None else []
                neg_docids = [docid for docid, rel in qrels[qid].items() if rel == 0] if qrels is not None else []

                data['positive_passages'] = [{
                    'docid': docid, 
                    **dict(zip(['title', 'text'], docid2doc[docid]))
                } for docid in pos_docids if docid in docid2doc]

                data['negative_passages'] = [{
                    'docid': docid, 
                    **dict(zip(['title', 'text'], docid2doc[docid]))
                } for docid in neg_docids if docid in docid2doc]
                
                yield qid, data

        elif self.config.schema == "seacrowd_pairs":
            id = -1
            for qid in qid2topic:
                pos_docids = [docid for docid, rel in qrels[qid].items() if rel == 1] if qrels is not None else []
                neg_docids = [docid for docid, rel in qrels[qid].items() if rel == 0] if qrels is not None else []

                positive_passages = [{
                    'docid': docid, 
                    **dict(zip(['title', 'text'], docid2doc[docid]))
                } for docid in pos_docids if docid in docid2doc]

                negative_passages = [{
                    'docid': docid, 
                    **dict(zip(['title', 'text'], docid2doc[docid]))
                } for docid in neg_docids if docid in docid2doc]

                # assemble data
                data = {}
                data['text_1'] = qid2topic[qid] # query

                if split in ["testA", "testB"]: # test sets only contains id and query
                    id += 1
                    data['id'] = id
                    data['text_2'] = ""
                    data['label'] = "none"

                    yield id, data
                else:
                    # generate positive pairs
                    for positive_doc in positive_passages:
                        id += 1
                        data['id'] = id
                        # flatten dict contents to String by concatenating title and text separated by double newline
                        data['text_2'] = positive_doc['title'] + "\n\n" + positive_doc["text"]
                        data['label'] = "pos"
                        yield id, data
                    
                    # generate negative pairs
                    for negative_doc in negative_passages:
                        id += 1
                        data['id'] = id
                        # flatten dict contents to String by concatenating title and text separated by double newline
                        data['text_2'] = negative_doc['title'] + "\n\n" + negative_doc["text"]
                        data['label'] = "neg"
                        yield id, data
