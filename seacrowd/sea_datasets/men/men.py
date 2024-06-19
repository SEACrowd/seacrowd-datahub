import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks
from seacrowd.utils.schemas import kb_features

_CITATION = """\
@misc{chanthran2024malaysian,
      title={Malaysian English News Decoded: A Linguistic Resource for Named Entity and Relation Extraction},
      author={Mohan Raj Chanthran and Lay-Ki Soon and Huey Fang Ong and Bhawani Selvaretnam},
      year={2024},
      eprint={2402.14521},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "men"

_DESCRIPTION = """\
The Malaysian English News (MEN) dataset includes 200 Malaysian English news article with human annotated entities and relations (in total 6,061 entities and 3,268 relation instances).
Malaysian English combines elements of standard English with Malay, Chinese, and Indian languages. Four human annotators were split into 2 groups, each group annotated 100 news articles
and inter-annotator agreement was calculated between 2 or more annotators working on the same task (entity annotation; F1-score 0.82, relation annotation; F1-score 0.51).
"""

_HOMEPAGE = "https://github.com/mohanraj-nlp/MEN-Dataset/tree/main"

_LANGUAGES = ["eng"]

_LICENSE = Licenses.MIT.value

_LOCAL = False

_URLS = "https://github.com/mohanraj-nlp/MEN-Dataset/archive/refs/heads/main.zip"

_SUPPORTED_TASKS = [Tasks.RELATION_EXTRACTION, Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class MENDataset(datasets.GeneratorBasedBuilder):
    """The Malaysian English News dataset comprises 200 articles with 6,061 annotated entities and 3,268 relations.
    Inter-annotator agreement for entity annotation was high (F1-score 0.82), but lower for relation annotation (F1-score 0.51)."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_kb",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema="seacrowd_kb",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "article": datasets.Value("string"),
                    "entities": datasets.Sequence({"id": datasets.Value("int64"), "label": datasets.Value("string"), "position": {"start": datasets.Value("int32"), "end": datasets.Value("int32")}}),
                    "relations": datasets.Sequence({"id": datasets.Value("string"), "head": datasets.Value("int32"), "tail": datasets.Value("int32"), "relation": datasets.Value("string"), "relation_source": datasets.Value("string")}),
                }
            )

        elif self.config.schema == "seacrowd_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                },
            ),
        ]

    def _MEN_repo_splitter(self, filepath: Path) -> Dict:
        articles = {}
        entities = os.path.join(filepath, "MEN-Dataset-main/data/annotated_set.json")
        relations = os.path.join(filepath, "MEN-Dataset-main/data/rel2id.json")

        with open(entities, "r") as annot_json:
            annots = json.load(annot_json)

        article_ids = [i["id"] for i in annots]
        for article_id in article_ids:
            articles[article_id] = os.path.join(filepath, f"MEN-Dataset-main/data/article_text/{article_id}.txt")

        data_dir = {"entities": entities, "articles": articles, "relations": relations}

        return data_dir

    def _generate_examples(self, filepath: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        filepath = self._MEN_repo_splitter(filepath)

        with open(filepath["entities"], "r") as entities_json:
            entities = json.load(entities_json)

        articles = {}
        for article_id in [i["id"] for i in entities]:
            with open(filepath["articles"][article_id], "r") as article_txt:
                article = article_txt.read()
            articles[article_id] = article

        i = 0
        for item in entities:
            article_id = item["id"]
            entities = item["entities"]
            relations = item["relations"]

            i += 1
            if self.config.schema == "source":
                yield i, {
                    "article": articles[article_id],
                    "entities": [
                        {
                            "id": entity["id"],
                            "label": entity["label"],
                            "position": {
                                "start": entity["position"]["start_offset"],
                                "end": entity["position"]["end_offset"],
                            },
                        }
                        for entity in entities
                    ],
                    "relations": [{"id": relation["id"], "head": relation["head"], "tail": relation["tail"], "relation": relation["relation"], "relation_source": relation["relation_source"]} for relation in relations],
                }

            elif self.config.schema == "seacrowd_kb":
                yield i, {
                    "id": str(i),
                    "passages": [{"id": article_id, "type": "text", "text": [articles[article_id]], "offsets": [[0, len(articles[article_id])]]}],
                    "entities": [
                        {
                            "id": f"{article_id}-entity-{entity['id']}",
                            "type": entity["label"],
                            "text": [articles[article_id][entity["position"]["start_offset"]:entity["position"]["end_offset"]]],
                            "offsets": [[entity["position"]["start_offset"], entity["position"]["end_offset"]]],
                            "normalized": [],
                        }
                        for entity in entities
                    ],
                    "events": [],
                    "coreferences": [],
                    "relations": [
                        {
                            "id": f"{article_id}-relation-{relation['id']}",
                            "type": relation["relation"],
                            "arg1_id": f"{article_id}-entity-{relation['head']}",
                            "arg2_id": f"{article_id}-entity-{relation['tail']}",
                            "normalized": [{"db_name": relation["relation_source"], "db_id": ""}],
                        }
                        for relation in relations
                    ],
                }
