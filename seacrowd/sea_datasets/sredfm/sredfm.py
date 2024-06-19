# Some code referenced from https://huggingface.co/datasets/Babelscape/SREDFM/blob/main/SREDFM.py

from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import jsonlines

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """\
@inproceedings{huguet-cabot-et-al-2023-redfm-dataset,
    title = "RED$^{\rm FM}$: a Filtered and Multilingual Relation Extraction Dataset",
    author = "Huguet Cabot, Pere-Lluís  and Tedeschi, Simone and Ngonga Ngomo, Axel-Cyrille and
      Navigli, Roberto",
    booktitle = "Proc. of the 61st Annual Meeting of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2306.09802",
}
"""

_DATASETNAME = "sredfm"


_DESCRIPTION = """\
SREDFM is an automatically annotated dataset for relation extraction task covering 18 languages, 400 relation types, 13 entity types, totaling more than 40 million triplet instances. SREDFM includes Vietnamnese.
"""

_HOMEPAGE = "https://github.com/babelscape/rebel"

_LANGUAGES = ["vie"]

_LICENSE = Licenses.CC_BY_SA_4_0.value

_LOCAL = False

_URLS = {
    "train": "https://huggingface.co/datasets/Babelscape/SREDFM/resolve/main/data/train.vi.jsonl",
    "dev": "https://huggingface.co/datasets/Babelscape/SREDFM/resolve/main/data/dev.vi.jsonl",
    "test": "https://huggingface.co/datasets/Babelscape/SREDFM/resolve/main/data/test.vi.jsonl",
    "relations_url": "https://huggingface.co/datasets/Babelscape/SREDFM/raw/main/relations.tsv",
}

_SUPPORTED_TASKS = [Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class SREDFMDataset(datasets.GeneratorBasedBuilder):
    """SREDFM is an automatically annotated dataset for relation extraction task.
    Relation Extraction (RE) is a task that identifies relationships between entities in a text,
    enabling the acquisition of relational facts and bridging the gap between natural language
    and structured knowledge. SREDFM covers 400 relation types, 13 entity types,
    totaling more than 40 million triplet instances."""

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
                    "docid": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "uri": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "uri": datasets.Value(dtype="string"),
                            "surfaceform": datasets.Value(dtype="string"),
                            "type": datasets.Value(dtype="string"),
                            "start": datasets.Value(dtype="int32"),
                            "end": datasets.Value(dtype="int32"),
                        }
                    ],
                    "relations": [
                        {
                            "subject": datasets.Value(dtype="int32"),
                            "predicate": datasets.Value(dtype="string"),
                            "object": datasets.Value(dtype="int32"),
                        }
                    ],
                }
            )

        elif self.config.schema == "seacrowd_kb":
            features = schemas.kb_features

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

        relation_names = dict()
        relation_path = data_dir["relations_url"]
        with open(relation_path, encoding="utf-8") as f:
            for row in f:
                rel_code, rel_name, _, _ = row.strip().split("\t")
                relation_names[rel_code] = rel_name

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_dir["train"], "relation_names": relation_names},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir["test"], "relation_names": relation_names},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_dir["dev"], "relation_names": relation_names},
            ),
        ]

    def _generate_examples(self, filepath: Path, relation_names: dict) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with jsonlines.open(filepath) as f:
                skip = set()
                for example in f.iter():
                    if example["docid"] in skip:
                        continue
                    skip.add(example["docid"])

                    entities = []
                    for entity in example["entities"]:
                        entities.append(
                            {
                                "uri": entity["uri"],
                                "surfaceform": entity["surfaceform"],
                                "start": entity["boundaries"][0],
                                "end": entity["boundaries"][1],
                                "type": entity["type"],
                            }
                        )

                    relations = []
                    for relation in example["relations"]:
                        if relation["predicate"]["uri"] not in relation_names or relation["confidence"] <= 0.75:
                            continue

                        relations.append(
                            {
                                "subject": entities.index(
                                    {
                                        "uri": relation["subject"]["uri"],
                                        "surfaceform": relation["subject"]["surfaceform"],
                                        "start": relation["subject"]["boundaries"][0],
                                        "end": relation["subject"]["boundaries"][1],
                                        "type": relation["subject"]["type"],
                                    }
                                ),
                                "predicate": relation_names[relation["predicate"]["uri"]],
                                "object": entities.index(
                                    {
                                        "uri": relation["object"]["uri"],
                                        "surfaceform": relation["object"]["surfaceform"],
                                        "start": relation["object"]["boundaries"][0],
                                        "end": relation["object"]["boundaries"][1],
                                        "type": relation["object"]["type"],
                                    }
                                ),
                            }
                        )

                    if len(relations) == 0:
                        continue

                    yield example["docid"], {
                        "docid": example["docid"],
                        "title": example["title"],
                        "uri": example["uri"],
                        "text": example["text"],
                        "entities": entities,
                        "relations": relations,
                    }

        elif self.config.schema == "seacrowd_kb":
            with jsonlines.open(filepath) as f:
                skip = set()
                i = 0
                for example in f.iter():
                    if example["docid"] in skip:
                        continue
                    skip.add(example["docid"])

                    i += 1
                    processed_text = example["text"].replace("\n", " ")
                    passages = [
                        {
                            "id": f"{i}-{example['uri']}",
                            "type": "text",
                            "text": [processed_text],
                            "offsets": [[0, len(processed_text)]],
                        }
                    ]

                    entities = []
                    for entity in example["entities"]:
                        entities.append(
                            {
                                "id": entity["uri"],
                                "type": entity["type"],
                                "text": [entity["surfaceform"]],
                                "offsets": [entity["boundaries"]],
                                "normalized": {"db_name": "", "db_id": ""},
                            }
                        )

                    relations = []
                    for relation in example["relations"]:
                        if relation["predicate"]["uri"] not in relation_names or relation["confidence"] <= 0.75:
                            continue

                        i += 1
                        sub = relation["subject"]
                        pred = relation["predicate"]
                        obj = relation["object"]
                        relations.append(
                            {
                                "id": f"{i}-{sub['uri']}-{pred['uri']}-{obj['uri']}",
                                "type": relation_names[pred["uri"]],
                                "arg1_id": str(
                                    entities.index(
                                        {
                                            "id": sub["uri"],
                                            "type": sub["type"],
                                            "text": [sub["surfaceform"]],
                                            "offsets": [sub["boundaries"]],
                                            "normalized": {"db_name": "", "db_id": ""},
                                        }
                                    )
                                ),
                                "arg2_id": str(
                                    entities.index(
                                        {
                                            "id": obj["uri"],
                                            "type": obj["type"],
                                            "text": [obj["surfaceform"]],
                                            "offsets": [obj["boundaries"]],
                                            "normalized": {"db_name": "", "db_id": ""},
                                        }
                                    )
                                ),
                                "normalized": {"db_name": "", "db_id": ""},
                            }
                        )

                    for entity in entities:
                        i += 1
                        entity["id"] = f"{i}-{entity['id']}"

                    if len(relations) == 0:
                        continue

                    yield example["docid"], {
                        "id": example["docid"],
                        "passages": passages,
                        "entities": entities,
                        "relations": relations,
                        "events": [],
                        "coreferences": [],
                    }
