from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import datasets
from datasets.download.download_manager import DownloadManager

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = r"""
@misc{pilar2023cebuaner,
    title={CebuaNER: A New Baseline Cebuano Named Entity Recognition Model},
    author={Ma. Beatrice Emanuela Pilar and Ellyza Mari Papas and Mary Loise Buenaventura and Dane Dedoroy and Myron Darrel Montefalcon and Jay Rhald Padilla and Lany Maceda and Mideth Abisado and Joseph Marvin Imperial},
    year={2023},
    eprint={2310.00679},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_LOCAL = False
_LANGUAGES = ["ceb"]
_DATASETNAME = "cebuaner"
_DESCRIPTION = """\
The CebuaNER dataset contains 4000+ news articles that have been tagged by
native speakers of Cebuano usin gthe BIO encoding schema for the named entity
recognition (NER) task.
"""

_HOMEPAGE = "https://github.com/mebzmoren/CebuaNER"
_LICENSE = Licenses.CC_BY_NC_SA_4_0.value
_URLS = {
    "annotator_1": "https://github.com/mebzmoren/CebuaNER/raw/main/data/annotated_data/final-1.txt",
    "annotator_2": "https://github.com/mebzmoren/CebuaNER/raw/main/data/annotated_data/final-2.txt",
}

# The alignment between annotators is high, and both can be used as gold-standard data.
# Hence, we chose the first value on the index.
_DEFAULT_ANNOTATOR = "annotator_1"

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class CebuaNERDataset(datasets.GeneratorBasedBuilder):
    """CebuaNER dataset from https://github.com/mebzmoren/CebuaNER"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "seq_label"
    LABEL_CLASSES = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-OTHER",
        "I-OTHER",
    ]

    # There are two annotators in the CebuaNER dataset but there's no canonical
    # label. Here, we decided to create loaders for both annotators. The
    # inter-annotator reliability is high so it's possible to treat either as
    # gold-standard data.
    dataset_names = sorted([f"{_DATASETNAME}_{annot}" for annot in _URLS.keys()])
    BUILDER_CONFIGS = []
    for name in dataset_names:
        source_config = SEACrowdConfig(
            name=f"{name}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=name,
        )
        BUILDER_CONFIGS.append(source_config)
        seacrowd_config = SEACrowdConfig(
            name=f"{name}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=name,
        )
        BUILDER_CONFIGS.append(seacrowd_config)

    # Create a configuration that loads the annotations of the first annotator
    # and treat that as the default.
    BUILDER_CONFIGS.extend([
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=_DATASETNAME,
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=_DATASETNAME,
        ),
    ])

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.features.ClassLabel(names=self.LABEL_CLASSES)),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.seq_label_features(self.LABEL_CLASSES)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> List[datasets.SplitGenerator]:
        if self.config.subset_id == _DATASETNAME:
            url = _URLS[_DEFAULT_ANNOTATOR] 
        else:
            _, annotator = self.config.subset_id.split("_", 1)
            url = _URLS[annotator]
        data_file = Path(dl_manager.download_and_extract(url))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_file, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_file, "split": "dev"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_file, "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        label_key = "ner_tags" if self.config.schema == "source" else "labels"
        examples: Iterable[Dict[str, List[str]]] = []
        with open(filepath, encoding="utf-8") as f:
            tokens = []
            ner_tags = []
            for line in f:
                # There's no clear delimiter in the IOB file so I'm separating each example based on the newline.
                # The -DOCSTART- delimiter only shows up in the very first example. In their notebook example
                #  https://github.com/mebzmoren/CebuaNER/blob/main/notebooks/Named-Entity-Recognition-with-Conditional-Random-Fields.ipynb,
                # they used '' as their article delimiter.
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        examples.append({"tokens": tokens, label_key: ner_tags})
                        if len(tokens) != len(ner_tags):
                            raise ValueError(f"Tokens and tags are not aligned! {len(tokens)} != {len(ner_tags)}")
                        tokens = []
                        ner_tags = []
                else:
                    # CebuaNER iob are separated by spaces
                    token, _, _, ner_tag = line.split(" ")
                    tokens.append(token)
                    ner_tags.append(ner_tag.rstrip())
            if tokens:
                examples.append({"tokens": tokens, label_key: ner_tags})
                if len(tokens) != len(ner_tags):
                    raise ValueError(f"Tokens and tags are not aligned! {len(tokens)} != {len(ner_tags)}")

        # The CebuaNER paper doesn't provide a recommended split. However, the Github repository
        # contains a notebook example of the split they used in the report:
        # https://github.com/mebzmoren/CebuaNER/blob/main/notebooks/Named-Entity-Recognition-with-Conditional-Random-Fields.ipynb
        if split == "train":
            final_examples = examples[0:2980]
        if split == "test":
            final_examples = examples[2980:3831]
        if split == "dev":
            final_examples = examples[3831:]

        for idx, eg in enumerate(final_examples):
            eg["id"] = idx
            yield idx, eg
