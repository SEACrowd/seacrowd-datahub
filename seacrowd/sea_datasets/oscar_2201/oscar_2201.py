import re
import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urljoin

import datasets

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks

_CITATION = """\
@inproceedings{abadji2022cleaner,
    author    = {Julien Abadji and
                Pedro Javier Ortiz Su{\'{a}}rez and
                Laurent Romary and
                Beno{\^{\i}}t Sagot},
    title     = {Towards a Cleaner Document-Oriented Multilingual Crawled Corpus},
    booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference,
                {LREC} 2022, Marseille, France, 20-25 June 2022},
    pages     = {4344--4355},
    publisher = {European Language Resources Association},
    year      = {2022},
    url       = {https://aclanthology.org/2022.lrec-1.463},
}
"""

_DATASETNAME = "oscar_2201"
_DESCRIPTION = """\
OSCAR or Open Super-large Crawled Aggregated coRpus is a huge multilingual corpus
obtained by language classification and filtering of the Common Crawl corpus using
the ungoliant architecture. Data is distributed by language in both original and
deduplicated form.
"""

_HOMEPAGE = "https://huggingface.co/datasets/oscar-corpus/OSCAR-2201"
_LICENSE = """\
These data are released under this licensing scheme \
We do not own any of the text from which these data has been extracted. \
We license the actual packaging of these data under the Creative Commons CC0 license ("no rights reserved") http://creativecommons.org/publicdomain/zero/1.0/ \
To the extent possible under law, Inria has waived all copyright and related or neighboring rights to OSCAR \
This work is published from: France. \
\
Should you consider that our data contains material that is owned by you and should therefore not be reproduced here, please: \
* Clearly identify yourself, with detailed contact data such as an address, telephone number or email address at which you can be contacted. \
* Clearly identify the copyrighted work claimed to be infringed. \
* Clearly identify the material that is claimed to be infringing and information reasonably sufficient to allow us to locate the material. \
\
We will comply to legitimate requests by removing the affected sources from the next release of the corpus.
"""
_BASE_URL = "https://huggingface.co/datasets/oscar-corpus/OSCAR-2201/resolve/main/compressed/{lang}_meta/"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "2022.1.0"
_SEACROWD_VERSION = "1.0.0"


class Oscar2201Dataset(datasets.GeneratorBasedBuilder):
    """OSCAR subset for SEA languages, version 2201."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "ssp"
    SUBSETS = ["war", "ceb", "min", "vi", "ta", "ilo", "tl", "lo", "km", "my", "jv", "id", "th", "su", "ms"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} {subset} source schema",
            schema="source",
            subset_id=subset,
        ) for subset in SUBSETS
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_seacrowd_ssp",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} {subset} SEACrowd schema",
            schema="seacrowd_ssp",
            subset_id=subset,
        )
        for subset in SUBSETS
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_jv_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "text": datasets.Value("string"),
                    "meta": {
                        "warc_headers": {
                            "warc-record-id": datasets.Value("string"),
                            "warc-date": datasets.Value("string"),
                            "content-type": datasets.Value("string"),
                            "content-length": datasets.Value("int32"),
                            "warc-type": datasets.Value("string"),
                            "warc-identified-content-language": datasets.Value("string"),
                            "warc-refers-to": datasets.Value("string"),
                            "warc-target-uri": datasets.Value("string"),
                            "warc-block-digest": datasets.Value("string"),
                        },
                        "identification": {
                            "label": datasets.Value("string"),
                            "prob": datasets.Value("float"),
                        },
                        "annotations": datasets.Sequence(datasets.Value("string")),
                        "line_identifications": [
                            {
                                "label": datasets.Value("string"),
                                "prob": datasets.Value("float"),
                            }
                        ],
                    },
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = schemas.ssp_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        base_path = _BASE_URL.format(lang=self.config.name.split("_")[2])

        checksum_url = urljoin(base_path, "checksum.sha256")
        checksum_path = Path(dl_manager.download(checksum_url))
        with open(checksum_path, encoding="utf-8") as f:
            filenames = [line.split()[1] for line in f if line]
            filenames = sorted(filenames, key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else x)
            data_urls = [urljoin(base_path, filename) for filename in filenames]

        data_paths = list(map(Path, dl_manager.download([url for url in data_urls if url.endswith(".jsonl.gz")])))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": data_paths,
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepaths: [Path], split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        key = 0
        for filepath in filepaths:
            with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    if self.config.schema == "source":
                        meta = dict()
                        meta["warc_headers"] = doc["warc_headers"]
                        meta["warc_headers"]["warc-identified-content-language"] = doc["warc_headers"].get("warc-identified-content-language")
                        meta["identification"] = doc["metadata"]["identification"]
                        meta["annotations"] = doc["metadata"]["annotation"]
                        meta["line_identifications"] = doc["metadata"]["sentence_identifications"]
                        yield key, {"id": key, "text": doc["content"], "meta": meta}
                    elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                        yield key, {"id": str(key), "text": doc["content"]}
                    key += 1