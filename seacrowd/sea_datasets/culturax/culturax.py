from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urljoin

import datasets
from pyarrow import parquet as pq

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses

_CITATION = """\
@article{nguyen2023culturax,
    author    = {Thuat Nguyen and Chien Van Nguyen and Viet Dac Lai and Hieu Man and Nghia Trung Ngo and Franck Dernoncourt and Ryan A. Rossi and Thien Huu Nguyen},
    title     = {CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages},
    journal   = {arXiv preprint arXiv:2309.09400},
    year      = {2023},
    url       = {https://arxiv.org/abs/2309.09400},
}
"""

_DATASETNAME = "culturax"
_DESCRIPTION = """\
CulturaX is a comprehensive multilingual dataset comprising 6.3 trillion tokens across 167
languages, designed for large language model development. It incorporates an advanced
cleaning and deduplication process, including language identification and fuzzy
deduplication with MinHash, to ensure high-quality data for model training. The dataset,
which spans 16TB in parquet format and 27TB when unpacked, is a combination of the latest
mC4 and OSCAR corpora, emphasizing non-English languages to support multilingual model
training. For data cleaning validation, CulturaX employs a SentencePiece tokenizer and
KenLM language models, utilizing recent Wikipedia dumps for perplexity scoring.
Before using this dataloader, please accept the acknowledgement at https://huggingface.co/datasets/uonlp/CulturaX and use huggingface-cli login for authentication.
"""

_LOCAL=False
_LANGUAGES = ["ind", "jav", "khm", "lao", "tgl", "min", "mya", "sun", "tha", "vie", "zlm", "ceb", "war", "cbk", "bcl"]  # We follow ISO639-3 language code (https://iso639-3.sil.org/code_tables/639/data)

_HOMEPAGE = "https://huggingface.co/datasets/uonlp/CulturaX"
_LICENSE = f"""{Licenses.OTHERS.value} | \
    The licence terms for CulturaX strictly follows those of mC4 and OSCAR. \
    Please refer to both below licenses when using this dataset. \
    - mC4 license: https://huggingface.co/datasets/allenai/c4#license \
    - OSCAR license: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301#licensing-information \
"""
_BASE_URL = "https://huggingface.co/datasets/uonlp/CulturaX/resolve/main/{lang}/"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class CulturaXDataset(datasets.GeneratorBasedBuilder):
    """CulturaX subset for SEA languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = "ssp"
    SUBSETS = ["id", "jv", "km", "lo", "tl", "min", "my", "su", "th", "vi", "ms", "ceb", "war", "cbk", "bcl"]

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{subset}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} {subset} source schema",
            schema="source",
            subset_id=subset,
        )
        for subset in SUBSETS
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
                    "text": datasets.Value("string"),
                    "timestamp": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "source": datasets.Value("string"),
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
        base_path = _BASE_URL.format(lang=self.config.name.split("_")[1])

        checksum_url = urljoin(base_path, "checksum.sha256")
        checksum_path = Path(dl_manager.download(checksum_url))
        with open(checksum_path, encoding="utf-8") as f:
            filenames = [line.split()[1] for line in f if line]
            data_urls = [urljoin(base_path, filename) for filename in filenames]

        data_paths = list(map(Path, dl_manager.download([url for url in data_urls if url.endswith(".parquet")])))

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
        """Yields examples as (key, example) tuples.

        Iterate over row groups in each filepaths, then yield each row as an example.
        """
        key = 0
        for filepath in filepaths:
            with open(filepath, "rb") as f:
                pf = pq.ParquetFile(f)
                for row_group in range(pf.num_row_groups):
                    df = pf.read_row_group(row_group).to_pandas()
                    for row in df.itertuples():
                        if self.config.schema == "source":
                            yield key, {
                                "text": row.text,
                                "timestamp": row.timestamp,
                                "url": row.url,
                                "source": row.source,
                            }
                        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
                            yield key, {
                                "id": str(key),
                                "text": row.text,
                            }
                        key += 1
