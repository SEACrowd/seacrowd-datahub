from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils import schemas
from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Licenses, Tasks

_CITATION = """
@misc{singh2024aya,
      title={Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning},
      author={Shivalika Singh and Freddie Vargus and Daniel Dsouza and Börje F. Karlsson and
      Abinaya Mahendiran and Wei-Yin Ko and Herumb Shandilya and Jay Patel and Deividas
      Mataciunas and Laura OMahony and Mike Zhang and Ramith Hettiarachchi and Joseph
      Wilson and Marina Machado and Luisa Souza Moura and Dominik Krzemiński and Hakimeh
      Fadaei and Irem Ergün and Ifeoma Okoh and Aisha Alaagib and Oshan Mudannayake and
      Zaid Alyafeai and Vu Minh Chien and Sebastian Ruder and Surya Guthikonda and Emad A.
      Alghamdi and Sebastian Gehrmann and Niklas Muennighoff and Max Bartolo and Julia Kreutzer
      and Ahmet Üstün and Marzieh Fadaee and Sara Hooker},
      year={2024},
      eprint={2402.06619},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "aya_collection_translated"

_DESCRIPTION = """
The Aya Collection is a massive multilingual collection consisting of 513 million instances of prompts and
completions covering a wide range of tasks. This dataset covers the translated prompts of the Aya Collection.
"""

_HOMEPAGE = "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split"

_LANGUAGES = ["ceb", "tha", "mya", "zsm", "jav", "ind", "vie", "sun", "ace", "bjn", "khm", "lao", "min"]

_LICENSE = Licenses.APACHE_2_0.value

_LOCAL = False

_URLS = {
    "ceb": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/cebuano",
    "tha": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/thai",
    "mya": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/burmese",
    "zsm": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/malayalam",
    "jav": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/javanese",
    "ind": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/indonesian",
    "vie": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/vietnamese",
    "sun": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/sundanese",
    "ace": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/achinese",
    "bjn": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/banjar",
    "khm": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/central_khmer",
    "lao": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/lao",
    "min": "https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/resolve/main/minangkabau",
}

_SUPPORTED_TASKS = [Tasks.INSTRUCTION_TUNING]

_SOURCE_VERSION = "1.0.0"

_SEACROWD_VERSION = "2024.06.20"


class AyaCollectionTranslatedDataset(datasets.GeneratorBasedBuilder):
    """
    The Aya Collection is a massive multilingual collection consisting of 513 million instances of prompts and
    completions covering a wide range of tasks. This dataset covers the translated prompts of the Aya Collection.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{LANG}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description=f"{_DATASETNAME} {LANG} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}_{LANG}",
        )
        for LANG in _LANGUAGES
    ] + [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_{LANG}_seacrowd_t2t",
            version=datasets.Version(_SEACROWD_VERSION),
            description=f"{_DATASETNAME} {LANG} SEACrowd schema",
            schema="seacrowd_t2t",
            subset_id=f"{_DATASETNAME}_{LANG}",
        )
        for LANG in _LANGUAGES
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_ind_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "inputs": datasets.Value("string"),
                    "targets": datasets.Value("string"),
                    "dataset_name": datasets.Value("string"),
                    "sub_dataset_name": datasets.Value("string"),
                    "task_type": datasets.Value("string"),
                    "template_id": datasets.Value("int64"),
                    "language": datasets.Value("string"),
                    "script": datasets.Value("string"),
                    "split": datasets.Value("string"),
                }
            )

        elif self.config.schema == "seacrowd_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        language = self.config.name.split("_")[3]

        if language in _LANGUAGES:
            data_train_paths = []
            for version in [0, 1, 2]:
                for all in [1, 2, 3]:
                    if version >= all:
                        continue
                    else:
                        try:
                            data_train_path = Path(dl_manager.download_and_extract(f"{_URLS[language]}/train-0000{version}-of-0000{all}.parquet?download=true"))
                            data_train_paths.append(data_train_path)
                        except Exception:
                            continue

            data_validation_path = Path(dl_manager.download_and_extract(f"{_URLS[language]}/validation-00000-of-00001.parquet?download=true"))
            data_test_path = Path(dl_manager.download_and_extract(f"{_URLS[language]}/test-00000-of-00001.parquet?download=true"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_train_paths,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_test_path,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_validation_path,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if isinstance(filepath, Path):
            dfs = [pd.read_parquet(filepath)]
        else:
            dfs = [pd.read_parquet(path) for path in filepath]

        df = pd.concat(dfs, ignore_index=True)

        for index, row in df.iterrows():
            if self.config.schema == "source":
                example = row.to_dict()

            elif self.config.schema == "seacrowd_t2t":
                example = {
                    "id": str(index),
                    "text_1": row["inputs"],
                    "text_2": row["targets"],
                    "text_1_name": "inputs",
                    "text_2_name": "targets",
                }

            yield index, example
