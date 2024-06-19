from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from seacrowd.utils.configs import SEACrowdConfig
from seacrowd.utils.constants import Tasks, Licenses, TASK_TO_SCHEMA, SCHEMA_TO_FEATURES

_CITATION = """\
    @inproceedings{kargaran2023glotlid,
    title        = {{GlotLID: Language Identification for Low-Resource Languages}},
    author       = {Kargaran, Amir Hossein and Imani, Ayyoob and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
    year         = 2023,
    booktitle    = {The 2023 Conference on Empirical Methods in Natural Language Processing},
    url          = {https://openreview.net/forum?id=dl4e3EBz5j}
}
"""

_DATASETNAME = "glotstorybook"
_DESCRIPTION = """\
The GlotStoryBook dataset is a compilation of children's storybooks from the Global
Storybooks project, encompassing 174 languages organized for machine translation tasks. It
features rows containing the text segment (text number), the language code, and the file
name, which corresponds to the specific book and story segment. This structure allows for
the comparison of texts across different languages by matching file names and text numbers
between rows.
"""

_HOMEPAGE = "https://huggingface.co/datasets/cis-lmu/GlotStoryBook"
_LICENSE = f"""{Licenses.OTHERS.value} | \
We do not own any of the text from which these data has been extracted. All the files are
collected from the repository located at https://github.com/global-asp/. The source
repository for each text and file is stored in the dataset. Each file in the dataset is
associated with one license from the CC family. The licenses include 'CC BY', 'CC BY-NC',
'CC BY-NC-SA', 'CC-BY', 'CC-BY-NC', and 'Public Domain'. We also license the code, actual
packaging and the metadata of these data under the cc0-1.0.
"""

_LOCAL=False
_LANGUAGES = ["khg", "khm", "mya", "tet", "tha", "vie"]

_URLS = "https://huggingface.co/datasets/cis-lmu/GlotStoryBook/resolve/main/GlotStoryBook.csv"

_SUPPORTED_TASKS = [Tasks.SELF_SUPERVISED_PRETRAINING]
_SOURCE_VERSION = "1.0.0"
_SEACROWD_VERSION = "2024.06.20"


class GlotStoryBookDataset(datasets.GeneratorBasedBuilder):
    """Compilation of storybooks from the Global Storybooks project"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    SEACROWD_VERSION = datasets.Version(_SEACROWD_VERSION)

    SEACROWD_SCHEMA_NAME = TASK_TO_SCHEMA[_SUPPORTED_TASKS[0]].lower()

    BUILDER_CONFIGS = [
        SEACrowdConfig(
            name=f"{_DATASETNAME}_source",
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema="source",
            subset_id=f"{_DATASETNAME}",
        ),
        SEACrowdConfig(
            name=f"{_DATASETNAME}_seacrowd_{SEACROWD_SCHEMA_NAME}",
            version=SEACROWD_VERSION,
            description=f"{_DATASETNAME} SEACrowd schema",
            schema=f"seacrowd_{SEACROWD_SCHEMA_NAME}",
            subset_id=f"{_DATASETNAME}",
        ),
    ]

    DEFAULT_CONFIG_NAME = f"{_DATASETNAME}_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "text_number": datasets.Value("int64"),
                    "license": datasets.Value("string"),
                    "text_by": datasets.Value("string"),
                    "translation_by": datasets.Value("string"),
                    "language": datasets.Value("string"),
                    "file_name": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "iso639-3": datasets.Value("string"),
                    "script": datasets.Value("string"),
                }
            )
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            features = SCHEMA_TO_FEATURES[self.SEACROWD_SCHEMA_NAME.upper()]

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data_path = Path(dl_manager.download_and_extract(_URLS))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath: Path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        df = pd.read_csv(filepath)
        df = df[df["ISO639-3"].isin(_LANGUAGES)]

        if self.config.schema == "source":
            for i, row in df.iterrows():
                yield i, {
                    "text": row["Text"],
                    "text_number": row["Text Number"],
                    "license": row["License"],
                    "text_by": row["Text By"],
                    "translation_by": row["Translation By"],
                    "language": row["Language"],
                    "file_name": row["File Name"],
                    "source": row["Source"],
                    "iso639-3": row["ISO639-3"],
                    "script": row["Script"],
                }
        elif self.config.schema == f"seacrowd_{self.SEACROWD_SCHEMA_NAME}":
            df = df.sort_values(by=["ISO639-3", "Source", "File Name", "Text Number"])
            df = df.groupby(["ISO639-3", "Source", "File Name"]).agg({"Text": " ".join}).reset_index()
            for i, row in df.iterrows():
                yield i, {
                    "id": str(i),
                    "text": row["Text"],
                }